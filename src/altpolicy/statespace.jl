"""
```
reset_historical_indices!(m)
```

Revert the `:alternative_policy` setting and the equation/state dictionaries to
their values under the historical rule. (These might still be the same as the
dictionaries under the alternative rule.)
"""
function reset_historical_rule_indices!(m::AbstractModel)
    # Get optimal policy settings
    alt_eqs    = m.equilibrium_conditions
    alt_states = m.endogenous_states

    # Restore original state and equation dictionaries
    if haskey(m.settings, :original_order_eqs)
        m.equilibrium_conditions = get_setting(m, :original_order_eqs)
    end
    if haskey(m.settings, :original_order_states)
        m.endogenous_states = get_setting(m, :original_order_states)
    end
    m <= Setting(:altpol_reordered, false)

    return alt_eqs, alt_states
end

"""
```
altpol_to_historical!(m)
```

Call `reset_historical_indices!` and restore the historical `AltPolicy`.  Return
the alternative `AltPolicy` instance, equations dictionary, and states
dictionary.
"""
function altpol_to_historical!(m::AbstractModel)
    # Reset state and equation dictionaries
    alt_eqs, alt_states = reset_historical_rule_indices!(m)

    # Revert to historical rule
    altpol = alternative_policy(m)
    m <= Setting(:alternative_policy, AltPolicy(:historical, eqcond, solve))

    return altpol, alt_eqs, alt_states
end

"""
```
historical_to_altpol!(m, altpol, eqs_alt, states_alt)
```

Update the model `m` with `altpol` and the equation and state dictionaries
`eqs_alt` and `states_alt`, all returned from `altpol_to_historical!`.
"""
function historical_to_altpol!(m::AbstractModel, altpol::AltPolicy,
                                eqs_alt::OrderedDict{Symbol, Int},
                                states_alt::OrderedDict{Symbol, Int})
    m <= Setting(:alternative_policy, altpol)
    m.equilibrium_conditions = eqs_alt
    m.endogenous_states      = states_alt
    return nothing
end

"""
```
historical_state_indices(m)
```

Return the indices of the *historical policy* states into the *alternative
policy* state vector. This function is nontrivial only if additional states are
added when forecasting under the alternative rule.
"""
function historical_state_indices(m::AbstractModel; augmented::Bool = true)
    if haskey(m.settings, :original_order_states)
        addl_states = augmented ? m.endogenous_states_augmented : OrderedDict{Symbol, Int}()
        old_states  = merge(get_setting(m, :original_order_states), addl_states)
        new_states  = merge(m.endogenous_states, addl_states)
        new_inds    = map(var -> new_states[var], keys(old_states))
        return new_inds
    else
        if augmented
            return collect(1:n_states_augmented(m))
        else
            return collect(1:n_states(m))
        end
    end
end

"""
```
historical_system(m, system)
```

Given `system`, return another `System` containing only the states found under
the historical policy rule.
"""
function historical_system(m::AbstractModel, system::System)
    new_inds = historical_state_indices(m)

    TTT = system[:TTT][new_inds, new_inds]
    RRR = system[:RRR][new_inds, :]
    CCC = system[:CCC][new_inds]
    trans = Transition(TTT, RRR, CCC)

    ZZ = system[:ZZ][:, new_inds]
    DD = copy(system[:DD])
    QQ = copy(system[:QQ])
    EE = copy(system[:EE])
    meas = Measurement(ZZ, DD, QQ, EE)

    ZZ_pseudo = system[:ZZ_pseudo][:, new_inds]
    DD_pseudo = copy(system[:DD_pseudo])
    pseudo = PseudoMeasurement(ZZ_pseudo, DD_pseudo)

    return System(trans, meas, pseudo)
end

"""
```
augment_states_altpolicy!(m, TTT, RRR, CCC)
```

This function calls `augment_states`, inserting the states in
`m.endogenous_states_augmented` *between* the original states y_t and any new
states that might have been added in order to define the alternative policy
rule. The indices of these new states in `m.endogenous_states` are shifted down
accordingly. Finally, the augmented transition matrices are returned.
"""
function augment_states_altpolicy(m::AbstractModel, TTT::Matrix{Float64},
                                  RRR::Matrix{Float64}, CCC::Vector{Float64})
    # Indices and numbers
    old_inds = historical_state_indices(m, augmented = false)
    new_inds = setdiff(1:n_states(m), old_inds)

    nstates_old  = length(old_inds)
    nstates_new  = length(new_inds)
    nstates_addl = length(m.endogenous_states_augmented)

    # Divide TTT matrix into quadrants:
    #   1. top left:     TTT matrix just for the normal m.endogenous_states
    #   2. top right:    new states entering into regular states
    #   3. bottom left:  regular states entering into new equations
    #   4. bottom right: new state-only transition matrix
    TTT_old = TTT[old_inds, old_inds]
    TTT12_new = TTT[old_inds, new_inds]
    TTT21_new = TTT[new_inds, old_inds]
    TTT22_new = TTT[new_inds, new_inds]

    # RRR matrix just for the normal m.endogenous_states
    RRR_old  = RRR[old_inds,:]
    RRR2_new   = RRR[new_inds,:]

    # CCC matrix just for the normal m.endogenous_states
    CCC_old = CCC[old_inds]
    CCC2_new  = CCC[new_inds]

    # Augment states on the old state space system. This requires temporarily
    # switching back to the cached original state dictionary so that
    # augment_states is happy
    alt_eqs, alt_states = reset_historical_rule_indices!(m)
    TTT_aug, RRR_aug, CCC_aug = DSGE.augment_states(m, TTT_old, RRR_old, CCC_old)
    historical_to_altpol!(m, alternative_policy(m), alt_eqs, alt_states)

    # Add new states back to the augmented state space system
    TTT12_new_aug = [TTT12_new; zeros(nstates_addl, nstates_new)]
    TTT21_new_aug = [TTT21_new  zeros(nstates_new, nstates_addl)]
    TTT_all       = [TTT_aug        TTT12_new_aug;
                     TTT21_new_aug  TTT22_new]

    RRR_all = [RRR_aug; RRR2_new]

    CCC_all = [CCC_aug; CCC2_new]

    # Shift new state indices down by number of newly inserted states
    old_keys = keys(get_setting(m, :original_order_states))
    new_keys = setdiff(keys(m.endogenous_states), old_keys)
    for k in new_keys
        m.endogenous_states[k] += nstates_addl
    end

    # Return matrices
    return TTT_all, RRR_all, CCC_all
end