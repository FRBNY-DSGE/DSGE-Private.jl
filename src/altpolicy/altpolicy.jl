"""
```
type AltPolicy
```

Type defining an alternative policy rule.

### Fields

- `key::Symbol`: alternative policy identifier

- `eqcond::Function`: a version of `DSGE.eqcond` which computes the equilibrium
  condition matrices under the alternative policy. Like `eqcond`, it should take
  in one argument of type `AbstractModel` and return the `Γ0`, `Γ1`, `C`, `Ψ`,
  and `Π` matrices.

- `solve::Function`: a version of `DSGE.solve` which solves the model under the
  alternative policy. Like `DSGE.solve`, it should take in one argument of type
  `AbstractModel` and return the `TTT`, `RRR`, and `CCC` matrices.

- `forecast_init::Function`: a function that initializes forecasts under the
  alternative policy rule. Specifically, it accepts a model, an `nshocks` x
  `n_forecast_periods` matrix of shocks to be applied in the forecast, and a
  vector of initial states for the forecast. It must return a new matrix of
  shocks and a new initial state vector. If no adjustments to shocks or initial
  state vectors are necessary under the policy rule, this field may be omitted.

- `color::Colorant`: color to plot this alternative policy in. Defaults to blue.

- `linestyle::Symbol`: line style for forecast plots under this alternative
  policy. See options from `Plots.jl`. Defaults to `:solid`.

"""
type AltPolicy
    key::Symbol
    eqcond::Function
    solve::Function
    forecast_init::Function
    color::Colorant
    linestyle::Symbol
end

function AltPolicy(key::Symbol, eqcond_fcn::Function, solve_fcn::Function;
                   forecast_init::Function = identity,
                   color::Colorant = RGB(0., 0., 1.),
                   linestyle::Symbol = :solid)

    AltPolicy(key, eqcond_fcn, solve_fcn, forecast_init, color, linestyle)
end

Base.string(a::AltPolicy) = string(a.key)

"""
```
altpol_to_historical!(m)
```

Revert the `:alternative_policy` setting and the equation/state dictionaries to
their values under the historical rule. (These might still be the same as the
dictionaries under the alternative rule.) Return the alternative `AltPolicy`
instance, equations dictionary, and states dictionary.
"""
function altpol_to_historical!(m::AbstractModel)
    # Get optimal policy settings
    altpol     = alternative_policy(m)
    alt_eqs    = m.equilibrium_conditions
    alt_states = m.endogenous_states

    # Revert to historical rule
    m <= Setting(:alternative_policy, AltPolicy(:historical, eqcond, solve))
    if haskey(m.settings, :original_order_eqs)
        m.equilibrium_conditions = get_setting(m, :original_order_eqs)
    end
    if haskey(m.settings, :original_order_states)
        m.endogenous_states = get_setting(m, :original_order_states)
    end

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
function historical_state_indices(m::AbstractModel)
    if haskey(m.settings, :original_order_states)
        old_states = merge(get_setting(m, :original_order_states), m.endogenous_states_augmented)
        new_states = merge(m.endogenous_states, m.endogenous_states_augmented)
        new_inds   = map(var -> new_states[var], keys(old_states))
        return new_inds
    else
        return collect(1:n_states_augmented(m))
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