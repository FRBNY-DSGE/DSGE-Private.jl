"""
```
pseudo_measurement{T<:AbstractFloat}(m::Model904{T},
    TTT::Matrix{T}, RRR::Matrix{T}, CCC::Vector{T})
```

Assign pseudo-measurement equation (a linear combination of states):

```
x_t = ZZ_pseudo*s_t + DD_pseudo
```
"""
function pseudo_measurement(m::HetDSGEGovDebt{T},
                                              TTT::Matrix{T},
                                              RRR::Matrix{T},
                                              CCC::Vector{T}) where {T<:AbstractFloat}
    endo   = m.endogenous_states
    pseudo = m.pseudo_observables

    _n_states = get_setting(m, :n_model_states_augmented) #n_states_augmented(m)
    _n_pseudo = n_pseudo_observables(m)

    # Initialize pseudo ZZ and DD matrices
    ZZ_pseudo = zeros(_n_pseudo, _n_states)
    DD_pseudo = zeros(_n_pseudo)

    ##########################################################
    ## PSEUDO-OBSERVABLE EQUATIONS
    ##########################################################

    ## Output
    ZZ_pseudo[pseudo[:y_t], first(endo[:y′_t])] = 1.

    ## π_t
    ZZ_pseudo[pseudo[:π_t], first(endo[:π′_t])] = 1.
    DD_pseudo[pseudo[:π_t]]            = 100*(m[:π_star]-1)

    ## Long Run Inflation
    ZZ_pseudo[pseudo[:LongRunInflation], first(endo[:π_star′_t])] = 1.
    DD_pseudo[pseudo[:LongRunInflation]]                 = 100. *(m[:π_star]-1.)

    ## Hours
    ZZ_pseudo[pseudo[:Hours], first(endo[:L′_t])] = 1.

    ## z_t
    ZZ_pseudo[pseudo[:z_t], first(endo[:z′_t])] = 1.

    # Collect indices and transforms
    return PseudoMeasurement(ZZ_pseudo, DD_pseudo)
end
