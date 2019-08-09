"""
```
measurement(m::GHLS{T})
```

Assign measurement equation

```
y_t = Ψ(s_t) + u_t
```
"""
function measurement(m::GHLS)
    endo      = m.endogenous_states
    endo_addl = m.endogenous_states_augmented
    exo       = m.exogenous_shocks
    obs       = m.observables

    _n_observables = n_observables(m)
    _n_states = n_states_augmented(m)
    _n_shocks_exogenous = n_shocks_exogenous(m)

    @inline Ψ(s_t::Vector{Float64}) =
[log(s_t[endo[:y_t]] * m[:gz] / s_t[endo_addl[:y_t1]]) + s_t[endo[:z_t]],
log(s_t[endo[:c_t]] * m[:gz] / s_t[endo_addl[:c_t1]]) + s_t[endo[:z_t]],
log(s_t[endo[:i_t]] * m[:gz] / s_t[endo_addl[:i_t1]]) + s_t[endo[:z_t]],
log(s_t[endo[:π_t]]),
log(s_t[endo[:Rnominal_t]])]

    return Ψ
end
