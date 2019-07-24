"""
```
measurement(m::GHLS{T}, TTT::Matrix{T}, RRR::Matrix{T},
            CCC::Vector{T}) where {T<:AbstractFloat}
```

Assign measurement equation

```
y_t = ZZ*s_t + DD + u_t
```

where

```
Var(ϵ_t) = QQ
Var(u_t) = EE
Cov(ϵ_t, u_t) = 0
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

    # We store the innovation as the last state since TPF isn't built to take in innovations as well as states
    @inline Ψ(s_t::Vector{Float64}) = [log(s_t[endo[:y_t]] * m[:gz] / s_t[endo_addl[:y_t1]]) + s_t[end],
(s_t[endo[:π_t]]-1.0)*100.0,
(s_t[endo[:R_t]]-1.0)*100.0,
#log(s_t[endo[:π_t]]),
#log(s_t[endo[:R_t]]),
log(s_t[endo[:c_t]] * m[:gz] / s_t[endo_addl[:c_t1]]) + s_t[end],
log(s_t[endo[:i_t]] * m[:gz] / s_t[endo_addl[:i_t1]]) + s_t[end]]

    return Ψ

end
