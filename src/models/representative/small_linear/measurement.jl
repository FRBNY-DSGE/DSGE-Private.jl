"""
```
measurement(m::SmallLinear{T}, TTT::Matrix{T},
            RRR::Matrix{T}, CCC::Vector{T}) where {T<:AbstractFloat}
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
function measurement(m::SmallLinear{T},
                     TTT::Matrix{T},
                     RRR::Matrix{T},
                     CCC::Vector{T}) where {T<:AbstractFloat}
    endo     = m.endogenous_states
    endo_new = m.endogenous_states_augmented
    exo      = m.exogenous_shocks
    obs      = m.observables

    _n_observables = n_observables(m)
    _n_states = n_states_augmented(m)
    _n_shocks_exogenous = n_shocks_exogenous(m)

    ZZ = zeros(_n_observables, _n_states)
    DD = zeros(_n_observables)
    EE = zeros(_n_observables, _n_observables)
    QQ = zeros(_n_shocks_exogenous, _n_shocks_exogenous)

    # Variance of innovations
    QQ[exo[:rm_sh],exo[:rm_sh]]   = (m[:σ_m])^2
    QQ[exo[:rm_f_sh],exo[:rm_f_sh]]   = (m[:σ_m])^2
    QQ[exo[:z_sh],exo[:z_sh]]   = (m[:σ_z])^2
    QQ[exo[:z_f_sh],exo[:z_f_sh]]   = (m[:σ_z])^2
    QQ[exo[:uip_sh],exo[:uip_sh]]   = (m[:σ_uip])^2

    ## Output growth domestic
    ZZ[obs[:obs_gdp], endo[:y_t]]  = 1.0
    ZZ[obs[:obs_gdp], endo[:y_t1]] = -1.0
    ZZ[obs[:obs_gdp], endo[:z_t]]  = 1.0
    DD[obs[:obs_gdp]]              = m[:γ_Q]

    ## Inflation domestic
    ZZ[obs[:obs_cpi], endo[:π_t]] = 4.0
    DD[obs[:obs_cpi]]             = m[:π_star]

    ## Federal Funds Rate
    ZZ[obs[:obs_nominalrate], endo[:r_n_t]] = 4.0
    DD[obs[:obs_nominalrate]]             = m[:π_star] + m[:rA] + 4.0*m[:γ_Q]

    # Measurement error domestic
    EE[obs[:obs_gdp], obs[:obs_gdp]]         = m[:e_y]^2
    EE[obs[:obs_cpi], obs[:obs_cpi]]         = m[:e_π]^2
    EE[obs[:obs_nominalrate], obs[:obs_nominalrate]] = m[:e_r_n]^2

    ## Output growth foreign
    ZZ[obs[:obs_gdp_f], endo[:y_t_f]]  = 1.0
    ZZ[obs[:obs_gdp_f], endo[:y_t1_f]] = -1.0
    ZZ[obs[:obs_gdp_f], endo[:z_t_f]]  = 1.0
    DD[obs[:obs_gdp_f]]              = m[:γ_Q_f]

    ## Inflation foreign
    ZZ[obs[:obs_cpi_f], endo[:π_t_f]] = 4.0
    DD[obs[:obs_cpi_f]]             = m[:π_star_f]

    ## Discount Rate
    ZZ[obs[:obs_nominalrate_f], endo[:r_n_t_f]] = 4.0
    DD[obs[:obs_nominalrate_f]]             = m[:π_star_f] + m[:rA] + 4.0*m[:γ_Q_f]

    # Measurement error foreign
    EE[obs[:obs_gdp_f], obs[:obs_gdp_f]]         = m[:e_y_f]^2
    EE[obs[:obs_cpi_f], obs[:obs_cpi_f]]         = m[:e_π_f]^2
    EE[obs[:obs_nominalrate_f], obs[:obs_nominalrate_f]] = m[:e_r_n_f]^2

    return Measurement(ZZ, DD, QQ, EE)
end
