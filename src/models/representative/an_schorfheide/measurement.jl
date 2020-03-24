"""
```
measurement(m::AnSchorfheide{T}, TTT::Matrix{T},
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
function measurement(m::AnSchorfheide{S},
                     TTT::Matrix{T},
                     RRR::Matrix{T},
                     CCC::Vector{T};
                     zero_type = nothing) where {T <: Real, S <: Real}
    endo     = m.endogenous_states
    endo_new = m.endogenous_states_augmented
    exo      = m.exogenous_shocks
    obs      = m.observables

    _n_observables = n_observables(m)
    _n_states = n_states_augmented(m)
    if _n_states != size(TTT,1)
        try
            _n_states = get_setting(m, :n_endogenous_states_klein)
            _n_states == size(TTT, 1)
        catch
            error("Size of state space implied by TTT $(size(TTT,1)) does not match the model's information.")
        end
    end
    _n_shocks_exogenous = n_shocks_exogenous(m)

    if isnothing(zero_type)
        zero_type = eltype(TTT) <: ForwardDiff.Dual ? Real : eltype(TTT)
    end
    ZZ = zeros(zero_type, _n_observables, _n_states)
    DD = zeros(zero_type, _n_observables)
    EE = zeros(zero_type, _n_observables, _n_observables)
    QQ = zeros(zero_type, _n_shocks_exogenous, _n_shocks_exogenous)

    ## Output growth
    ZZ[obs[:obs_gdp], endo[:y_t]]  = 1.0
    ZZ[obs[:obs_gdp], endo[:y_t1]] = -1.0
    ZZ[obs[:obs_gdp], endo[:z_t]]  = 1.0
    DD[obs[:obs_gdp]]              = m[:γ_Q]

    ## Inflation
    ZZ[obs[:obs_cpi], endo[:π_t]] = 4.0
    DD[obs[:obs_cpi]]             = m[:π_star]

    ## Federal Funds Rate
    ZZ[obs[:obs_nominalrate], endo[:R_t]] = 4.0
    DD[obs[:obs_nominalrate]]             = m[:π_star] + m[:rA] + 4.0*m[:γ_Q]

    # Measurement error
    EE[obs[:obs_gdp], endo[:y_t]]         = m[:e_y]^2
    EE[obs[:obs_cpi], endo[:π_t]]         = m[:e_π]^2
    EE[obs[:obs_nominalrate], endo[:R_t]] = m[:e_R]^2

    # Variance of innovations
    if get_setting(m, :standard_normal_exog_shocks)
        QQ[exo[:z_sh],exo[:z_sh]]   = 1.
        QQ[exo[:g_sh],exo[:g_sh]]   = 1.
        QQ[exo[:rm_sh],exo[:rm_sh]] = 1.
    else
        QQ[exo[:z_sh],exo[:z_sh]]   = (m[:σ_z])^2
        QQ[exo[:g_sh],exo[:g_sh]]   = (m[:σ_g])^2
        QQ[exo[:rm_sh],exo[:rm_sh]] = (m[:σ_R])^2
    end
    return Measurement(ZZ, DD, QQ, EE)
end
