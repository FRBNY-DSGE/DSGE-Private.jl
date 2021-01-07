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
function measurement(m::AnSchorfheide{T},
                     TTT::Matrix{T},
                     RRR::Matrix{T},
                     CCC::Vector{T}; reg::Int = 1, information_set::UnitRange = reg:reg) where {T<:AbstractFloat}
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

    for para in m.parameters
        if !isempty(para.regimes)
            if (haskey(get_settings(m), :model2para_regime) ? haskey(get_setting(m, :model2para_regime), para.key) : false)
                ModelConstructors.toggle_regime!(para, reg, get_setting(m, :model2para_regime)[para.key])
            else
                ModelConstructors.toggle_regime!(para, reg)
            end
        end
    end
#=
    # Set up for calculating k-periods ahead expectations and expected sums
    permanent_t = length(information_set[findfirst(information_set .== reg):end]) - 1 + reg

    # Remove integrated states (e.g. states w/unit roots)
    no_integ_inds = inds_states_no_integ_series(m)
    if ((haskey(get_settings(m), :add_altpolicy_pgap) && haskey(get_settings(m), :pgap_type)) ?
        (get_setting(m, :add_altpolicy_pgap) && get_setting(m, :pgap_type) == :ngdp) : false) ||
        (haskey(get_settings(m), :ait_Thalf) ? (exp(log(0.5) / get_setting(m, :ait_Thalf)) ≈ 0.) : false)
        no_integ_inds = setdiff(no_integ_inds, [m.endogenous_states[:pgap_t]])
    end
    if (haskey(get_settings(m), :gdp_Thalf) ? (exp(log(0.5) / get_setting(m, :gdp_Thalf)) ≈ 0.) : false)
        no_integ_inds = setdiff(no_integ_inds, [m.endogenous_states[:ygap_t]])
    end
    if haskey(get_settings(m), :ρ_rw) ? (get_setting(m, :ρ_rw) ≈ 1.) : false
        no_integ_inds = setdiff(no_integ_inds, [m.endogenous_states[:rw_t]])
    end
    if haskey(get_settings(m), :rw_ρ_smooth) ? (get_setting(m, :rw_ρ_smooth) ≈ 1.) : false
        no_integ_inds = setdiff(no_integ_inds, [m.endogenous_states[:Rref_t]])
    end
    integ_series = length(no_integ_inds) != n_states_augmented(m) # Are the series integrated?
=#
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
    EE[obs[:obs_gdp], obs[:obs_gdp]]                 = m[:e_y]^2
    EE[obs[:obs_cpi], obs[:obs_cpi]]                 = m[:e_π]^2
    EE[obs[:obs_nominalrate], obs[:obs_nominalrate]] = m[:e_R]^2

    # Variance of innovations
    QQ[exo[:z_sh],exo[:z_sh]]   = (m[:σ_z])^2
    QQ[exo[:g_sh],exo[:g_sh]]   = (m[:σ_g])^2
    QQ[exo[:rm_sh],exo[:rm_sh]] = (m[:σ_R])^2

    for para in m.parameters
        if !isempty(para.regimes)
            ModelConstructors.toggle_regime!(para, 1)
        end
    end

    return Measurement(ZZ, DD, QQ, EE)
end
