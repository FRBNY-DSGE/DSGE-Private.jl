"""
```
measurement(m::SmetsWouters{T}, TTT::Matrix{T}, RRR::Matrix{T},
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
function measurement(m::SmetsWouters{T},
                     TTT::Matrix{T},
                     RRR::Matrix{T},
                     CCC::Vector{T}; reg::Int = 1) where {T<:AbstractFloat}
    endo      = m.endogenous_states
    endo_addl = m.endogenous_states_augmented
    exo       = m.exogenous_shocks
    obs       = m.observables

    _n_observables = n_observables(m)
    _n_states = n_states_augmented(m)
    _n_shocks_exogenous = n_shocks_exogenous(m)

    ZZ = zeros(_n_observables, _n_states)
    DD = zeros(_n_observables)
    EE = zeros(_n_observables, _n_observables)
    QQ = zeros(_n_shocks_exogenous, _n_shocks_exogenous)
    
    _n_coint = 0
    _n_lags = 0

    for para in m.parameters
        if !isempty(para.regimes)
            if (haskey(get_settings(m), :model2para_regime) ? haskey(get_setting(m, :model2para_regime), para.key) : false)
                ModelConstructors.toggle_regime!(para, reg, get_setting(m, :model2para_regime)[para.key])
            else
                ModelConstructors.toggle_regime!(para, reg)
            end
        end
    end


    ## Output growth - Quarterly!
    ZZ[obs[:obs_gdp], endo[:y_t]]       = 1.0
    ZZ[obs[:obs_gdp], endo_addl[:y_t1]] = -1.0
    ZZ[obs[:obs_gdp], endo[:z_t]]       = 1.0
    #DD[obs[:obs_gdp]]                   = 100*(exp(m[:zstar])-1)
    DD[obs[:obs_gdp]]                   = 100 * (m[:gam] + m[:alp] * log(m[:ups])/(1-m[:alp])) 

    ## Consumption Growth
    ZZ[obs[:obs_consumption], endo[:c_t]]       = 1.0
    ZZ[obs[:obs_consumption], endo_addl[:c_t1]] = -1.0
    ZZ[obs[:obs_consumption], endo[:z_t]]       = 1.0
    DD[obs[:obs_consumption]]                   = 100*(m[:gam] + m[:alp] * log(m[:ups])/(1-m[:alp]))

    ## Investment Growth
    ZZ[obs[:obs_investment], endo[:i_t]]       = 1.0
    ZZ[obs[:obs_investment], endo_addl[:i_t1]] = -1.0
    ZZ[obs[:obs_investment], endo[:z_t]]       = 1.0
    DD[obs[:obs_investment]]                   = 100*(m[:gam] + m[:alp] * log(m[:ups])/(1-m[:alp]))


    ## Hours growth
    ZZ[obs[:obs_hours], endo[:L_t]] = 1/100
    DD[obs[:obs_hours]]             = m[:Ladj] + log(m[:Lstar])

    ## Labor Share/real wage growth
    ZZ[obs[:obs_wages], endo[:w_t]]       = 1.0
    ZZ[obs[:obs_wages], endo_addl[:w_t1]] = -1.0
    ZZ[obs[:obs_wages], endo[:z_t]]       = 1.0
    DD[obs[:obs_wages]]                   = 100*(m[:gam] + m[:alp] * log(m[:ups])/(1-m[:alp]))

    ## Inflation (GDP Deflator)
    ZZ[obs[:obs_gdpdeflator], endo[:pistar]]  = 1.0
    DD[obs[:obs_gdpdeflator]]              = log(m[:pistar])

    ## Nominal interest rate
    ZZ[obs[:obs_nominalrate], endo[:R_t]]       = 1.0
    DD[obs[:obs_nominalrate]]                   = m[:Rstarn]

    if n_coint > 0
        #consumption - output cointegration
        ZZ[obs[:cons_coint], endo[:c_t]] = 1
        ZZ[obs[:cons_coint], endo[:y_t]] = -1
        DD[obs[:cons_coint]] = 100 * log(m[:cstar] / m[:ystar])

        #investment - output
        ZZ[obs[:investment_coint], endo[:i_t]] = 1
        ZZ[obs[:investment_coint], endo[:y_t]] = -1
        DD[obs[:investment_coint]] = 100 * log(m[:istar]/m[:ystar])

        #real wage - output
        ZZ[obs[:wage_coint], endo[:w_t]] = 1
        ZZ[obs[:wage_coint], endo[:y_t]] = -1
        DD[obs[:wage_coint]] = 100 * log(m[:wstar] / m[:ystar]- m[:wadj])
    end


    #Measurement error
    #=
    EE[obs[:obs_gdp],1] = m[:e_y]^2
    EE[obs[:obs_hours],2] = m[:e_L]^2
    EE[obs[:obs_wages],3] = m[:e_w]^2
    EE[obs[:obs_gdpdeflator],4] = m[:e_π]^2
    EE[obs[:obs_nominalrate],5] = m[:e_R]^2
    EE[obs[:obs_consumption],6] = m[:e_c]^2
    EE[obs[:obs_investment],7] = m[:e_i]^2
=#

    ## Measurement error variances (aligned with observables)

    EE[obs[:obs_gdp],          obs[:obs_gdp]]          = m[:e_y]^2
    EE[obs[:obs_hours],        obs[:obs_hours]]        = m[:e_L]^2
    EE[obs[:obs_wages],        obs[:obs_wages]]        = m[:e_w]^2
    EE[obs[:obs_gdpdeflator],  obs[:obs_gdpdeflator]]  = m[:e_pi]^2
    EE[obs[:obs_nominalrate],  obs[:obs_nominalrate]]  = m[:e_R]^2
    EE[obs[:obs_consumption],  obs[:obs_consumption]]  = m[:e_c]^2
    EE[obs[:obs_investment],   obs[:obs_investment]]   = m[:e_i]^2


    #Variance of innovations
    QQ[exo[:g_sh], exo[:g_sh]]           = m[:sig_g]^2
    QQ[exo[:b_sh], exo[:b_sh]]           = m[:sig_b]^2
    QQ[exo[:μ_sh], exo[:μ_sh]]           = m[:sig_mu]^2
    QQ[exo[:z_sh], exo[:z_sh]]           = m[:sig_z]^2
    QQ[exo[:laf_sh], exo[:laf_sh]]       = m[:sig_laf]^2
    QQ[exo[:r_sh], exo[:r_sh]]         = m[:sig_rm]^2

    # These lines set the standard deviations for the anticipated
    # shocks to be equal to the standard deviation for the
    # unanticipated policy shock
    for i = 1:n_anticipated_shocks(m)
        ZZ[obs[Symbol("obs_nominalrate$i")], :] = ZZ[obs[:obs_nominalrate], :]' * (TTT^i)
        DD[obs[Symbol("obs_nominalrate$i")]]    = m[:Rstarn]
        if subspec(m) == "ss1"
            QQ[exo[Symbol("rm_shl$i")], exo[Symbol("rm_shl$i")]] = m[Symbol("sig_rm")]^2 / n_anticipated_shocks(m)
        else
            QQ[exo[Symbol("rm_shl$i")], exo[Symbol("rm_shl$i")]] = m[Symbol("sig_rm")]^2 / 16
        end
    end

    # Adjustment to DD because measurement equation assumes CCC is the zero vector
    if any(CCC .!= 0)
        DD += ZZ*((UniformScaling(1) - TTT)\CCC)
    end

    for para in m.parameters
        if !isempty(para.regimes)
            ModelConstructors.toggle_regime!(para, 1)
        end
    end

    return Measurement(ZZ, DD, QQ, EE)
end
