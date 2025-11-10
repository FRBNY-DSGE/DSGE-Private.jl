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
function measurement(m::DSSW{T},
                     TTT::Matrix{T},
                     RRR::Matrix{T},
                     CCC::Vector{T}; reg::Int = 1) where {T<:AbstractFloat}

    endo      = m.endogenous_states
    endo_addl = m.endogenous_states_augmented
    exo       = m.exogenous_shocks
    obs       = m.observables

    n_observables = length(keys(obs))
    n_states = length(keys(endo))
    n_shocks_exogenous = length(keys(exo))
    n_endo_addl = length(keys(endo_addl))

    n_coint = get_setting(m, :n_coint)
    n_lags = get_setting(m, :lags)

    ZZ = zeros(n_observables, n_states + n_endo_addl)
    DD = zeros(n_observables)
    EE = zeros(n_observables, n_observables)
    QQ = zeros(n_shocks_exogenous, n_shocks_exogenous)


    # Output growth - Quarterly!
    ZZ[obs[:output_growth], endo[:y_t]]       = 1.0
    ZZ[obs[:output_growth], endo_addl[:y_t1]] = -1.0
    ZZ[obs[:output_growth], endo[:z_t]]       = 1.0
    DD[obs[:output_growth]]                   = 100 * (m[:gam] + (m[:alp] * log(m[:ups])/(1-m[:alp])))

    # Consumption Growth
    ZZ[obs[:consumption], endo[:c_t]]       = 1.0
    ZZ[obs[:consumption], endo_addl[:c_t1]] = -1.0
    ZZ[obs[:consumption], endo[:z_t]]       = 1.0
    DD[obs[:consumption]]                   = 100 * (m[:gam] + (m[:alp] * log(m[:ups])/(1-m[:alp])))

    # Investment Growth
    ZZ[obs[:investment_growth], endo[:i_t]]       = 1.0
    ZZ[obs[:investment_growth], endo_addl[:i_t1]] = -1.0
    ZZ[obs[:investment_growth], endo[:z_t]]       = 1.0
    DD[obs[:investment_growth]]                   = 100 * (m[:gam] + (m[:alp] * log(m[:ups])/(1-m[:alp])))


    # Hours growth
    ZZ[obs[:hours], endo[:L_t]] = 1/100
    DD[obs[:hours]]             = m[:Ladj] + log(m[:Lstar])

    # wage growth
    ZZ[obs[:wage_growth], endo[:w_t]]       = 1.0
    ZZ[obs[:wage_growth], endo_addl[:w_t1]] = -1.0
    ZZ[obs[:wage_growth], endo[:z_t]]       = 1.0
    DD[obs[:wage_growth]]                   = 100*(m[:gam] + (m[:alp] * log(m[:ups])/(1-m[:alp])))


    ## Inflation
    ZZ[obs[:inflation], endo[:pi_t]]  = 1.0
    DD[obs[:inflation]]              = 100* log(m[:pistar])

    ## Nominal interest rate
    ZZ[obs[:nominal_rate], endo[:R_t]]       = 4.
    DD[obs[:nominal_rate]]                   = 400*log(m[:Rstarn])


    if n_coint > 0
        #consumption - output cointegration
        ZZ[obs[:consumption_coint], endo[:c_t]] = 1
        ZZ[obs[:consumption_coint], endo[:y_t]] = -1
        DD[obs[:consumption_coint]] = 100 * log(m[:cstar] / m[:ystar])

        #investment - output
        ZZ[obs[:investment_coint], endo[:i_t]] = 1
        ZZ[obs[:investment_coint], endo[:y_t]] = -1
        DD[obs[:investment_coint]] = 100 * log(m[:istar]/m[:ystar])

        #real wage - output
        ZZ[obs[:wage_coint], endo[:w_t]] = 1
        ZZ[obs[:wage_coint], endo[:y_t]] = -1
        DD[obs[:wage_coint]] = 100 * (log(m[:wstar] / m[:ystar])- m[:wadj])
    end

    #Variance of innovations
    QQ[exo[:g_sh], exo[:g_sh]]           = m[:sig_g]^2
    QQ[exo[:b_sh], exo[:b_sh]]           = m[:sig_b]^2
    QQ[exo[:mu_sh], exo[:mu_sh]]           = m[:sig_mu]^2
    QQ[exo[:z_sh], exo[:z_sh]]           = m[:sig_z]^2
    QQ[exo[:phi_sh], exo[:phi_sh]]           = m[:sig_phi]^2
    QQ[exo[:laf_sh], exo[:laf_sh]]       = m[:sig_laf]^2
    QQ[exo[:r_sh], exo[:r_sh]]         = m[:sig_r]^2

    for para in m.parameters
        if !isempty(para.regimes)
            ModelConstructors.toggle_regime!(para, 1)
        end
    end

    return Measurement(ZZ, DD, QQ, EE)
end
