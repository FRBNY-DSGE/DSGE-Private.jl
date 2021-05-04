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
function measurement(m::BayerBornLuetticke{T},
                     TTT::AbstratMatrix{T},
                     RRR::AbstractMatrix{T},
                     CCC::AbstractVector{T}) where {T<:Real}

# TODO: this implementation of measurement equation assumes that
#       the loaded data has already been demeaned by the
#       average across time, hence we set DD = 0 in this function

    # endo      = m.endogenous_states
    endo      = get_aggregate_endogenous_states(m)
    # endo_new  = m.endogenous_states_augmented
    exo       = m.exogenous_shocks
    obs       = m.observables

    _n_model_states = get_setting(m, :n_model_states_augmented)
    _n_states = n_backward_looking_states(m)
    _n_jumps = n_jumps(m)

    _n_observables = n_observables(m)
    _n_shocks_exogenous = n_shocks_exogenous(m)

    ZZ = zeros(_n_observables, _n_model_states)
    DD = zeros(_n_observables)
    EE = zeros(_n_observables, _n_observables)
    QQ = zeros(_n_shocks_exogenous, _n_shocks_exogenous)

    ## Measurement equation: states to observables

    # GDP growth per capita
    ZZ[obs[:obs_gdp], endo[:Ygrowth′_t]] = 1.0

    # Consumption growth per capita
    ZZ[obs[:obs_consumption], endo[:Cgrowth′_t]] = 1.0

    # Investment growth per capita
    ZZ[obs[:obs_investment], [:Igrowth′_t]] = 1.0

    # Wage growth
    ZZ[obs[:obs_wages], endo[:wgrowth′_t]] = 1.0

    # Hours
    ZZ[obs[:obs_hours], endo[:N′_t]] = 1.0

    # GDP Deflator inflation
    ZZ[obs[:obs_gdpdeflator], endo[:π′_t]] = 1.0

    # Nominal interest rate
    ZZ[obs[:obs_nominalrate], endo[:RB′_t]] = 1.0

    # Wealth inequality
    ZZ[obs[:obs_W90share], endo[:W90_share′_t]] = 1.0

    # Income inequality
    ZZ[obs[:obs_I90share], endo[:I90_share′_t]] = 1.0

    # Idiosyncratic income risk
    ZZ[obs[:obs_sigmasq], endo[:σ′_t]] = 1.0

    # Idiosyncratic income risk
    ZZ[obs[:obs_taxprogressivity], endo[:τ_prog′_t]] = 1.0

    ## Measurement error
    EE[obs[:obs_W90share], obs[:obs_W90share]] = m[:e_W90_share]^2
    EE[obs[:obs_I90share], obs[:obs_I90share]] = m[:e_I90_share]^2
    EE[obs[:obs_taxprogressivity], obs[:obs_taxprogressivity]] = m[:e_τ_prog]^2
    EE[obs[:obs_sigmasq], obs[:obs_sigmasq]] = m[:e_σ]^2

    ## Variance of innovations
    QQ[exo[:A_sh], exo[:A_sh]]     = m[:σ_A]^2
    QQ[exo[:Z_sh], exo[:Z_sh]]     = m[:σ_Z]^2
    QQ[exo[:Ψ_sh], exo[:Ψ_sh]]     = m[:σ_Ψ]^2
    QQ[exo[:μ_p_sh], exo[:μ_p_sh]] = m[:σ_μ_p]^2
    QQ[exo[:μ_w_sh], exo[:μ_w_sh]] = m[:σ_μ_w]^2
    QQ[exo[:G_sh], exo[:G_sh]]     = m[:σ_G]^2
    QQ[exo[:R_sh], exo[:R_sh]]     = m[:σ_R]^2
    QQ[exo[:S_sh], exo[:S_sh]]     = m[:σ_G]^2
    QQ[exo[:P_sh], exo[:P_sh]]     = m[:σ_P]^2

  #=  # These lines set the standard deviations for the anticipated
    # shocks to be equal to the standard deviation for the
    # unanticipated policy shock
    for i = 1:n_anticipated_shocks(m)
        ZZ[obs[Symbol("obs_nominalrate$i")], :] = ZZ[obs[:obs_nominalrate], :]' * (TTT^i)
        DD[obs[Symbol("obs_nominalrate$i")]]    = m[:Rstarn]
        if subspec(m) == "ss1"
            QQ[exo[Symbol("rm_shl$i")], exo[Symbol("rm_shl$i")]] = m[Symbol("σ_rm")]^2 / n_anticipated_shocks(m)
        else
            QQ[exo[Symbol("rm_shl$i")], exo[Symbol("rm_shl$i")]] = m[Symbol("σ_rm")]^2 / 16
        end
    end
=#
    # Adjustment to DD because measurement equation assumes CCC is the zero vector
    if any(CCC .!= 0)
        DD .+= ZZ*((I - TTT) \ CCC)
    end

    return Measurement(ZZ, DD, QQ, EE)
end
