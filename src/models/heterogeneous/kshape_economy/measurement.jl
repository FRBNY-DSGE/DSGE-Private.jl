"""
```
measurement(m::mBBQ{T}) where {T<:AbstractFloat}
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

ZZ is a selection matrix from the combined (state + control) vector produced by the
Klein/SGU solution. Must be called after `setup_indices!(m)`.

Data is assumed to be demeaned, so DD = 0.
"""
function measurement(m::mBBQ{T}) where {T<:AbstractFloat}
    endo = m.endogenous_states
    exo  = m.exogenous_shocks
    obs  = m.observables

    n_vars = get_setting(m, :n_model_states)
    n_obs  = n_observables(m)
    n_sh   = n_shocks_exogenous(m)

    ZZ = zeros(T, n_obs, n_vars)
    DD = zeros(T, n_obs)
    EE = zeros(T, n_obs, n_obs)
    QQ = zeros(T, n_sh,  n_sh)

    ## Measurement equation: states/controls → observables
    ZZ[obs[:obs_gdp],                first(endo[:Y′_t])]      = one(T)
    ZZ[obs[:obs_consumption],        first(endo[:C′_t])]      = one(T)
    ZZ[obs[:obs_investment],         first(endo[:I′_t])]      = one(T)
    ZZ[obs[:obs_nominalrate],        first(endo[:R_cb′_t])]   = one(T)
    ZZ[obs[:obs_wages],              first(endo[:w′′t])]      = one(T)
    ZZ[obs[:obs_unemployment_rate],  first(endo[:unemp′_t])]  = one(T)
    ZZ[obs[:obs_lumpsum_transfer],   first(endo[:LT′_t])]     = one(T)
    ZZ[obs[:obs_profit],             first(endo[:Profit′_t])] = one(T)
    ZZ[obs[:obs_central_bank_assets], first(endo[:x_cb′_t])] = one(T)
    ZZ[obs[:obs_stock_returns],      first(endo[:r_a′_t])]    = one(T)

    ## Measurement error: EE remains zero.

    ## Variance of innovations
    QQ[exo[:Z_sh], exo[:Z_sh]] = m[:σ_Z]^2
    QQ[exo[:G_sh], exo[:G_sh]] = m[:σ_G]^2
    QQ[exo[:D_sh], exo[:D_sh]] = m[:σ_D]^2
    QQ[exo[:R_sh], exo[:R_sh]] = m[:σ_R]^2
    QQ[exo[:ι_sh], exo[:ι_sh]] = m[:σ_ι]^2
    QQ[exo[:η_sh], exo[:η_sh]] = m[:σ_η]^2
    QQ[exo[:w_sh], exo[:w_sh]] = m[:σ_w]^2

    return Measurement(ZZ, DD, QQ, EE)
end
