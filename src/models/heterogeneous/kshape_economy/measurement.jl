"""
```
measurement(m::BayerBornLuetticke{T},
            TTT::AbstractMatrix{T},
            TTT_jump::AbstractMatrix{T},
            RRR::AbstractMatrix{T},
            CCC::AbstractVector{T}) where {T<:Real}
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
function measurement(m::mBBQ{T},
                     TTT::AbstractMatrix{T},
                     TTT_jump::AbstractMatrix{T},
                     RRR::AbstractMatrix{T},
                     CCC::AbstractVector{T}) where {T<:Real}

# TODO: this implementation of measurement equation assumes that
#       the loaded data has already been demeaned by the
#       average across time, hence we set DD = 0 in this function

    endo      = m.endogenous_states
    # endo_new  = m.endogenous_states_augmented
    exo       = m.exogenous_shocks
    obs       = m.observables

    _n_model_states = get_setting(m, :n_model_states_augmented)
    _n_states = n_backward_looking_states(m)
    _n_jumps = n_jumps(m)

    _n_observables = n_observables(m)
    _n_shocks_exogenous = n_shocks_exogenous(m)

    track_states_only = haskey(get_settings(m), :klein_track_backward_looking_states_only) &&
        get_setting(m, :klein_track_backward_looking_states_only)
    _ZZ = track_states_only ? spzeros(_n_observables, _n_model_states) : zeros(_n_observables, _n_model_states) # _ZZ is a selection matrix
    DD  = zeros(_n_observables)
    EE  = zeros(_n_observables, _n_observables)
    QQ  = zeros(_n_shocks_exogenous, _n_shocks_exogenous)

    ## Measurement equation: states to observables
    _ZZ[obs[:obs_gdp], first(endo[:Ygrowth′_t])] = 1.0

    _ZZ[obs[:obs_consumption], first(endo[:Cgrowth′_t])] = 1.0

    _ZZ[obs[:obs_investment], first(endo[:Igrowth′_t])] = 1.0

    _ZZ[obs[:obs_nominalrate], first(endo[:R_cb′_t])] = 1.0 #or RB′_t or RR′_t?

    _ZZ[obs[:obs_wages], first(endo[:wgrowth′_t])] = 1.0

    _ZZ[obs[:obs_unemployment_rate], first(endo[:unemp′_t])] = 1.0

    _ZZ[obs[:obs_lumpsum_transfer], first(endo[:LT′_t])] = 1.0

    _ZZ[obs[:obs_profit], first(endo[:Profit′_t])] = 1.0 # or Profit_FI′_t?

    _ZZ[obs[:obs_central_bank_assets], first(endo[:x_cb′_t])] = 1.0

    _ZZ[obs[:obs_stock_returns], first(endo[:r_a′_t])] = 1.0

    ZZ = if track_states_only
        # Construct measurement matrix from selection matrix, using
        # BlockArrays.jl and BandedMatrices.jl to efficiently construct
        # vcat(I, TTT_jump), so that y = _ZZ * [I; TTT_jump] * states
        MX = PseudoBlockArray{T}(undef, [_n_states, size(TTT_jump, 1)], [_n_states]) # use Pseudo b/c multiplication is faster
        setblock!(MX, Diagonal(Ones(_n_states)), 1, 1)
        setblock!(MX, TTT_jump, 2, 1)
        Array(_ZZ * MX)
    else
        _ZZ # ZZ is the selection matrix when we treat both states and jumps as model states
    end

    ## Measurement error disabled: EE remains zero matrix.

    ## Variance of innovations (mBBQ shock set)
    QQ[exo[:Z_sh], exo[:Z_sh]] = m[:σ_Z]^2
    QQ[exo[:G_sh], exo[:G_sh]] = m[:σ_G]^2
    QQ[exo[:D_sh], exo[:D_sh]] = m[:σ_D]^2
    QQ[exo[:R_sh], exo[:R_sh]] = m[:σ_R]^2
    QQ[exo[:ι_sh], exo[:ι_sh]] = m[:σ_ι]^2
    QQ[exo[:η_sh], exo[:η_sh]] = m[:σ_η]^2
    QQ[exo[:w_sh], exo[:w_sh]] = m[:σ_w]^2

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
