function jacobian(m::HetDSGEGovDebt{S}) where {S <: Real}

    reset_grids!(m; init_grids = false) # to make this work with an adaptive agrid, we cannot recreate grid of steady-state approximation
    # truncate_distribution!(m)

    # Load in endogenous state and eqcond indices
    endo = augment_model_states(m.endogenous_states_original, # m.endogenous_states_unnormalized,
                                n_model_states_original(m))
    eq   = m.equilibrium_conditions
    # Load in parameters, steady-state parameters, and grids
    r::Float64     = m[:r].scaledvalue
    α::Float64     = m[:α].value
    H::Float64     = m[:H].value
    δ::Float64     = m[:δ].value
    μ_sp::Float64  = m[:μ_sp].value
#    ρ_sp::Float64  = m[:ρ_sp].value
#    σ_sp::Float64  = m[:σ_sp].value
    γ::Float64     = m[:γ].scaledvalue
    g::Float64     = m[:g].value
    η::Float64     = m[:η].value
    ρ_B::Float64    = m[:ρ_b].value
    ρ_G::Float64    = m[:ρ_g].value
    ρ_z::Float64    = m[:ρ_z].value
    ρ_μ::Float64    = m[:ρ_μ].value
    ρ_lamw::Float64 = m[:ρ_λ_w].value
    ρ_lamf::Float64 = m[:ρ_λ_f].value
    ρ_mon::Float64  = m[:ρ_rm].value
    ρ_π_star::Float64  = m[:ρ_π_star].value
    spp::Float64   = m[:spp].value
    ϕh::Float64    = m[:ϕh].value
    ρ_R::Float64    = m[:ρR].value
    ψπ::Float64    = m[:ψπ].value
    ψy::Float64    = m[:ψy].value
    κ_p::Float64  = m[:κ_p].value
    κ_w::Float64  = m[:κ_w].value

    Tg::Float64 = m[:Tg].value
    δb::Float64 = m[:δb].value
    bg::Float64 = m[:bg].value

    R = 1 + r

    ell::Vector{Float64}  = m[:lstar].value
    c::Vector{Float64}    = m[:cstar].value # Should be able to comment this out, reduce allocation
    D::Vector{Float64}    = m[:Dstar].value
    β::Float64            = m[:βstar].value

    T::Float64     = m[:Tstar].value
    ω::Float64     = m[:ωstar].value
    xstar::Float64 = m[:xstar].value
    ystar::Float64 = m[:ystar].value
    Rk::Float64    = m[:Rkstar].value
    kstar::Float64 = m[:kstar].value

    agrid::Vector{Float64} = m.grids[:agrid].points
    awts::Vector{Float64}  = m.grids[:agrid].weights
    sgrid::Vector{Float64} = m.grids[:sgrid].points
    swts::Vector{Float64}  = m.grids[:sgrid].weights
    fgrid::Matrix{Float64} = m.grids[:fgrid]

    elo::Float64 = m[:elo].value
    ehi::Float64 = m[:ehi].value

    na::Int = get_setting(m, :na)
    ns::Int = get_setting(m, :ns)
    nans = na*ns

    gfunc_type = get_setting(m, :gfunc_type)
    if gfunc_type == :mollifier
        gfunc_wbounds = (x, y, z) -> mollifier_hetdsgegovdebt(x, y, z) # need this for generate_us_and_es
        gpfunc_wbounds = (x, y, z) -> dmollifier_hetdsgegovdebt(x, y, z)
    elseif gfunc_type == :lognormal
        gfunc_wbounds = (x, y, z) -> lognormal_hetdsgegovdebt(x, y, z, m[:μ_e].value, m[:σ_e].value)
        gpfunc_wbounds = (x, y, z) -> dlognormal_hetdsgegovdebt(x, y, z, m[:μ_e].value, m[:σ_e].value)
    end

    gpfunc(x) = gpfunc_wbounds(x, ehi, elo)
    gfunc(x) = gfunc_wbounds(x, ehi, elo)

    unc = 1 ./ ell .<= repeat(agrid, ns) .+ η
    ι   = (agrid[end] - agrid[1]) / na

    dF1_dELL, dF1_dRZ, dF1_dELLP, dF1_dWHP, dF1_dTTP, ee =
        euler_equation_hetdsgegovdebt(na, ns, gpfunc, gfunc, agrid, sgrid, fgrid, unc, ι,
                               R, γ, β, η, ell, T, ω, H)

    # KF Equation
    dF2_dWH, dF2_dRZ, dF2_dTT,dF2_dELL, bigΨ, dF2_dM =
        kolmogorov_fwd_hetdsgegovdebt(na, ns, gfunc, gpfunc, agrid, sgrid, fgrid, unc,
                               R, γ, ell, D, η, T, ω, H)

    # Market clearing, lambda function
    c = min.(1 ./ ell,repeat(agrid,ns).+η)
    lam = dot(D, (1 ./ c)) # average marginal utility which the union uses to set wages
    aggc = dot(D, c)
    ϕ = lam*ω/(H^ϕh) # now that we know lam in steady state, choose disutility to target hours H

    setup_indices!(m)
    normalize_model_state_indices!(m)

    nvars = get_setting(m, :nvars)

    # Make the Jacobian. Left dimension is [ell, D, scalars],
    # right dimension is twice the left dimension's size b/c perturb w.r.t. today & tomorrow's values
    JJ = zeros(S, nvars, 2*nvars)
    # Euler equation
    JJ[eq[:eq_euler],endo[:l′_t]] = dF1_dELLP
    JJ[eq[:eq_euler],endo[:z′_t]] = -dF1_dRZ
    JJ[eq[:eq_euler],endo[:w′_t]] = dF1_dWHP
    JJ[eq[:eq_euler],endo[:L′_t]] = dF1_dWHP
    JJ[eq[:eq_euler],endo[:t′_t]] = dF1_dTTP
    JJ[eq[:eq_euler],endo[:b_t]]  = ell
    JJ[eq[:eq_euler],endo[:l_t]]  = dF1_dELL
    JJ[eq[:eq_euler],endo[:R_t]]  = dF1_dRZ

    # KF eqn
    JJ[eq[:eq_kolmogorov_fwd],endo[:kf′_t]] = -Matrix{Float64}(I, nans, nans)
    JJ[eq[:eq_kolmogorov_fwd],endo[:kf_t]]  = dF2_dM * ι
    JJ[eq[:eq_kolmogorov_fwd],endo[:l_t]]   = dF2_dELL
    JJ[eq[:eq_kolmogorov_fwd],endo[:R_t]]   = dF2_dRZ
    JJ[eq[:eq_kolmogorov_fwd],endo[:z_t]]   = -dF2_dM * dF2_dRZ * ι
    JJ[eq[:eq_kolmogorov_fwd],endo[:w_t]]   = dF2_dM * dF2_dWH * ι
    JJ[eq[:eq_kolmogorov_fwd],endo[:L_t]]   = dF2_dM * dF2_dWH * ι
    JJ[eq[:eq_kolmogorov_fwd],endo[:t_t]]   = dF2_dM * dF2_dTT * ι

    # aggregate consumption
    JJ[first(eq[:eq_agg_consumption]),first(endo[:C_t])] = -aggc # normalize C_t by mean consumption
    JJ[first(eq[:eq_agg_consumption]), endo[:l_t]]       = -(D .* unc .* c)
    JJ[first(eq[:eq_agg_consumption]), endo[:kf_t]]      = ι .* c # note, now we linearize
    JJ[first(eq[:eq_agg_consumption]),first(endo[:z_t])] = -dot(ι .* c, dF2_dRZ)
    JJ[first(eq[:eq_agg_consumption]),first(endo[:w_t])] =  dot(ι .* c, dF2_dWH)
    JJ[first(eq[:eq_agg_consumption]),first(endo[:L_t])] =  dot(ι .* c, dF2_dWH)
    JJ[first(eq[:eq_agg_consumption]),first(endo[:t_t])] =  dot(ι .* c, dF2_dTT)

    # lambda = average marginal utility
    JJ[first(eq[:eq_lambda]),first(endo[:margutil_t])] = lam
    JJ[first(eq[:eq_lambda]),endo[:kf_t]]              = -(ι ./ c) # note, now we linearize
    JJ[first(eq[:eq_lambda]),first(endo[:z_t])]        =  dot(ι ./ c, dF2_dRZ)
    JJ[first(eq[:eq_lambda]),first(endo[:w_t])]        = -dot(ι ./ c, dF2_dWH)
    JJ[first(eq[:eq_lambda]),first(endo[:L_t])]        = -dot(ι ./ c, dF2_dWH)
    JJ[first(eq[:eq_lambda]),first(endo[:t_t])]        = -dot(ι ./ c, dF2_dTT)
    JJ[first(eq[:eq_lambda]),endo[:l_t]]               = -(unc .* D ./ c)

    # transfer
    JJ[first(eq[:eq_transfers]),first(endo[:t_t])]         = T
    JJ[first(eq[:eq_transfers]),first(endo[:capreturn_t])] = -Rk * kstar * exp(-γ)
    JJ[first(eq[:eq_transfers]),first(endo[:k_t])]         = -Rk * kstar * exp(-γ)
    JJ[first(eq[:eq_transfers]),first(endo[:z_t])]         = Rk * kstar * exp(-γ)
    JJ[first(eq[:eq_transfers]),first(endo[:I_t])]         = xstar
    JJ[first(eq[:eq_transfers]),first(endo[:mc_t])]        = ystar
    #JJ[first(eq[:eq_transfers]),first(endo[:y_t])]        = (1-1/g)*ystar
    #JJ[first(eq[:eq_transfers]),first(endo[:g_t])]        = (ystar/g)
    JJ[first(eq[:eq_transfers]),first(endo[:tg_t])]        = Tg

    # investment
    JJ[first(eq[:eq_investment]),first(endo[:Q_t])]  = 1.
    JJ[first(eq[:eq_investment]),first(endo[:μ_t])]  = 1.
    JJ[first(eq[:eq_investment]),first(endo[:I′_t])] =  spp * (exp(3 * γ)) / R
    JJ[first(eq[:eq_investment]),first(endo[:z′_t])] =  spp * (exp(3 * γ)) / R
    JJ[first(eq[:eq_investment]),first(endo[:I_t])]  = -spp * (exp(3 * γ)) / R - spp * exp(2 * γ)
    JJ[first(eq[:eq_investment]),first(endo[:I_t1])] =  spp *  exp(2 * γ)
    JJ[first(eq[:eq_investment]),first(endo[:z_t])]  = -spp *  exp(2 * γ)

    # tobin's q
    JJ[first(eq[:eq_tobin_q]),first(endo[:margutil_t])]   =  1.
    JJ[first(eq[:eq_tobin_q]),first(endo[:margutil′_t])]  = -1.
    JJ[first(eq[:eq_tobin_q]),first(endo[:Q_t])]          =  1.
    JJ[first(eq[:eq_tobin_q]),first(endo[:z′_t])]         =  1.
    JJ[first(eq[:eq_tobin_q]),first(endo[:capreturn′_t])] = -Rk/R
    JJ[first(eq[:eq_tobin_q]),first(endo[:Q′_t])]         = -(1. - δ) / R

    # capital accumulation
    JJ[first(eq[:eq_capital_accumulation]),first(endo[:k′_t])] = 1.
    JJ[first(eq[:eq_capital_accumulation]),first(endo[:k_t])]  = -(1 - δ) * exp(-γ)
    JJ[first(eq[:eq_capital_accumulation]),first(endo[:z_t])]  =  (1 - δ) * exp(-γ) # we divide this line by kstar here, which does not happen
    JJ[first(eq[:eq_capital_accumulation]),first(endo[:μ_t])]  = -xstar / kstar # in the Jacobian from autodiff (since it doesn't know it should
    JJ[first(eq[:eq_capital_accumulation]),first(endo[:I_t])]  = -xstar / kstar # unless the residual is divided by kstar)

    # wage phillips curve
    JJ[first(eq[:eq_wage_phillips]),first(endo[:π_w_t])]      = -1.
    JJ[first(eq[:eq_wage_phillips]),first(endo[:λ_w_t])]      = 1. #(ϕ*H^ϕh)/Φw
    JJ[first(eq[:eq_wage_phillips]),first(endo[:L_t])]        = κ_w * ϕh #(ϕ*(H^ϕh)*(1+lamw)/lamw*Φw)*ϕh
    JJ[first(eq[:eq_wage_phillips]),first(endo[:margutil_t])] = -κ_w #-(ϕ*(H^ϕh)*(1+lamw)/lamw*Φw)
    JJ[first(eq[:eq_wage_phillips]),first(endo[:w_t])]        = -κ_w #-(ϕ*(H^ϕh)*(1+lamw)/lamw*Φw)
    JJ[first(eq[:eq_wage_phillips]),first(endo[:π_w′_t])]     = β

    # price phillips curve
    JJ[first(eq[:eq_price_phillips]),first(endo[:π_t])]   = -1.
    JJ[first(eq[:eq_price_phillips]),first(endo[:mc_t])]  = κ_p #(1+lamf)/(lamf*Φp)
    JJ[first(eq[:eq_price_phillips]),first(endo[:λ_f_t])] = 1. #1/Φp
    # JJ[first(eq[:eq_price_phillips]),first(endo[:π′_t])]  = 1 / R # coefficient π′_t is β̃, not 1/R
    JJ[first(eq[:eq_price_phillips]),first(endo[:π′_t])]  = exp(γ) / R

    # marginal cost
    JJ[first(eq[:eq_marginal_cost]),first(endo[:mc_t])]        = 1.
    JJ[first(eq[:eq_marginal_cost]),first(endo[:w_t])]         = -(1 - α)
    JJ[first(eq[:eq_marginal_cost]),first(endo[:capreturn_t])] = -α

    # gdp
    JJ[first(eq[:eq_gdp]),first(endo[:y_t])] = 1.
    JJ[first(eq[:eq_gdp]),first(endo[:z_t])] = α
    JJ[first(eq[:eq_gdp]),first(endo[:k_t])] = -α
    JJ[first(eq[:eq_gdp]),first(endo[:L_t])] = -(1-α)

    # optimal k/l ratio
    JJ[first(eq[:eq_optimal_kl]),first(endo[:capreturn_t])] = 1.
    JJ[first(eq[:eq_optimal_kl]),first(endo[:w_t])]         = -1.
    JJ[first(eq[:eq_optimal_kl]),first(endo[:L_t])]         = -1.
    JJ[first(eq[:eq_optimal_kl]),first(endo[:k_t])]         = 1.
    JJ[first(eq[:eq_optimal_kl]),first(endo[:z_t])]         = -1.

    # taylor rule
    JJ[first(eq[:eq_taylor]),first(endo[:i_t])]  = -1.
    JJ[first(eq[:eq_taylor]),first(endo[:i_t1])] = ρ_R
    JJ[first(eq[:eq_taylor]),first(endo[:π_t])]  = (1 - ρ_R) * ψπ
    JJ[first(eq[:eq_taylor]),first(endo[:π_star_t])]  = -(1 - ρ_R) * ψπ
    JJ[first(eq[:eq_taylor]),first(endo[:y_t])]  = (1 - ρ_R) * ψy
    JJ[first(eq[:eq_taylor]),first(endo[:y_t1])] = -(1 - ρ_R) * ψy
    JJ[first(eq[:eq_taylor]),first(endo[:z_t])]  = (1 - ρ_R) * ψy
    JJ[first(eq[:eq_taylor]),first(endo[:rm_t])] = 1.

    # fisher eqn
    JJ[first(eq[:eq_fisher]),first(endo[:R_t])]  = 1.
    JJ[first(eq[:eq_fisher]),first(endo[:π′_t])] = 1.
    JJ[first(eq[:eq_fisher]),first(endo[:i_t])]  = -1.

    # wage inflation
    JJ[first(eq[:eq_nominal_wage_inflation]),first(endo[:π_w_t])] = 1.
    JJ[first(eq[:eq_nominal_wage_inflation]),first(endo[:π_t])]   = -1.
    JJ[first(eq[:eq_nominal_wage_inflation]),first(endo[:z_t])]   = -1.
    JJ[first(eq[:eq_nominal_wage_inflation]),first(endo[:w_t])]   = -1.
    JJ[first(eq[:eq_nominal_wage_inflation]),first(endo[:w_t1])]  = 1.

    # fiscal rule
    JJ[first(eq[:eq_fiscal_rule]),first(endo[:tg_t])]  = -Tg
    JJ[first(eq[:eq_fiscal_rule]),first(endo[:R_t])]   = δb * bg / R
    JJ[first(eq[:eq_fiscal_rule]),first(endo[:bg_t])]  = δb * bg * exp(-γ)
    JJ[first(eq[:eq_fiscal_rule]),first(endo[:z_t])]   = -δb * bg * exp(-γ)
    JJ[first(eq[:eq_fiscal_rule]),first(endo[:y_t])]   = δb * (1 - (1 / g)) * ystar
    JJ[first(eq[:eq_fiscal_rule]),first(endo[:g_t])]   = δb * ystar / g

    # govt budget constraint
    JJ[first(eq[:eq_g_budget_constraint]),first(endo[:bg′_t])] = -(bg / R)
    JJ[first(eq[:eq_g_budget_constraint]),first(endo[:R_t])]   = bg / R
    JJ[first(eq[:eq_g_budget_constraint]),first(endo[:bg_t])]  = bg * exp(-γ)
    JJ[first(eq[:eq_g_budget_constraint]),first(endo[:z_t])]   = -bg * exp(-γ)
    JJ[first(eq[:eq_g_budget_constraint]),first(endo[:y_t])]   = (1 - (1 / g)) * ystar
    JJ[first(eq[:eq_g_budget_constraint]),first(endo[:g_t])]   = ystar / g
    JJ[first(eq[:eq_g_budget_constraint]),first(endo[:tg_t])]  = -Tg

    # resource constraints
    JJ[first(eq[:eq_resource_constraint]),first(endo[:y_t])] = ystar / g
    JJ[first(eq[:eq_resource_constraint]),first(endo[:g_t])] = -ystar / g
    JJ[first(eq[:eq_resource_constraint]),first(endo[:I_t])] = -xstar
    JJ[first(eq[:eq_resource_constraint]),first(endo[:C_t])] = -aggc

    JJ[first(eq[:LI]),first(endo[:i′_t1])] = 1.
    JJ[first(eq[:LI]),first(endo[:i_t])]   = -1.

    JJ[first(eq[:LY]),first(endo[:y′_t1])] = 1.
    JJ[first(eq[:LY]),first(endo[:y_t])]   = -1.

    JJ[first(eq[:LW]),first(endo[:w′_t1])] = 1.
    JJ[first(eq[:LW]),first(endo[:w_t])]   = -1.

    JJ[first(eq[:LX]),first(endo[:I′_t1])] = 1.
    JJ[first(eq[:LX]),first(endo[:I_t])]   = -1.

    # discount factor shock
    JJ[first(eq[:eq_b]),first(endo[:b′_t])] = 1.
    JJ[first(eq[:eq_b]),first(endo[:b_t])]  = -ρ_B

    # g/y shock
    JJ[first(eq[:eq_g]),first(endo[:g′_t])] = 1.
    JJ[first(eq[:eq_g]),first(endo[:g_t])]  = -ρ_G

    # tfp growth shock
    JJ[first(eq[:eq_z]),first(endo[:z′_t])] = 1.
    JJ[first(eq[:eq_z]),first(endo[:z_t])]  = -ρ_z

    # investment shock
    JJ[first(eq[:eq_μ]),first(endo[:μ′_t])] = 1.
    JJ[first(eq[:eq_μ]),first(endo[:μ_t])]  = -ρ_μ

    # wage mkup shock
    JJ[first(eq[:eq_λ_w]),first(endo[:λ_w′_t])] = 1.
    JJ[first(eq[:eq_λ_w]),first(endo[:λ_w_t])]  = -ρ_lamw

    # price mkup shock
    JJ[first(eq[:eq_λ_f]),first(endo[:λ_f′_t])] = 1.
    JJ[first(eq[:eq_λ_f]),first(endo[:λ_f_t])]  = -ρ_lamf

    # monetary policy shock
    JJ[first(eq[:eq_rm]),first(endo[:rm′_t])] = 1.
    JJ[first(eq[:eq_rm]),first(endo[:rm_t])]  = -ρ_mon

    # inflation expectations
    JJ[first(eq[:eq_π_star]),first(endo[:π_star′_t])] = 1.
    JJ[first(eq[:eq_π_star]),first(endo[:π_star_t])]  = -ρ_π_star

    if !m.testing && get_setting(m, :normalize_distr_variables)
        JJ  = normalize(m, JJ)
    end

    return JJ
end

function euler_equation_hetdsgegovdebt(na::Int, ns::Int,
                                       gpfunc::Function, gfunc::Function,
                                       agrid::Vector{Float64}, sgrid::Vector{Float64},
                                       fgrid::Matrix{Float64},
                                       unc::BitArray, ι::Float64,
                                       R::Float64, γ::Float64, β::Float64,
                                       η::Float64, ell::Vector{Float64}, T::Float64,
                                       ω::Float64, H::Float64)
    nans = na*ns
    ee  = zeros(nans, nans) # ee[i,j] takes you from i to j
    ξ   = zeros(nans, nans)
    Ξ   = zeros(nans, nans)
    dF1_dELL = zeros(nans, nans)
    dF1_dRZ  = zeros(nans)
    dF1_dELLP = zeros(nans, nans)
    dF1_dWHP = zeros(nans)
    dF1_dTTP = zeros(nans)

    @inbounds for iss=1:ns
        @inbounds for ia=1:na
            i  = na*(iss-1)+ia
            sumELL = 0.
            sumRZ  = 0.
            sumWH  = 0.
            sumTT  = 0.
            @inbounds for isp=1:ns
                @inbounds for iap=1:na
                    ip = na*(isp-1)+iap
                    ee[i,ip] = (agrid[iap] - R*(exp(-γ))*max(agrid[ia]-1/ell[i], -η) - T)/(ω*H*sgrid[isp])
                    ξ[i,ip] = ι * ((β*R*exp(-γ))/(ω*H*sgrid[isp])^2)*max(ell[ip],1/(agrid[iap]+η))*gpfunc(ee[i,ip])*fgrid[iss,isp]
                    Ξ[i,ip] = ι * ((β*R*exp(-γ))/(ω*H*sgrid[isp]))*max(ell[ip],1/(agrid[iap]+η))*gfunc(ee[i,ip])*fgrid[iss,isp]
                    sumELL += ξ[i,ip]*R*(exp(-γ))*unc[i]/ell[i]
                    sumRZ  += ξ[i,ip]*R*(exp(-γ))*max(agrid[ia] - 1/ell[i],-η)
                    dF1_dELLP[i,ip] = Ξ[i,ip]*unc[ip]
                    sumWH  += Ξ[i,ip] + ξ[i,ip]*ee[i,ip]*(ω*H*sgrid[isp])
                    sumTT  += ξ[i,ip]*T
                end
            end
            dF1_dELL[i,i] = -ell[i] - sumELL
            dF1_dRZ[i] = ell[i] - sumRZ
            dF1_dWHP[i] = -sumWH
            dF1_dTTP[i] = -sumTT
        end
    end
    return dF1_dELL, dF1_dRZ, dF1_dELLP, dF1_dWHP, dF1_dTTP, ee
end

function kolmogorov_fwd_hetdsgegovdebt(na::Int, ns::Int,
                                       gfunc::Function, gpfunc::Function,
                                       agrid::Vector{S}, sgrid::Vector{S},
                                       fgrid::Matrix{S}, unc::BitArray,
                                       R::S, γ::S, ell::Vector{S}, D::Vector{S},
                                       η::S, T::S, ω::S, H::S) where {S <: Real}
    nans     = na * ns
    bigΨ     = zeros(S, nans, nans)
    smallψ   = zeros(S, nans, nans)
    dF2_dRZ  = zeros(S, nans)
    dF2_dM   = zeros(S, nans, nans)
    dF2_dELL = zeros(S, nans, nans)
    dF2_dWH  = zeros(S, nans)
    dF2_dTT  = zeros(S, nans)

    @inbounds for iss = 1:ns
        @inbounds for ia = 1:na
            i = na * (iss - 1) + ia # (a, s)
            @inbounds for isp = 1:ns
                @inbounds for iap = 1:na
                    ip              = na * (isp - 1) + iap # (a', s')
                    ee              = (agrid[iap] - R * (exp(-γ)) * max(agrid[ia] - 1 / ell[i], -η) - T) / (ω * H * sgrid[isp])
                    bigΨ[ip, i]     = D[i] * gfunc(ee) * fgrid[iss,isp] / (ω * H * sgrid[isp])
                    smallψ[ip, i]   = D[i] * gpfunc(ee) * fgrid[iss,isp] / ((ω * H * sgrid[isp])^2)
                    dF2_dELL[ip, i] = -smallψ[ip, i] * (R * exp(-γ)) * (unc[i] / ell[i])
                    dF2_dM[ip, i]   = gfunc(ee) * fgrid[iss, isp] / (ω * H * sgrid[isp])
                    dF2_dRZ[ip]    -= smallψ[ip, i] * (R * exp(-γ)) * max(agrid[ia] - 1 / ell[i], -η)
                    dF2_dWH[ip]    -= bigΨ[ip, i] + smallψ[ip, i] * ee * (ω * H * sgrid[isp])
                    dF2_dTT[ip]    -= smallψ[ip, i] * T
                end
            end
        end
    end

    return dF2_dWH, dF2_dRZ, dF2_dTT, dF2_dELL, bigΨ, dF2_dM
end
