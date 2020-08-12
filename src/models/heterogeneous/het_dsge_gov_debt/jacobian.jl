function jacobian(m::HetDSGEGovDebt)
    reset_grids!(m)
    #truncate_distribution!(m)

    # Load in endogenous state and eq cond indices
    endo = augment_model_states(m.endogenous_states_original,#m.endogenous_states_unnormalized,
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
    μ::Vector{Float64}    = m[:μstar].value
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

    qp(z) = dmollifier_hetdsgegovdebt(z, ehi, elo)
    qfunction_hetdsgegovdebt(x) = mollifier_hetdsgegovdebt(x, ehi, elo)

    unc = 1 ./ ell .<= repeat(agrid,ns) .+ η

    dF1_dELL, dF1_dRZ, dF1_dELLP, dF1_dWHP, dF1_dTTP, ee =
        euler_equation_hetdsgegovdebt(na, ns, qp, qfunction_hetdsgegovdebt, agrid, sgrid, fgrid, unc,
                               R, γ, β, η, ell, T, ω, H)

    # KF Equation
    dF2_dWH, dF2_dRZ, dF2_dTT,dF2_dELL, bigΨ, dF2_dM =
        kolmogorov_fwd_hetdsgegovdebt(na, ns, qfunction_hetdsgegovdebt, qp, agrid, sgrid, fgrid, unc,
                               R, γ, ell, μ, η, T, ω, H, ee)

    # Market clearing, lambda function
    c = min.(1 ./ ell,repeat(agrid,ns).+η)
    lam = dot(μ, (1 ./ c)) # average marginal utility which the union uses to set wages
    aggc = dot(μ, c)
    ϕ = lam*ω/(H^ϕh) # now that we know lam in steady state, choose disutility to target hours H

    setup_indices!(m)
    normalize_model_state_indices!(m)

    nvars = get_setting(m, :nvars)

    # Make the Jacobian. Left dimension is [ell, μ, scalars],
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
    JJ[eq[:eq_kolmogorov_fwd],endo[:kf_t]]  = dF2_dM
    JJ[eq[:eq_kolmogorov_fwd],endo[:l_t]]   = dF2_dELL
    JJ[eq[:eq_kolmogorov_fwd],endo[:R_t]]   = dF2_dRZ
    JJ[eq[:eq_kolmogorov_fwd],endo[:z_t]]   = -dF2_dM*dF2_dRZ
    JJ[eq[:eq_kolmogorov_fwd],endo[:w_t]]   = dF2_dM*dF2_dWH
    JJ[eq[:eq_kolmogorov_fwd],endo[:L_t]]   = dF2_dM*dF2_dWH
    JJ[eq[:eq_kolmogorov_fwd],endo[:t_t]]   = dF2_dM*dF2_dTT

    # aggregate consumption
    JJ[first(eq[:eq_agg_consumption]),first(endo[:C_t])] = -aggc
    JJ[first(eq[:eq_agg_consumption]), endo[:l_t]]       = -(μ .* unc .* c)
    JJ[first(eq[:eq_agg_consumption]), endo[:kf_t]]      = c # note, now we linearize
    JJ[first(eq[:eq_agg_consumption]),first(endo[:z_t])] = -dot(c, dF2_dRZ)
    JJ[first(eq[:eq_agg_consumption]),first(endo[:w_t])] =  dot(c, dF2_dWH)
    JJ[first(eq[:eq_agg_consumption]),first(endo[:L_t])] =  dot(c, dF2_dWH)
    JJ[first(eq[:eq_agg_consumption]),first(endo[:t_t])] =  dot(c, dF2_dTT)

    # lambda = average marginal utility
    JJ[first(eq[:eq_lambda]),first(endo[:margutil_t])] = lam
    JJ[first(eq[:eq_lambda]),endo[:kf_t]]              = -(1 ./ c) # note, now we linearize
    JJ[first(eq[:eq_lambda]),first(endo[:z_t])]        =  dot(1 ./ c, dF2_dRZ)
    JJ[first(eq[:eq_lambda]),first(endo[:w_t])]        = -dot(1 ./ c, dF2_dWH)
    JJ[first(eq[:eq_lambda]),first(endo[:L_t])]        = -dot(1 ./ c, dF2_dWH)
    JJ[first(eq[:eq_lambda]),first(endo[:t_t])]        = -dot(1 ./ c, dF2_dTT)
    JJ[first(eq[:eq_lambda]),endo[:l_t]]               = -(unc .* μ ./ c)

    # transfer
    JJ[first(eq[:eq_transfers]),first(endo[:t_t])]         = T
    JJ[first(eq[:eq_transfers]),first(endo[:capreturn_t])] = -Rk * kstar
    JJ[first(eq[:eq_transfers]),first(endo[:k_t])]         = -Rk * kstar
    JJ[first(eq[:eq_transfers]),first(endo[:z_t])]         = Rk * kstar
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
    JJ[first(eq[:eq_capital_accumulation]),first(endo[:k_t])]  = -(1 - δ)
    JJ[first(eq[:eq_capital_accumulation]),first(endo[:z_t])]  =  (1 - δ)
    JJ[first(eq[:eq_capital_accumulation]),first(endo[:μ_t])]  = -xstar / kstar
    JJ[first(eq[:eq_capital_accumulation]),first(endo[:I_t])]  = -xstar / kstar

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
    JJ[first(eq[:eq_price_phillips]),first(endo[:π′_t])]  = 1 / R

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

    if !m.testing && get_setting(m, :normalize_distr_variables)
        JJ  = normalize(m, JJ)
    end
    return JJ
end

function euler_equation_hetdsgegovdebt(na::Int, ns::Int,
                                       qp::Function, qfunction::Function,
                                       agrid::Vector{Float64}, sgrid::Vector{Float64},
                                       fgrid::Matrix{Float64},
                                       unc::BitArray,
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
    for iss=1:ns
        for ia=1:na
            i  = na*(iss-1)+ia
            sumELL = 0.
            sumRZ  = 0.
            sumWH  = 0.
            sumTT  = 0.
            for isp=1:ns
                for iap=1:na
                    ip = na*(isp-1)+iap
                    ee[i,ip] = (agrid[iap] - R*(exp(-γ))*max(agrid[ia]-1/ell[i], -η) - T)/(ω*H*sgrid[isp])
                    ξ[i,ip] = ((β*R*exp(-γ))/(ω*H*sgrid[isp])^2)*max(ell[ip],1/(agrid[iap]+η))*qp(ee[i,ip])*fgrid[iss,isp]
                    Ξ[i,ip] = ((β*R*exp(-γ))/(ω*H*sgrid[isp]))*max(ell[ip],1/(agrid[iap]+η))*qfunction(ee[i,ip])*fgrid[iss,isp]
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
                                qfunction::Function, qp::Function,
                                agrid::Vector{Float64}, sgrid::Vector{Float64},
                                fgrid::Matrix{Float64}, unc::BitArray,
                                R::Float64, γ::Float64,
                                ell::Vector{Float64}, μ::Vector{Float64},
                                η::Float64, T::Float64, ω::Float64, H::Float64, ee::Matrix{Float64})
    nans = na*ns
    bigΨ    = zeros(nans, nans)
    smallψ = zeros(nans, nans)
    dF2_dRZ = zeros(nans)
    dF2_dM = zeros(nans, nans)
    dF2_dELL = zeros(nans, nans)
    dF2_dWH = zeros(nans)
    dF2_dTT = zeros(nans)

    for isp=1:ns
        for iap=1:na
            ip  = na*(isp-1)+iap
            sumWH = 0.
            sumRZ = 0.
            sumTT = 0.
            for iss=1:ns
                for ia=1:na
                    i = na*(iss-1)+ia
                    bigΨ[ip,i] = μ[i]*qfunction(ee[i,ip])*fgrid[iss,isp]/(ω*H*sgrid[isp])
                    smallψ[ip,i] = μ[i]*qp(ee[i,ip])*fgrid[iss,isp]/((ω*H*sgrid[isp])^2)
                    sumWH += bigΨ[ip,i] + smallψ[ip,i]*ee[i,ip]*(ω*H*sgrid[isp])
                    sumRZ += smallψ[ip,i]*(R*exp(-γ))*max(agrid[ia] - 1/ell[i],-η)
                    dF2_dELL[ip,i] = smallψ[ip,i]*(R*exp(-γ))*(unc[i]/ell[i])
                    sumTT += smallψ[ip,i]*T
                    dF2_dM[ip,i] = qfunction(ee[i,ip])*fgrid[iss,isp]/(ω*H*sgrid[isp]) # note, now we linearize
                end
            end
            dF2_dWH[ip] = -sumWH
            dF2_dRZ[ip] = -sumRZ
            dF2_dTT[ip] = -sumTT
        end
    end
    return dF2_dWH, dF2_dRZ, dF2_dTT, dF2_dELL, bigΨ, dF2_dM
end

function normalize(m::HetDSGEGovDebt, JJ::Matrix{Float64})

    Qx, Qy, Qleft, Qright  = compose_normalization_matrices(m)

    m <= Setting(:n_predetermined_variables, size(Qx, 1))
    m <= Setting(:Qx, Qx)
    m <= Setting(:Qy, Qy)
    #Qleft = get_setting(m, :Qleft) #m[:Qleft].value
    #Qright = get_setting(m, :Qright) #m[:Qright].value

	Jac1 = Qleft*JJ*Qright

    return Jac1
end

# use proxy distributions method - now `averaging' so that Qx'*Qx should not `add mass'
# Basically, maps an `n` length vector to a smaller vector by binning grid points into new bins of size binsize
# (with a size of 1 being an `n` to `n` mapping)
# TODO: Refactor to use Kronecker and Diagonal, and BlockBandedMatrix
function avg_prox(n::Int, binsize::Int = 3, minn_noag::Int = 0)
    minn_noag_use = min(minn_noag,n)
    n_gps = Int(floor((n-minn_noag_use)/binsize)) # number of groups, binsize is the size of the bins we reallocate distribution to
    n_noag = n - binsize*n_gps # Leftover points after rebinning distribution, assumed these are at the left end of the grid
    Prox = cat(eye(n_noag), kron(eye(n_gps),fill(1 ./ sqrt(binsize), 1, binsize)), dims = [1 2])
    n_new = n_noag + n_gps
    return (Prox, n_new, n_noag, n_gps)
end
# note: setting minn_noag = 0 seems to work the best

# TODO: use SparseArrays or something like that here; can use QR on it
function make_S(n::Int, n_noag::Int = 0, binsize::Int = 1)
    P1 = ones(n,1)*sqrt(binsize)
    P1[1:n_noag] .= 1. # set unbinned points to 1
    Ptemp = eye(n)
    Ptemp = Ptemp[:,2:end] # Remove the first column of the identity matrix, and replace w/ P1
    P2 = Ptemp
    P = [P1 P2]
    (QQQ,Rjunk)=qr(P) # Get eigenvalues
    S         = QQQ[:,2:end]'
    return S
end

# TODO: USE SPARSE MATRICES/BANDED MATRICES TO IMPLEMENT THE REDUCTION, KRONECKER
function compose_normalization_matrices(m::HetDSGEGovDebt)
    if get_setting(m, :poor_man_reduc)
        na = get_setting(m, :na1_state) #:na) # Want the na1_state, not na, b/c want dimension of low skill cash on hand
        ns = get_setting(m, :ns)
        n = na*ns #get_setting(m, :na1) + get_setting(m, :na2) #
        nscalars = get_setting(m, :nscalars)
        nyscalars = get_setting(m, :nyscalars)
        nascalars = get_setting(m, :nascalars)
        mindens = get_setting(m, :mindens)
        μ = m[:μstar].value

        na1 = maximum(findall(μ[1:na].>mindens)) # Chop off unneeded cash-on-hand grid points for low-skill workers b/c no one there
        m <= Setting(:na1_state, na1)
        m <= Setting(:na1_jump, na1)

        setup_indices!(m) # Update the indices
        init_states_and_jumps!(m, get_setting(m, :states), get_setting(m, :jumps)) # Update mapping from states/jumps to indices
        normalize_model_state_indices!(m) # Update state indices to enforce Kolmogorov distribution integrates to 1
        endogenous_states_augmented = [:C_t1]
       for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i + first(m.endogenous_states[get_setting(m, :jumps)[end]]) end #first(collect(values(m.endogenous_states))[end]) end # Augment the model w/post-steady-state states

        m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states) +
                     length(m.endogenous_states_augmented)) # Update number of states (incl. augmented states)

        Reduc = eye(n)
        Reduc=Reduc[[1:na1;na+1:n],:] # Remove eliminated cash-on-hand grid points for low skill workers

        minn_noag = 0
        binsize = get_setting(m, :binsize) # this is the maximum binsize which seemed to leave the IRFs unchanged, you can experiment with this
        # setting binsize = 1 should return what we had before
        (ProxL, n_newL, n_noagL, n_gpsL) = avg_prox(na1, binsize, minn_noag) # Get new indices corresponding to binning
        (ProxH, n_newH, n_noagH, n_gpsH) = avg_prox(na, binsize,minn_noag)   # reduction of distributions
        m <= Setting(:na1_state, n_newL)
        m <= Setting(:na2_state, n_newH)
        if get_setting(m, :reduce_ell) # Reduce ell by using binning
            m <= Setting(:na1_jump, n_newL)
            m <= Setting(:na2_jump, n_newH)
        end
        setup_indices!(m) # Update indices, states, jumps, augmenetd states, etc.
        init_states_and_jumps!(m, get_setting(m, :states), get_setting(m, :jumps))
        normalize_model_state_indices!(m)
        endogenous_states_augmented = [:C_t1]
        for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i + first(m.endogenous_states[get_setting(m, :jumps)[end]]) end #first(collect(values(m.endogenous_states))[end]) end

        m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states) +
                     length(m.endogenous_states_augmented))

        # S is made from an orthogonal matrix, so further reduction by projecting onto an orthogonal basis?
        Prox = cat(ProxL, ProxH, dims = [1 2]) # TODO: Create a BlockBanded Matrix here rather than cat
        SL = make_S(n_newL, n_noagL, binsize)
        SH = make_S(n_newH, n_noagH, binsize)
        S = cat(SL, SH, dims = [1 2]) # TODO: Create a BlockBanded Matrix here rather than cat

        # TODO: Create a BlockBanded Matrix here rather than cat
        if get_setting(m, :reduce_ell)
            Qleft     = cat(Prox*Reduc,S*Prox*Reduc,eye(nscalars), dims = [1 2]) # ell is in the first Reduc, so we add
            Qx        = cat(S*Prox*Reduc,eye(nascalars), dims = [1 2])           # an Prox * Reduc to further reduce
            Qy        = cat(Prox*Reduc,eye(nyscalars), dims = [1 2])
        else
            Qleft     = cat(Reduc,S*Prox*Reduc,eye(nscalars), dims = [1 2])
            Qx        = cat(S*Prox*Reduc,eye(nascalars), dims = [1 2]) # same as if reducing ell b/c just applies to distribution
            Qy        = cat(Reduc,eye(nyscalars), dims = [1 2])
        end

        Qright    = cat(Qx',Qy',Qx',Qy', dims = [1,2])

        return Qx, Qy, Qleft, Qright # Qx, Qy are the individual components of Qleft, Qright

    else
        na = get_setting(m, :na)
        ns = get_setting(m, :ns)
        nscalars = get_setting(m, :nscalars)
        nyscalars = get_setting(m, :nyscalars)
        nascalars = get_setting(m, :nascalars)

        # Create PPP matrix
        P1 = kron(Matrix{Float64}(I, ns,ns),ones(na,1))
        Ptemp = Matrix{Float64}(I, na, na)
        Ptemp = Ptemp[:, 2:end]
        P2 = kron(Matrix{Float64}(I, ns, ns), Ptemp)
        P  = hcat(P1, P2)

        Q,R = qr(P)
        Q = Array(Q)
        S         = Q[:, ns+1:end]'

        nans = na*ns

        Qleft     = cat(Matrix{Float64}(I, nans, nans),S,Matrix{Float64}(I, nscalars, nscalars), dims = [1 2])
        Qx        = cat(S,Matrix{Float64}(I, nascalars, nascalars), dims = [1 2])
        Qy        = cat(Matrix{Float64}(I, nans, nans),Matrix{Float64}(I, nyscalars, nyscalars), dims = [1 2])
        Qright    = cat(Qx',Qy',Qx',Qy', dims = [1,2])

        return Qx, Qy, Qleft, Qright

    end
end

function truncate_distribution!(m::HetDSGEGovDebt)
    mindens = get_setting(m, :mindens)
    trunc_distr = get_setting(m, :trunc_distr)
    rescale_weights = get_setting(m, :rescale_weights)

    na = get_setting(m, :na)
    μ = m[:μstar].value
    ell = m[:lstar].value
    c = m[:cstar].value
    agrid = m.grids[:agrid].points
    #alo = get_setting(m, :alo)
    swts::Vector{Float64}  = m.grids[:sgrid].weights

    if trunc_distr
        oldna = na
        na = maximum(findall(μ[1:na]+μ[na+1:2*na] .> mindens)) # used to be 1e-8
        m[:μstar] = μ[[1:na;oldna+1:oldna+na]]
        m[:lstar] = ell[[1:na;oldna+1:oldna+na]]
        m[:cstar] = c[[1:na;oldna+1:oldna+na]]
        if rescale_weights
            ahi = agrid[na]
            alo = agrid[1]
            ascale = ahi-alo
        end
        m <= Setting(:na, na)
        m <= Setting(:ahi, ahi)
        m <= Setting(:ascale, ascale)
        m.grids[:agrid] = Grid(uniform_quadrature(ascale), alo, ahi, na, scale = ascale)
        m.grids[:weights_total] = kron(swts, m.grids[:agrid].weights)
        nans = na*get_setting(m, :ns)
        m <= Setting(:n, nans)
        m <= Setting(:na, na)

        setup_indices!(m)

        init_states_and_jumps!(m, get_setting(m, :states), get_setting(m, :jumps))

        normalize_model_state_indices!(m)

        endogenous_states_augmented = [:C_t1]
        for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i + first(m.endogenous_states[get_setting(m, :jumps)[end]]) end #first(collect(values(m.endogenous_states))[end]) end

        m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states) +
                     length(m.endogenous_states_augmented))

    end
end
