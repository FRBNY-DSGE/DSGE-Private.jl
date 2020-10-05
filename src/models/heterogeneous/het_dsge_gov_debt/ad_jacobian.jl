include("util.jl")
using UnPack, SparseDiffTools, SparsityDetection, SparseArrays
# REMAINDER TODO:
# 1. Create function that creates Jacobian caches or sparsity patterns
# 2. TODO: figure out how to cache zero residual input
function autodiff_jacobian(m::HetDSGEGovDebt{T}) where {T}
    # Make sure grids and indices correspond to steady state
    reset_grids!(m; init_grids = false) # to make this work with an adaptive agrid, we cannot recreate grid of steady-state approximation
    # truncate_distribution!(m)

    x = zeros(T, 2 * get_setting(m, :nvars)::Int)  # linearize around steady state
    nt′, nt = construct_steadystate_namedtuples(m) # construct NamedTuple of steady state values

    # Indexing and steady state value
    id      = DSGE.augment_model_states(m.endogenous_states_original, DSGE.n_model_states_original(m)) # endogenous states and jump variables
    eq      = m.equilibrium_conditions

    # Load in parameters, steady-state parameters, and grids. May need to add Type Assert back to @unpack macro if this is type unstable
    θ = parameters2namedtuple(m; include_steadystate = true)

#=    @unpack α, δ, γ, η, ρ_b, ρ_g, ρ_z, ρ_μ, ρ_λ_w, ρ_λ_f, ρ_rm, spp, ϕh, ρR, ψπ, ψy, κ_p, κ_w = m
    @unpack H, Tg, δb, bg, xstar, ystar, Rkstar, kstar, π_star = m=#

    agrid = m.grids[:agrid].points::Vector{Float64}
    awts  = m.grids[:agrid].weights::Vector{Float64}
    sgrid = m.grids[:sgrid].points::Vector{Float64}
    swts  = m.grids[:sgrid].weights::Vector{Float64}
    fgrid = m.grids[:fgrid]::Matrix{Float64}

    na   = DSGE.get_setting(m, :na)::Int64
    ns   = DSGE.get_setting(m, :ns)::Int64

    # Define some wrapper functions around e shock distribution and investment cost function
    gfunc_type = get_setting(m, :gfunc_type)
    if gfunc_type == :mollifier
        gfunc_wbounds = (x, y, z) -> mollifier_hetdsgegovdebt(x, y, z) # need this for generate_us_and_es
        gpfunc_wbounds = (x, y, z) -> dmollifier_hetdsgegovdebt(x, y, z)
    elseif gfunc_type == :lognormal
        gfunc_wbounds = (x, y, z) -> lognormal_hetdsgegovdebt(x, y, z, θ[:μ_e].value, θ[:σ_e].value)
        gpfunc_wbounds = (x, y, z) -> dlognormal_hetdsgegovdebt(x, y, z, θ[:μ_e].value, θ[:σ_e].value)
    end

    @inline g_fn(x::S0) where {S0 <: Real}  = gfunc_wbounds(x,  θ[:ehi], θ[:elo])
    @inline gp_fn(x::S0) where {S0 <: Real} = gpfunc_wbounds(x, θ[:ehi], θ[:elo])

    @inline s_fn(x::S0) where {S0 <: Real} = (θ[:spp] / 2.0) * (x - exp(θ[:γ])) ^ 2.0 # so that s(exp(γ)) = s'(exp(γ)) = 0.
    @inline sp_fn(x::S0) where {S0 <: Real} = θ[:spp] * (x - exp(θ[:γ]))

    agrη_unc       = repeat(m.grids[:agrid].points, get_setting(m, :ns)::Int64) .+ m[:η].value
    unc            = agrη_unc .>= (1 ./ m[:lstar].value)
    unc_float      = float.(unc)
    agrη_unc[unc] .= 0.

    dF2_dWH, dF2_dRZ, dF2_dTT, dF2_dELL, bigΨ, dF2_dM = kolmogorov_fwd_hetdsgegovdebt(na, ns, g_fn, gp_fn,
                                                                                      agrid, sgrid, fgrid, unc,
                                                                                      m[:r] + 1., m[:γ].scaledvalue,
                                                                                      m[:lstar].value, m[:Dstar].value, m[:η].value,
                                                                                      m[:Tstar].value, m[:ωstar].value, m[:H].value)
    resF = (dF, xarg) ->_jac_residuals_hetdsgegovdebt(dF, xarg, id, nt′, nt, θ, agrη_unc, unc_float, s_fn, sp_fn, g_fn, m.grids,
                                                      dF2_dWH, dF2_dRZ, dF2_dTT, dF2_dELL, dF2_dM,
                                                      m.equilibrium_conditions)
    output = Vector{eltype(x)}(undef, length(x) ÷ 2)
    JJ = ForwardDiff.jacobian(resF, output, x)
    if !m.testing && get_setting(m, :normalize_distr_variables)
        JJ = DSGE.normalize(m, JJ)
    end
    return JJ

#=    resF = (dF, xarg) -> F(dF, xarg, xss, id, idss)
    if run_jacobian
        if isnothing(jac_cache)
            output = Vector{eltype(x)}(undef, length(x) ÷ 2)
            if isnothing(JJ)
                JJ::Matrix{eltype(x)} = ForwardDiff.jacobian(resF, output, x)
            else
                ForwardDiff.jacobian!(JJ, resF, output, x)
            end
        elseif !isnothing(colorvec) && !create_cache
            JJ = isnothing(JJ) ? Matrix{eltype(x)}(undef, length(x) ÷ 2, length(x)) : JJ
            forwarddiff_color_jacobian!(JJ, resF, x, colorvec = colorvec)
        else
            JJ = isnothing(JJ) ? Matrix{eltype(x)}(undef, length(x) ÷ 2, length(x)) : JJ
            forwarddiff_color_jacobian!(JJ, resF, x, jac_cache)
        end

        if create_cache
            @assert isnothing(jac_cache)
            colors = isnothing(colorvec) ? matrix_colors(sparse(JJ)) : colorvec
            jac_cache = ForwardColorJacCache(resF, x, dx = output, colorvec = colors, sparsity = sparse(JJ))
            return JJ, jac_cache
        else
            if !m.testing && get_setting(m, :normalize_distr_variables)
                JJ = DSGE.normalize(m, JJ)
            end
            return JJ
        end
    else
        output = Vector{eltype(x)}(undef, length(x) ÷ 2)
        return resF(output, x)
    end
=#
end

function _jac_residuals_hetdsgegovdebt(dF::AbstractVector{<: Real}, x::AbstractVector{<: Real},
                                       id::AbstractDict{Symbol, UnitRange}, nt′::NamedTuple, nt::NamedTuple,
                                       θ::NamedTuple, agrη_unc::AbstractArray{S0},
                                       unc_float::AbstractArray{S0}, s_fn::Function, sp_fn::Function,
                                       g_fn::Function, grids::OrderedDict{Symbol,Union{Grid, Array}}, dF2_dWH::AbstractVector{S0},
                                       dF2_dRZ::AbstractVector{S0}, dF2_dTT::AbstractVector{S0},
                                       dF2_dELL::AbstractMatrix{S0}, dF2_dM::AbstractMatrix{S0},
                                       eq::OrderedDict{Symbol,UnitRange}) where {S0 <: Real}

    # Get grid information
    agrid = grids[:agrid].points::Vector{Float64}
    awts  = grids[:agrid].weights::Vector{Float64}
    sgrid = grids[:sgrid].points::Vector{Float64}
    swts  = grids[:sgrid].weights::Vector{Float64}
    fgrid = grids[:fgrid]::Matrix{Float64}
    ι     = awts[1] # uniform quadrature so only need first element
    na    = length(agrid)
    ns    = length(sgrid)
    nans  = (na * ns)

    # Get parameters
    @unpack α, δ, γ, η, ρ_b, ρ_g, ρ_z, ρ_μ, ρ_λ_w, ρ_λ_f, ρ_rm, spp, ϕh, ρR, ψπ, ψy, κ_p, κ_w, elo, ehi = θ
    @unpack H, Tg, δb, bg, xstar, ystar, Rkstar, kstar, π_star = θ
    β    = θ[:βstar]
    ω    = θ[:ωstar]
    T    = θ[:Tstar]
    ell  = θ[:lstar]
    D    = θ[:Dstar]
    R    = 1 + θ[:r]
    βtil = exp(γ) / R

    # Map variables from log deviations (from steady state) into levels (except kf′_t, kf_t, z′_t, z_t, rm′_t, rm_t, t′_t, t_t)
    @sslogdeviations2levels k′_t, i′_t1, y′_t1, w′_t1, I′_t1, bg′_t, b′_t, g′_t, μ′_t = x, id, nt′
    @sslogdeviations2levels λ_w′_t, λ_f′_t, l′_t, C′_t, R′_t, i′_t, w′_t, L′_t = x, id, nt′
    @sslogdeviations2levels π′_t, π_w′_t, margutil′_t, y′_t, I′_t, mc′_t, Q′_t, capreturn′_t, tg′_t = x, id, nt′
    @sslogdeviations2levels k_t, i_t1, y_t1, w_t1, I_t1, bg_t, b_t, g_t, μ_t = x, id, nt
    @sslogdeviations2levels λ_w_t, λ_f_t, l_t, C_t, R_t, i_t, w_t, L_t = x, id, nt
    @sslogdeviations2levels π_t, π_w_t, margutil_t, y_t, I_t, mc_t, Q_t, capreturn_t, tg_t = x, id, nt

    # Map variables from deviations to levels for z′_t, z_t, rm′_t, rm_t
    @ssdeviations2levels z′_t, rm′_t = x, id, nt′
    @ssdeviations2levels z_t, rm_t = x, id, nt

    # Handle transfers separately b/c negative steady state value
    t′_t = exp(x[id[:t′_t][1]]) * nt′[:t′_t]
    t_t  = exp(x[id[:t_t][1]])  * nt[:t_t]

    # Grab log deviations for the wage and price Phillips curves
    Lhat_t        = x[id[:L_t][1]]
    margutilhat_t = x[id[:margutil_t][1]]
    what_t        = x[id[:w_t][1]]
    λtil_w_t      = x[id[:λ_w_t][1]]
    πhat_w′_t     = x[id[:π_w′_t][1]]
    πhat_w_t      = x[id[:π_w_t][1]]
    mchat_t       = x[id[:mc_t][1]]
    λtil_f_t      = x[id[:λ_f_t][1]]
    πhat′_t       = x[id[:π′_t][1]]
    πhat_t        = x[id[:π_t][1]]

    # Set up the Kolmogorov forward equation and D(a, s)
    D̂_t = ι .* ((@view x[id[:kf_t]]) + dF2_dWH .* (x[id[:w_t][1]] + x[id[:L_t][1]]) -
                dF2_dRZ .* x[id[:z_t][1]] + dF2_dTT .* x[id[:t_t][1]])
    D_t = D̂_t + ι * nt[:kf_t]

    # Define useful quantities
    cfunc′  = agrη_unc + unc_float ./ l′_t # Not the best but needed to ensure autodiff recognizes all the places
    cfunc   = agrη_unc + unc_float ./ l_t  # where it should autodiff, according to the analytical Jacobian
    eposzt  = exp(z_t)
    eposzpt = exp(z′_t)
    enegzt  = exp(-z_t)
    enegzpt = exp(-z′_t)

    # Set up Euler equation
    l_t_RHS = Vector{eltype(x)}(undef, nans) # RHS of Euler Equation
    euler_equation!(l_t_RHS, β, b_t, R_t, z′_t, t′_t, w′_t, L′_t, cfunc, cfunc′, na, ns, ι, agrid, sgrid, fgrid, g_fn)

    # Functional equations

    # Euler equation
    dF[eq[:eq_euler]] = l_t_RHS - l_t

    # KF equation
    dF[eq[:eq_kolmogorov_fwd]]  = dF2_dM * D̂_t + (dF2_dRZ .* x[id[:R_t][1]]) +
        dF2_dELL * (@view x[id[:l_t]]) - (@view x[id[:kf′_t]])

    # Equations mapping functions to scalars
    dF[eq[:eq_agg_consumption][1]] = (D_t' * cfunc) - C_t
    dF[eq[:eq_lambda][1]]          = margutil_t - (D_t' * (1 ./ cfunc))

    # Aggregate equations directly linearized (block 1)
    dF[eq[:eq_transfers][1]]  = tg_t + t_t - (capreturn_t * k_t * enegzt - I_t + (1. - mc_t) * y_t)
    dF[eq[:eq_investment][1]] = (margutil_t * Q_t * μ_t * (1 - s_fn(I_t / I_t1 * eposzt)) +
                                 βtil * margutil′_t * enegzpt * Q′_t * μ′_t * sp_fn(I′_t / I_t * eposzpt) *
                                 (I′_t / I_t * eposzpt)^2 - (margutil_t * (1 + Q_t * μ_t * sp_fn(I_t / I_t1 * eposzt) *
                                                                           I_t / I_t1 * eposzt))) / exp(nt[:margutil_t])
    dF[eq[:eq_tobin_q][1]]    = Q_t - (βtil * margutil′_t * enegzpt / margutil_t * (capreturn′_t + Q′_t * (1. - δ)))
    dF[eq[:eq_capital_accumulation][1]] = ((k′_t - ((1. - δ) * k_t * enegzt + μ_t * (1 - s_fn(I_t / I_t1 * eposzt)) * I_t))) /
        exp(nt[:k_t]) # dividing by kstar helps normalize the equation and to match analytical Jacobian

    # Aggregate equations linearized by hand
    dF[eq[:eq_wage_phillips][1]]  = κ_w * (ϕh * Lhat_t - margutilhat_t - what_t) + λtil_w_t + β * πhat_w′_t - πhat_w_t
    dF[eq[:eq_price_phillips][1]] = κ_p * mchat_t + λtil_f_t + βtil * πhat′_t - πhat_t # I believe this is correct but in case ...
    # dF[eq[:eq_price_phillips][1]] = κ_p * mĉ_t + λ̃_f_t + π̂′_t / exp(xss[xssid[:R_t][1]]) - π̂_t # we want to match analytical Jacobian

    # Aggregate equations first logged then linearized
    dF[eq[:eq_marginal_cost][1]] = log(mc_t) - log(((1. - α) ^ (1. - α) * α ^ α) ^ (-1) * w_t ^ (1. - α) * capreturn_t ^ α)
    dF[eq[:eq_gdp][1]]           = log(y_t) - log(exp(-α * z_t) * k_t ^ α * L_t ^ (1. - α))
    dF[eq[:eq_optimal_kl][1]]    = log(capreturn_t / w_t) - log(α / (1. - α) * L_t / k_t * eposzt)
    dF[eq[:eq_taylor][1]]        = log((i_t1 / exp(nt[:i_t])) ^ ρR *        # Note î_t = log(1 + i_t) - log(1 + i), so when
                                       ((π_t / exp(nt[:π_t])) ^ ψπ * (y_t / y_t1 * exp(z_t - γ)) ^ ψy) ^ (1 - ρR) * # brought to levels, i_t is really 1 + i_t.
                                       exp(rm_t)) - log(i_t / exp(nt[:i_t]))                           # Same w/inflation, i.e. the π_t here is really Π_t
    dF[eq[:eq_fisher][1]]                 = log(R_t) - log(i_t / π′_t)
    dF[eq[:eq_nominal_wage_inflation][1]] = log(π_w_t) - log(π_t * eposzt * w_t / w_t1)

    # Aggregate equations directly linearized (block 2)
    dF[eq[:eq_fiscal_rule][1]]         = δb * (bg_t * enegzt + (1 - 1 / g_t) * y_t - bg / R_t) + (1. - δ) * Tg - tg_t
    dF[eq[:eq_g_budget_constraint][1]] = bg_t * enegzt + (1 - 1 / g_t) * y_t - tg_t - bg′_t / R_t
    dF[eq[:eq_resource_constraint][1]] = y_t / g_t - C_t - I_t

    # Lags, e.g. log(y′ₜ₋₁) - log(yss) = log(yₜ) - log(yss) ⇒ log(y′ₜ₋₁) = log(yₜ)
    dF[eq[:LI][1]] = log(i′_t1) - log(i_t)
    dF[eq[:LY][1]] = log(y′_t1) - log(y_t)
    dF[eq[:LW][1]] = log(w′_t1) - log(w_t)
    dF[eq[:LX][1]] = log(I′_t1) - log(I_t)

    # Add exogenous processes
    dF[eq[:eq_b][1]]   = log(b′_t) - ρ_b * log(b_t)
    # dF[eq[:eq_g][1]]   = log(g′_t) - ρ_g * log(g_t)
    dF[eq[:eq_g][1]]   = x[id[:g′_t][1]] - ρ_g * x[id[:g_t][1]]
    # dF[eq[:eq_z][1]]   = z′_t - ρ_z * z_t # leave in levels b/c z′_t = log(exp(z′_t)), etc.
    dF[eq[:eq_z][1]]   = x[id[:z′_t][1]] - ρ_z * x[id[:z_t][1]] # leave in levels b/c z′_t = log(exp(z′_t)), etc.
    dF[eq[:eq_μ][1]]   = log(μ′_t) - ρ_μ * log(μ_t)
    dF[eq[:eq_λ_w][1]] = log(λ_w′_t) - ρ_λ_w * log(λ_w_t)
    dF[eq[:eq_λ_f][1]] = log(λ_f′_t) - ρ_λ_f * log(λ_f_t)
    # dF[eq[:eq_rm][1]]  = rm′_t - ρ_rm * rm_t # leave in deviations of levels b/c rm′_t = log(exp(rm′_t), etc.
    dF[eq[:eq_rm][1]]  = x[id[:rm′_t][1]] - ρ_rm * x[id[:rm_t][1]] # leave in deviations of levels b/c rm′_t = log(exp(rm′_t), etc.
end


# Construct nonlinear Euler equation
@inline function euler_equation!(ret_ell::AbstractVector{S0}, β::S1, b_t::S0,
                                 R_t::S0, z′_t::S0, t′_t::S0, ω′_t::S0, H′_t::S0, # recall Hₜ ≡ Lₜ
                                 cfunc::AbstractVector{S0}, cfunc′::AbstractVector{S0},
                                 na::Int, ns::Int, ι::S1,
                                 agrid::AbstractVector{S1}, sgrid::AbstractVector{S1}, fgrid::AbstractMatrix{S1},
                                 g_fn::Function) where {S0 <: Real, S1 <: Real}

    enegzpt = exp(-z′_t)
    Renegzpt = R_t * enegzpt
    ω′_H′ = ω′_t * H′_t
    # Loop over s
    @inbounds @simd for iss = 1:ns
        # Loop over a
        @inbounds @simd for ia = 1:na
            # Index for outcome
            i  = na * (iss-1) + ia # (a, s)
            ell_euler = 0.

            # Sum over s'
            @inbounds @simd for isp = 1:ns
                # Sum over a'
                ω′_s′_H′ = ω′_H′ * sgrid[isp]

                @inbounds @simd for iap = 1:na
                    # Index for input
                    ip = na * (isp - 1) + iap # (a', s')

                    # ee = (a' - Rₜ * exp(-z') (a - cₜ(a, s)) - T') / (ω's'H')
                    ee = (agrid[iap] - Renegzpt *
                          (agrid[ia] - cfunc[i]) - t′_t) / ω′_s′_H′

                    # ι * ∑∑(exp(-z'ₜ)/c'(a', s')) * g(ee) * p(s'|s) /(ω's'H') # f[iss, isp] = p(s'|s)
                    ell_euler += (enegzpt / cfunc′[ip]) * g_fn(ee) * fgrid[iss, isp] / ω′_s′_H′
                end
            end
            ret_ell[i] = (ι * β * b_t * R_t) * ell_euler
        end
    end

    return ret_ell
end

function construct_steadystate_namedtuples(m::HetDSGEGovDebt)

    # Set up
    cas = min.(1 ./ m[:lstar].value,             # Need c(a, s) for mapping functionals to aggregates
               repeat(m.grids[:agrid].points, length(m.grids[:sgrid].points)) .+ m[:η].value)
    ι = (m.grids[:agrid].points[end] - m.grids[:agrid].points[1]) / length(m.grids[:agrid].points)
    Cstar = dot(m[:Dstar].value, cas)

    # Create NamedTuple mapping variable′ names to values
    nt′ = (# State Variables
           kf′_t  = m[:Dstar].value / ι,         # only linearized, not log-linearized
           k′_t   = log(m[:kstar]),
           i′_t1  = log((1 + m[:r]) * m[:π_star]),
           y′_t1  = log(m[:ystar]),
           w′_t1  = log(m[:ωstar]),
           I′_t1  = log(m[:xstar]),
           bg′_t  = log(m[:bg]),
           b′_t   = 0.,                # log(1)
           g′_t   = log(m[:g]),  # This makes all other equations zero, but the resource constraint
           # g′_t   = log(m[:ystar] / (Cstar + m[:xstar])) # resource constraint = 0 but many other equations nonzero
           z′_t   = m[:γ].scaledvalue, # z′_t is log(exp(z′_t)) so no need to log
           μ′_t   = 0.,                # log(1)
           λ_w′_t = 0.,               # log(1)
           λ_f′_t = 0.,                # log(1)
           rm′_t  = 0.,                # rm′_t is log(exp(rm′_t)) so no need to log

           # Jump Variables
           l′_t         = log.(m[:lstar].value),
           C′_t         = log(Cstar),
           R′_t         = log(1 + m[:r]),
           i′_t         = log((1 + m[:r]) * m[:π_star]),
           t′_t         = m[:Tstar].value,       # is currently set to the level b/c it is negative
           w′_t         = log(m[:ωstar]),
           L′_t         = log(m[:H]),
           π′_t         = log(m[:π_star]),
           π_w′_t       = log(m[:π_star] * exp(m[:γ])),
           margutil′_t  = log(dot(m[:Dstar].value, (1 ./ cas))), # average marginal utility used by union to set wages
           y′_t         = log(m[:ystar]),
           I′_t         = log(m[:xstar]),
           mc′_t        = 0.,                               # log(1)
           Q′_t         = 0.,                               # log(1)
           capreturn′_t = log(m[:Rkstar]),
           tg′_t        = log(m[:Tg]))

    # Create NamedTuple mapping variable names to same values as variable′ names
    # Could be made into a macro (mirror the syntax for UnPack; it would mostly be
    # writing [DSGE.unprime(k) = nt.k for k in keys(nt′)] but adjusted for the fact
    # that macros operate on expressions (but w/ NamedTuples, you have the keys
    # as part of the Type Parameter and just need to access the field nt.k
    nt = (# State Variables
           kf_t  = nt′[:kf′_t],
           k_t   = nt′[:k′_t],
           i_t1  = nt′[:i′_t1],
           y_t1  = nt′[:y′_t1],
           w_t1  = nt′[:w′_t1],
           I_t1  = nt′[:I′_t1],
           bg_t  = nt′[:bg′_t],
           b_t   = nt′[:b′_t],
           g_t   = nt′[:g′_t],
           z_t   = nt′[:z′_t],
           μ_t   = nt′[:μ′_t],
           λ_w_t = nt′[:λ_w′_t],
           λ_f_t = nt′[:λ_f′_t],
           rm_t  = nt′[:rm′_t],

           # Jump Variables
           l_t         = nt′[:l′_t],
           C_t         = nt′[:C′_t],
           R_t         = nt′[:R′_t],
           i_t         = nt′[:i′_t],
           t_t         = nt′[:t′_t],
           w_t         = nt′[:w′_t],
           L_t         = nt′[:L′_t],
           π_t         = nt′[:π′_t],
           π_w_t       = nt′[:π_w′_t],
           margutil_t  = nt′[:margutil′_t],
           y_t         = nt′[:y′_t],
           I_t         = nt′[:I′_t],
           mc_t        = nt′[:mc′_t],
           Q_t         = nt′[:Q′_t],
           capreturn_t = nt′[:capreturn′_t],
           tg_t        = nt′[:tg′_t])

    return nt′, nt
end
