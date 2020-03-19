function jacobian_ad(m::HetDSGEGovDebt, x::Vector{Float64})

    include_functional_eqs = get_setting(m, :autodiff_include_functional_eqs)

    α      = m[:α].value::Float64
    δ      = m[:δ].value::Float64
    γ      = m[:γ].scaledvalue::Float64
    η      = m[:η].value::Float64
    ρ_B    = m[:ρ_b].value::Float64
    ρ_G    = m[:ρ_g].value::Float64
    ρ_z    = m[:ρ_z].value::Float64
    ρ_μ    = m[:ρ_μ].value::Float64
    ρ_lamw = m[:ρ_λ_w].value::Float64
    ρ_lamf = m[:ρ_λ_f].value::Float64
    ρ_mon  = m[:ρ_rm].value::Float64
    spp    = m[:spp].value::Float64
    ϕh     = m[:ϕh].value::Float64
    ρ_R    = m[:ρR].value::Float64
    ψπ     = m[:ψπ].value::Float64
    ψy     = m[:ψy].value::Float64
    κ_p    = m[:κ_p].value::Float64
    κ_w    = m[:κ_w].value::Float64

    Tg     = m[:Tg].value::Float64
    δb     = m[:δb].value::Float64
    bg     = m[:bg].value::Float64

    π_star = m[:π_star].value::Float64

    β      = m[:βstar].value::Float64

    # Temp
    Φ_w = 1.
    ϕ   = 1.
    τw  = 1.
    π_star_w = 1.
    Φ_p = 1.
    τF  = 1.
    iii = 1.

    xgrid = m.grids[:xgrid].points::Vector{Float64}
    xwts  = m.grids[:xgrid].weights::Vector{Float64}
    sgrid = m.grids[:sgrid].points::Vector{Float64}
    swts  = m.grids[:sgrid].weights::Vector{Float64}
    fgrid = m.grids[:fgrid]::Matrix{Float64}
    xswts = kron(swts, xwts)::Vector{Float64}

    zlo = m[:zlo].value::Float64
    zhi = m[:zhi].value::Float64

    nx   = DSGE.get_setting(m, :nx)::Int64
    ns   = DSGE.get_setting(m, :ns)::Int64
    nxns = nx*ns::Int64

    xgrη = repeat(xgrid, ns) .+ η

    @inline function mollifier(z, ehi, elo)
        if z < ehi && z > elo
            return (2.0 / (ehi - elo)) *
                exp(-1.0 / (1.0 - (-1.0 + 2.0 * (z - elo) /
                                   (ehi - elo))^2)) / 0.443993816237631

        end
        return 0.0
    end

    @inline q_fn(x) = mollifier(x, zhi, zlo)

    @inline function s_fn(x::S) where S<:Real
        return (spp / 2.0) * x ^ 2.0
    end
    @inline function s_fn_prime(x::S) where S<:Real
        return spp * x
    end

    @inline function euler_equation(# nx::Int, ns::Int,
                                    # q_fn::Function,
                                    # xgrid::Vector{Float64},
                                    # sgrid::Vector{Float64},
                                    # fgrid::Matrix{Float64},
                                    # xswts::Vector{Float64},
                                    ret_ell::Vector{Number},
                                    R_t::S, z′_t::S, #β::Float64,
                                    #η::Float64,
                                    #l_t::Vector{S}, l′_t::Vector{S},
                                    t′_t::S, ω′_t::S, L′_t::S,
                                    cfunc::Vector{S},
                                    cfunc′::Vector{S}) where S<:Real
        nxns = nx*ns
        ezt = exp(-z′_t)
        Rezt = (1+R_t) * ezt
        ωLt = ω′_t * L′_t
        # Loop over s
        @inbounds @simd for iss=1:ns
            # Loop over i
            @inbounds @simd for ia=1:nx
                # Index for outcome
                i  = nx * (iss-1) + ia
                ell_euler = 0.
                # Sum over s'
                @inbounds @simd for isp=1:ns
                    # Sum over j
                    ωLt_s_isp = ωLt * sgrid[isp]

                    @inbounds @simd for iap=1:nx
                        # Index for input
                        ip = nx*(isp-1)+iap

                        # ee = (aⱼ - (1+Rₜ)*e⁻ᶻ'(aⱼ - cₜ(aⱼ, s)) - T')/(w's'H')
                        ee = (xgrid[iap] - Rezt *
                              (xgrid[iap] - cfunc[ip]) - t′_t) / ωLt_s_isp

                        # ι*∑∑(e⁻ᶻ'/c'(aⱼ,s'))*g(ee)*p(s'|s)/(w'*s'*H')
                        ell_euler += xswts[ip]*(ezt / cfunc'[ip]) * q_fn(ee) *
                            (fgrid[iss,isp]) / ωLt_s_isp
                    end
                end
                ret_ell[i] = ell_euler
            end
        end
        return ret_ell
    end

    @inline function kolmogorov_fwd_eq(# nx::Int, ns::Int,
                                       # q_fn::Function,
                                       # xgrid::Vector{Float64},
                                       # sgrid::Vector{Float64},
                                       # fgrid::Matrix{Float64},
                                       # xswts::Vector{Float64},
                                       ret_kol::Vector{Real},
                                       R_t::S, z′_t::S,
                                       #η::Float64,
                                       kf_t::Vector{S},
                                       #l_t::Vector{S},
                                       t′_t::S, w′_t::S,
                                       H′_t::Real, cfunc::Vector{S}) where S<:Real

        nxns = nx*ns
        Rezt = (1+R_t)*(exp(-z′_t))
        wHt  = w′_t * H′_t
        # Loop over j
        @inbounds @simd for isp=1:ns
            wHt_s_isp = wHt * sgrid[isp]
            # Loop over s'
            @inbounds @simd for iap=1:nx
                m_kolmogorov = 0.
                # Index for outcome
                ip  = nx*(isp-1)+iap
                xgrid_iap = xgrid[iap]

                # Sum over i
                @inbounds @simd for ia=1:nx
                    # Sum over s
                    @inbounds @simd for iss=1:ns
                        # Index for input
                        i = nx*(iss-1)+ia
                        # (aⱼ - (1+rₜ)e⁻ᶻ'(aᵢ - cₜ(aᵢ, s) - T')/ w's'H'
                        ee = (xgrid_iap - Rezt *
                              (xgrid[ia] - cfunc[ip]) - t′_t) / wHt_s_isp
                        # mₜ(aᵢ, s) * g(ee) * p(s'|s)/w's'H' # f[iss, isp] = p(isp|iss)
                        m_kolmogorov += xswts[i] * kf_t[i] * q_fn(ee) *
                            (fgrid[iss, isp]) / wHt_s_isp
                    end
                end
                # Set index j (outcome)
                ret_kol[ip] = m_kolmogorov
            end
        end
        return ret_kol
    end

    function F(x::Vector{S}) where {S<:Real}

        endo = DSGE.augment_model_states(m.endogenous_states_original,
                                         DSGE.n_model_states_original(m))

        kf′_t = x[1:600]

        k′_t   = x[601:601][1]
        i′_t1  = x[602:602][1]
        y′_t1  = x[603:603][1]
        w′_t1  = x[604:604][1]
        I′_t1  = x[605:605][1]
        bg′_t  = x[606:606][1]
        b′_t   = x[607:607][1]
        g′_t   = x[608:608][1]
        z′_t   = x[609:609][1]
        μ′_t   = x[610:610][1]
        λ_w′_t = x[611:611][1]
        λ_f′_t = x[612:612][1]
        rm′_t  = x[613:613][1]

        l′_t   = x[614:1213]

        C′_t   = x[1214:1214][1]
        R′_t   = x[1215:1215][1]
        i′_t   = x[1216:1216][1]
        t′_t   = x[1217:1217][1]
        w′_t   = x[1218:1218][1]
        L′_t   = x[1219:1219][1]
        π′_t   = x[1220:1220][1]
        π_w′_t = x[1221:1221][1]
        margutil′_t = x[1222:1222][1]
        y′_t   = x[1223:1223][1]
        I′_t   = x[1224:1224][1]
        mc′_t  = x[1225:1225][1]
        Q′_t   = x[1226:1226][1]
        capreturn′_t = x[1227:1227][1]
        tg′_t  = x[1228:1228][1]

        kf_t  = x[1229:1828]

        k_t   = x[1829:1829][1]
        i_t1  = x[1830:1830][1]
        y_t1  = x[1831:1831][1]
        w_t1  = x[1832:1832][1]
        I_t1  = x[1833:1833][1]
        bg_t  = x[1834:1834][1]
        b_t   = x[1835:1835][1]
        g_t   = x[1836:1836][1]
        z_t   = x[1837:1837][1]
        μ_t   = x[1838:1838][1]
        λ_w_t = x[1839:1839][1]
        λ_f_t = x[1840:1840][1]
        rm_t  = x[1841:1841][1]
        l_t   = x[1842:2441]

        C_t   = x[2442:2442][1]
        R_t   = x[2443:2443][1]
        i_t   = x[2444:2444][1]
        t_t   = x[2445:2445][1]
        w_t   = x[2446:2446][1]
        L_t   = x[2447:2447][1]
        π_t   = x[2448:2448][1]
        π_w_t = x[2449:2449][1]
        margutil_t = x[2450:2450][1]
        y_t   = x[2451:2451][1]
        I_t   = x[2452:2452][1]
        mc_t  = x[2453:2453][1]
        Q_t   = x[2454:2454][1]
        capreturn_t = x[2455:2455][1]
        tg_t  = x[2456:2456][1]

        cfunc′ = min.(1 ./ l′_t, xgrη)
        cfunc  = min.(1 ./ l_t,  xgrη)

        ezt = exp(-z_t)

        #@show "Euler Eq."
        ell_euler_eq = Vector{Number}(undef, nxns)
        ell_euler_eq = euler_equation(ell_euler_eq, R′_t, z′_t, t′_t, w′_t,
                                      L′_t, cfunc, cfunc′)

        #@show "KF"
        # KF Equation
        kf_eq = Vector{Real}(undef, nxns)
        kf_eq = kolmogorov_fwd_eq(kf_eq, R′_t, z′_t, kf_t, t′_t, w′_t, L′_t, cfunc)

        eq_euler           = β*b′_t*(1+R_t)*ell_euler_eq - l′_t
        eq_kolmogorov_fwd  = kf_eq - kf′_t

        eq_agg_consumption = (xswts .* kf_t)' * cfunc - C_t
        eq_lambda          = margutil_t - (xswts .* kf_t)'*(1 ./ cfunc)
        eq_transfers       = capreturn_t * k_t * ezt - I_t + (1-mc_t) * y_t - tg_t - t_t

        # Note: this beta is listed as βtil which differs from ̱β but Ethan thinks it
        # drops out when take derivative so doesnt matter
        eq_investment = margutil_t * Q_t * μ_t * (1- s_fn(I_t / I_t1 * ezt)) +
            β*margutil′_t * exp(-z′_t) * Q′_t * μ′_t * s_fn_prime(I′_t/I_t*exp(z′_t)) *
            (I′_t/I_t*exp(z′_t))^2 - margutil_t *
            (1 + Q_t * μ_t*s_fn_prime(I_t/I_t1*exp(z_t)) * I_t/I_t1*exp(z_t))

        eq_tobin_q = β * margutil′_t * exp(-z′_t) / margutil_t *
            (capreturn′_t + Q′_t * (1-δ))

        eq_capital_accumulation = k′_t - (1-δ) * k_t * ezt -
            μ_t * (1 - s_fn(I_t/I_t1 * exp(z_t))) * I_t

        # TODO: Figure out how to d\eal with kappa_w and kappa_p coefficients
        ### Rows 1207 and 1208 LEAVE ALONE till hear from keshav
        eq_wage_phillips =  1/(λ_w_t*Φ_w) * ((1+λ_w_t)*ϕ*L_t^ϕh - (1-τw)*margutil_t * w_t) +
            β*b′_t*(π_w′_t/π_star_w - 1)*(π_w′_t/π_star_w)*(L′_t/L_t) -
            (π_w_t/π_star_w - 1)*(π_w_t/π_star_w)

        eq_price_phillips = -1/(λ_w_t*Φ_p)*((1+τF) - (1+λ_f_t)*mc_t) +
            (1/(1 + R_t))*(π′_t/π_t -1)*(π′_t/π_star)*y′_t*exp(z′_t)*y_t -
            (π_t/π_star -1)*(π_t/π_star) # see above

        eq_marginal_cost = ((1-α)^(1-α)*α^α)^(-1)*w_t^(1-α)*capreturn_t^α - mc_t
        eq_gdp           = exp(-α*z_t)*k_t^α*L_t^(1-α) - y_t
        eq_optimal_kl    = α/(1-α)*L_t/k_t*ezt - capreturn_t/w_t
        eq_taylor        = ((1+i_t) / (1+iii)) ^ ρ_R *
            ((π / π_star) ^ ψπ * (y_t / y_t1 * exp(-γ)) ^ ψy) ^ (1-ρ_R) *
            exp(rm′_t) - (1+i_t) / (1+iii)

        # This is not equal because we're not log-linearizing the equations
        eq_fisher = (1+R_t) - (1 + i_t) / π′_t

        eq_nominal_wage_inflation = π_t * exp(z_t) * w_t / w_t1

        eq_fiscal_rule = δb * (bg_t * ezt + (1 - 1/g_t) * y_t - bg / (1+R_t)) +
            (1-δ)*Tg - tg_t

        eq_g_budget_constraint = bg_t * ezt + g_t - tg_t - bg′_t * bg / (1+R_t)

        eq_resource_constraint = y_t/g_t - C_t - I_t

        eq_LI  = i′_t1 - i_t
        eq_LY  = y′_t1 - y_t
        eq_LW  = w′_t1 - w_t
        eq_LX  = I′_t1 - I_t

        # Add shock for this & below; doesn't matter b/c take derivative
        eq_b   = b′_t - ρ_B * b_t
        eq_g   = g′_t - ρ_G * g_t
        eq_z   = z′_t - ρ_z * z_t
        eq_μ   = μ′_t - ρ_μ * μ_t
        eq_λ_w = λ_w′_t - ρ_lamw * λ_w_t
        eq_λ_f = λ_f′_t - ρ_lamf * λ_f_t
        eq_rm  = rm′_t  - ρ_mon * rm_t
        return [eq_euler;
                eq_kolmogorov_fwd;
                eq_agg_consumption; eq_lambda; eq_transfers;
                eq_investment; eq_tobin_q; eq_capital_accumulation; eq_wage_phillips;
                eq_price_phillips; eq_marginal_cost; eq_gdp; eq_optimal_kl; eq_taylor;
                eq_fisher;
                eq_nominal_wage_inflation; eq_fiscal_rule; eq_g_budget_constraint;
                eq_resource_constraint;
                eq_LI; eq_LY; eq_LW; eq_LX;
                eq_b; eq_g; eq_z; eq_μ; eq_λ_w; eq_λ_f; eq_rm]
    end

    function F_light(x::Vector{S}) where {S<:Real}

        endo = DSGE.augment_model_states(m.endogenous_states_original,
                                         DSGE.n_model_states_original(m))

        kf′_t = x[1:600]

        k′_t   = x[601:601][1]
        i′_t1  = x[602:602][1]
        y′_t1  = x[603:603][1]
        w′_t1  = x[604:604][1]
        I′_t1  = x[605:605][1]
        bg′_t  = x[606:606][1]
        b′_t   = x[607:607][1]
        g′_t   = x[608:608][1]
        z′_t   = x[609:609][1]
        μ′_t   = x[610:610][1]
        λ_w′_t = x[611:611][1]
        λ_f′_t = x[612:612][1]
        rm′_t  = x[613:613][1]

        l′_t   = x[614:1213]

        C′_t   = x[1214:1214][1]
        R′_t   = x[1215:1215][1]
        i′_t   = x[1216:1216][1]
        t′_t   = x[1217:1217][1]
        w′_t   = x[1218:1218][1]
        L′_t   = x[1219:1219][1]
        π′_t   = x[1220:1220][1]
        π_w′_t = x[1221:1221][1]
        margutil′_t = x[1222:1222][1]
        y′_t   = x[1223:1223][1]
        I′_t   = x[1224:1224][1]
        mc′_t  = x[1225:1225][1]
        Q′_t   = x[1226:1226][1]
        capreturn′_t = x[1227:1227][1]
        tg′_t  = x[1228:1228][1]

        kf_t  = x[1229:1828]

        k_t   = x[1829:1829][1]
        i_t1  = x[1830:1830][1]
        y_t1  = x[1831:1831][1]
        w_t1  = x[1832:1832][1]
        I_t1  = x[1833:1833][1]
        bg_t  = x[1834:1834][1]
        b_t   = x[1835:1835][1]
        g_t   = x[1836:1836][1]
        z_t   = x[1837:1837][1]
        μ_t   = x[1838:1838][1]
        λ_w_t = x[1839:1839][1]
        λ_f_t = x[1840:1840][1]
        rm_t  = x[1841:1841][1]
        l_t   = x[1842:2441]

        C_t   = x[2442:2442][1]
        R_t   = x[2443:2443][1]
        i_t   = x[2444:2444][1]
        t_t   = x[2445:2445][1]
        w_t   = x[2446:2446][1]
        L_t   = x[2447:2447][1]
        π_t   = x[2448:2448][1]
        π_w_t = x[2449:2449][1]
        margutil_t = x[2450:2450][1]
        y_t   = x[2451:2451][1]
        I_t   = x[2452:2452][1]
        mc_t  = x[2453:2453][1]
        Q_t   = x[2454:2454][1]
        capreturn_t = x[2455:2455][1]
        tg_t  = x[2456:2456][1]

        cfunc′ = min.(1 ./ l′_t, xgrη)
        cfunc  = min.(1 ./ l_t,  xgrη)

        ezt = exp(-z_t)

        eq_agg_consumption = (xswts .* kf_t)' * cfunc - C_t
        eq_lambda          = margutil_t - (xswts .* kf_t)'*(1 ./ cfunc)
        eq_transfers       = capreturn_t * k_t * ezt - I_t + (1-mc_t) * y_t - tg_t - t_t

        # Note: this beta is listed as βtil which differs from ̱β but Ethan thinks it
        # drops out when take derivative so doesnt matter
        eq_investment = margutil_t * Q_t * μ_t * (1- s_fn(I_t / I_t1 * ezt)) +
            β*margutil′_t * exp(-z′_t) * Q′_t * μ′_t * s_fn_prime(I′_t/I_t*exp(z′_t)) *
            (I′_t/I_t*exp(z′_t))^2 - margutil_t *
            (1 + Q_t * μ_t*s_fn_prime(I_t/I_t1*exp(z_t)) * I_t/I_t1*exp(z_t))

        eq_tobin_q = β * margutil′_t * exp(-z′_t) / margutil_t *
            (capreturn′_t + Q′_t * (1-δ))

        eq_capital_accumulation = k′_t - (1-δ) * k_t * ezt -
            μ_t * (1 - s_fn(I_t/I_t1 * exp(z_t))) * I_t

        # TODO: Figure out how to d\eal with kappa_w and kappa_p coefficients
        ### Rows 1207 and 1208 LEAVE ALONE till hear from keshav
        eq_wage_phillips =  1/(λ_w_t*Φ_w) * ((1+λ_w_t)*ϕ*L_t^ϕh - (1-τw)*margutil_t * w_t) +
            β*b′_t*(π_w′_t/π_star_w - 1)*(π_w′_t/π_star_w)*(L′_t/L_t) -
            (π_w_t/π_star_w - 1)*(π_w_t/π_star_w)

        eq_price_phillips = -1/(λ_w_t*Φ_p)*((1+τF) - (1+λ_f_t)*mc_t) +
            (1/(1 + R_t))*(π′_t/π_t -1)*(π′_t/π_star)*y′_t*exp(z′_t)*y_t -
            (π_t/π_star -1)*(π_t/π_star) # see above

        eq_marginal_cost = ((1-α)^(1-α)*α^α)^(-1)*w_t^(1-α)*capreturn_t^α - mc_t
        eq_gdp           = exp(-α*z_t)*k_t^α*L_t^(1-α) - y_t
        eq_optimal_kl    = α/(1-α)*L_t/k_t*ezt - capreturn_t/w_t
        # TYPO in paper: e^z_t vs. e^-γ
        eq_taylor        = ((1+i_t1) / (1+iii)) ^ ρ_R *
            ((π / π_star) ^ ψπ * (y_t / y_t1 * ezt) ^ ψy) ^ (1-ρ_R) *
            exp(rm_t) - (1+i_t) / (1+iii)

        # This is not equal because we're not log-linearizing the equations
        eq_fisher = (1+R_t) - (1 + i_t) / π′_t

        eq_nominal_wage_inflation = π_w_t - π_t * exp(z_t) * w_t / w_t1

        eq_fiscal_rule = δb * (bg_t * ezt + (1 - 1/g_t) * y_t - bg / (1+R_t)) + (1-δ)*Tg - tg_t

        eq_g_budget_constraint = bg_t * ezt + g_t - tg_t - bg′_t * bg / (1+R_t)

        eq_resource_constraint = y_t/g_t - C_t - I_t

        eq_LI  = i′_t1 - i_t
        eq_LY  = y′_t1 - y_t
        eq_LW  = w′_t1 - w_t
        eq_LX  = I′_t1 - I_t

        # Add shock for this & below; doesn't matter b/c take derivative
        eq_b   = b′_t - ρ_B * b_t
        eq_g   = g′_t - ρ_G * g_t
        eq_z   = z′_t - ρ_z * z_t
        eq_μ   = μ′_t - ρ_μ * μ_t
        eq_λ_w = λ_w′_t - ρ_lamw * λ_w_t
        eq_λ_f = λ_f′_t - ρ_lamf * λ_f_t
        eq_rm  = rm′_t  - ρ_mon * rm_t
        return [eq_agg_consumption; eq_lambda; eq_transfers;
                eq_investment; eq_tobin_q; eq_capital_accumulation; eq_wage_phillips;
                eq_price_phillips; eq_marginal_cost; eq_gdp; eq_optimal_kl;
                eq_taylor; eq_fisher;
                eq_nominal_wage_inflation; eq_fiscal_rule; eq_g_budget_constraint;
                eq_resource_constraint;
                eq_LI; eq_LY; eq_LW; eq_LX;
                eq_b; eq_g; eq_z; eq_μ; eq_λ_w; eq_λ_f; eq_rm]
    end

    JJ = if include_functional_eqs
        ForwardDiff.jacobian(F, x)
    else
        vcat(jacobian(m; functional_eqs_only = true), ForwardDiff.jacobian(F_light, x))
    end
    if !m.testing && get_setting(m, :normalize_distr_variables)
        JJ  = normalize(m, JJ)
    end
    return JJ

end
