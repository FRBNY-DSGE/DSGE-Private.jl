using Interpolations

# include(joinpath(@__DIR__, "parameters.jl"))
# include(joinpath(@__DIR__, "make_grids.jl"))
# include(joinpath(@__DIR__, "solve_SS.jl"))
# include(joinpath(@__DIR__, "Cal_SS_stats.jl"))
# include(joinpath(@__DIR__, "steady_anal.jl"))

function ss_CSC()
    param = make_parameters()
    grid = make_grids(param)
    grid["K"] = 27.0

    c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist, AProb, Value, mc, L_1, L_2, n_1, n_2,
    r_l_1, r_l_2, r_k, Profit, v, J, u_1, u_2, V_1, V_2, M_1, M_2, f_1, f_2, w_1, w_2, WW, RR, RBRB, Y, param, Lambda,
    TT, grid, K_NI_agg, B_NI_agg, THETA, NWb, Profit_FI, Ab, B_FI, B_MMMF, T_MMMF, B_gov, B_g, K_g_agg, TAX_revenue,
    UB = solve_SS(grid, param)

    meshes_b, meshes_a, meshes_se = ndgrid(grid["b"], grid["a"], grid["se"])
    meshes = Dict("b" => meshes_b, "a" => meshes_a, "se" => meshes_se)

    mutil_c_n = 1.0 ./ (c_n_guess .^ param["sigma"])
    mutil_c_a = 1.0 ./ (c_a_guess .^ param["sigma"])
    mutil_c = AProb .* mutil_c_a .+ (1 .- AProb) .* mutil_c_n

    Vb = RBRB / (1 - param["death_rate"]) * param["b_a_aux"] .* mutil_c
    Vb = reshape(reshape(Vb, (Int(grid["nb"]) * Int(grid["na"]), Int(grid["nse"]))), (Int(grid["nb"]), Int(grid["na"]), Int(grid["nse"])))

    r_a_aux = RR + param["death_rate"] / (1 - param["death_rate"]) * param["q"]
    Va = AProb .* (r_a_aux + param["q"]) .* mutil_c_a .+ (1 .- AProb) .* r_a_aux .* mutil_c_n .+ (1 .- AProb) .* psi_guess
    Va = reshape(reshape(Va, (Int(grid["nb"]) * Int(grid["na"]), Int(grid["nse"]))), (Int(grid["nb"]), Int(grid["na"]), Int(grid["nse"])))

    SS_stats, grid, param = Cal_SS_stats(
        mu_dist, c_a_guess, c_n_guess, AProb, b_n_star, b_a_star, a_a_star, grid, param, meshes, TT, Y, RR, w_1, w_2,
        r_l_1, r_l_2, r_k, Profit, n_1, n_2, u_1, u_2, v, J, V_1, V_2, M_1, M_2, f_1, f_2, mc, Lambda, NWb, THETA,
        Profit_FI, Ab, B_FI, B_MMMF, T_MMMF, B_gov, B_g, K_g_agg, TAX_revenue, UB, mutil_c, mutil_c_a, mutil_c_n, Va,
        Value,
    )

    cum_dist = cumsum(cumsum(cumsum(mu_dist, dims=1), dims=2), dims=3)
    marginal_b = cumsum(dropdims(sum(sum(mu_dist, dims=2), dims=3), dims=(2, 3)))
    marginal_a = cumsum(dropdims(sum(sum(mu_dist, dims=1), dims=3), dims=(1, 3)))
    marginal_se = cumsum(dropdims(sum(sum(mu_dist, dims=2), dims=1), dims=(1, 2)))

    Copula_itp = interpolate((marginal_b, vec(marginal_a'), marginal_se), cum_dist, Gridded(Linear()))
    Copula_itp_ext = extrapolate(Copula_itp, Line())
    cop_b, cop_a, cop_se = ndgrid(marginal_b, vec(marginal_a'), marginal_se)
    Copula_fix = Copula_itp_ext.(cop_b, cop_a, cop_se)

    param["w_bar"] = SS_stats["w"]
    param["psi"] = (1 - param["tau_w"]) * param["w_bar"] * (((L_1 + L_2) / (SS_stats["N_tilde_1"] * grid["se_bar_1"] + SS_stats["N_tilde_2"] * grid["se_bar_2"]))^param["xi"])
    grid["L"] = L_1 + L_2
    n_agg = (L_1 + L_2) / (SS_stats["N_tilde_1"] * grid["se_bar_1"] + SS_stats["N_tilde_2"] * grid["se_bar_2"])

    anal_stats, SS_stats_steady, Q_dists = steady_anal(
        grid, param, meshes, SS_stats, AProb, mu_dist, n_agg, SS_stats["H_tilde"], b_n_star, b_a_star, a_a_star,
        c_n_guess, c_a_guess,
    )
    merge!(SS_stats, SS_stats_steady)

    return Dict{String, Any}(
        "c_n_guess" => c_n_guess,
        "b_n_star" => b_n_star,
        "c_a_guess" => c_a_guess,
        "b_a_star" => b_a_star,
        "a_a_star" => a_a_star,
        "psi_guess" => psi_guess,
        "mu_dist" => mu_dist,
        "AProb" => AProb,
        "Value" => Value,
        "mc" => mc,
        "L_1" => L_1,
        "L_2" => L_2,
        "n_1" => n_1,
        "n_2" => n_2,
        "r_l_1" => r_l_1,
        "r_l_2" => r_l_2,
        "r_k" => r_k,
        "Profit" => Profit,
        "v" => v,
        "J" => J,
        "u_1" => u_1,
        "u_2" => u_2,
        "V_1" => V_1,
        "V_2" => V_2,
        "M_1" => M_1,
        "M_2" => M_2,
        "f_1" => f_1,
        "f_2" => f_2,
        "w_1" => w_1,
        "w_2" => w_2,
        "WW" => WW,
        "RR" => RR,
        "RBRB" => RBRB,
        "Y" => Y,
        "Lambda" => Lambda,
        "TT" => TT,
        "grid" => grid,
        "param" => param,
        "K_NI_agg" => K_NI_agg,
        "B_NI_agg" => B_NI_agg,
        "THETA" => THETA,
        "NWb" => NWb,
        "Profit_FI" => Profit_FI,
        "Ab" => Ab,
        "B_FI" => B_FI,
        "B_MMMF" => B_MMMF,
        "T_MMMF" => T_MMMF,
        "B_gov" => B_gov,
        "B_g" => B_g,
        "K_g_agg" => K_g_agg,
        "TAX_revenue" => TAX_revenue,
        "UB" => UB,
        "mutil_c" => mutil_c,
        "mutil_c_a" => mutil_c_a,
        "mutil_c_n" => mutil_c_n,
        "Vb" => Vb,
        "Va" => Va,
        "meshes" => meshes,
        "SS_stats" => SS_stats,
        "Copula_fix" => Copula_fix,
        "anal_stats" => anal_stats,
        "Q_dists" => Q_dists,
    )
end
