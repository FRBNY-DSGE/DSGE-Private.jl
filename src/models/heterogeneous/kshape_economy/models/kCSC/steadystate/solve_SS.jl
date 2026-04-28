# include(joinpath(@__DIR__, "../../..", "helpers", "ndgrid.jl"))
# include(joinpath(@__DIR__, "compute_agg.jl"))
# # include(joinpath(@__DIR__, "../../..", "helpers", "policyguess.jl"))
# include(joinpath(@__DIR__, "compute_K.jl"))

function solve_SS(grid::Dict{String,Any}, param::Dict{String,Any})
    t_start = time()

    meshes_b, meshes_a, meshes_se = ndgrid(grid["b"], grid["a"], grid["se"])
    meshes = Dict("b" => meshes_b, "a" => meshes_a, "se" => meshes_se)

    AProb = param["guessadj"] * ones(Int(grid["nb"]), Int(grid["na"]), Int(grid["nse"]))
    mu_dist = ones(Int(grid["nb"]), Int(grid["na"]), Int(grid["nse"])) / (Int(grid["nb"]) * Int(grid["na"]) * Int(grid["nse"]))

    mc, L_1, L_2, n_1, n_2, r_l_1, r_l_2, r_k, Profit, v, J, u_1, u_2, V_1, V_2, M_1, M_2, f_1, f_2, w_1, w_2, WW, RR,
    RBRB, Y, _, Lambda, param, grid, THETA, NWb, Profit_FI, Ab, B_FI, B_MMMF, T_MMMF, B_gov, B_g, K_g_agg, TAX_revenue,
    UB = compute_agg(meshes, grid, param)

    c_a_guess, c_n_guess, psi_guess, _ = policyguess(meshes, WW, RR, RBRB, param, grid)

    excess, c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist, AProb_new, Value, mc, L_1, L_2,
    n_1, n_2, r_l_1, r_l_2, r_k, Profit, v, J, u_1, u_2, V_1, V_2, M_1, M_2, f_1, f_2, w_1, w_2, WW, RR, RBRB, Y,
    param, Lambda, TT, grid, K_NI_agg, B_NI_agg, THETA, NWb, Profit_FI, Ab, B_FI, B_MMMF, T_MMMF, B_gov, B_g, K_g_agg,
    TAX_revenue, UB = compute_K(grid["K"], c_a_guess, c_n_guess, psi_guess, AProb, mu_dist, grid, param, meshes)

    grid["L_1"] = L_1
    grid["L_2"] = L_2

    mc, L_1, L_2, n_1, n_2, r_l_1, r_l_2, r_k, Profit, v, J, u_1, u_2, V_1, V_2, M_1, M_2, f_1, f_2, w_1, w_2, WW, RR,
    RBRB, Y, _, Lambda, param, grid, THETA, NWb, Profit_FI, Ab, B_FI, B_MMMF, T_MMMF, B_gov, B_g, K_g_agg, TAX_revenue,
    UB = compute_agg(meshes, grid, param)

    _ = time() - t_start

    return c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist, AProb_new, Value, mc, L_1, L_2,
           n_1, n_2, r_l_1, r_l_2, r_k, Profit, v, J, u_1, u_2, V_1, V_2, M_1, M_2, f_1, f_2, w_1, w_2, WW, RR, RBRB,
           Y, param, Lambda, TT, grid, K_NI_agg, B_NI_agg, THETA, NWb, Profit_FI, Ab, B_FI, B_MMMF, T_MMMF, B_gov, B_g,
           K_g_agg, TAX_revenue, UB
end
