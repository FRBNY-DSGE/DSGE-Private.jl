using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "..", "..", "policies_SS.jl"))
include(joinpath(@__DIR__, "..", "..", "find_dist.jl"))
include(joinpath(@__DIR__, "..", "..", "CalValueSS.jl"))

function compute_K(K::Float64, c_a_guess::Array{Float64,3}, c_n_guess::Array{Float64,3},
                   psi_guess::Array{Float64,3}, AProb::Array{Float64,3}, mu_dist::Array{Float64,3},
                   grid::Dict{String,Any}, param::Dict{String,Any}, meshes::Dict{String,Array{Float64,3}})
    grid["K"] = K

    dist_PFI = 1000.0
    dist_AP = 1000.0
    iter_AP = 0
    tol_AP = 1e-10
    max_iter_AP = 500

    mc, L_1, L_2, n_1, n_2, r_l_1, r_l_2, r_k, Profit, v, J_aux, u_1, u_2, V_1, V_2, M_1, M_2, f_1, f_2, w_1, w_2,
    WW, RR, RBRB, Y, P_SE, Lambda, param, grid, THETA, NWb, Profit_FI, Ab, B_FI, B_MMMF, T_MMMF, B_gov, B_g, K_g_agg,
    TAX_revenue, UB = compute_agg(meshes, grid, param)

    _, _, _, inc = policyguess(meshes, WW, RR, RBRB, param, grid)

    AProb_new = copy(AProb)
    Value = zeros(size(AProb))
    H = spzeros(length(AProb[:]), length(AProb[:]))

    b_n_star = zeros(size(c_n_guess))
    b_a_star = zeros(size(c_a_guess))
    a_a_star = zeros(size(c_a_guess))

    while dist_AP > tol_AP && iter_AP < max_iter_AP
        iter_AP += 1

        c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, dist_PFI =
            policies_SS(c_a_guess, c_n_guess, psi_guess, AProb, grid, inc, RR, RBRB, P_SE, param, meshes)

        AProb_new, Value, H = CalValueSS(b_n_star, b_a_star, a_a_star, c_a_guess, c_n_guess, AProb, P_SE, param, grid)

        dist_AP = maximum(abs.(AProb_new[:] .- AProb[:]))
        AProb = AProb_new
    end

    n_states = Int(grid["nb"]) * Int(grid["na"])
    I_sparse = sparse(1:n_states, 1:n_states, ones(n_states), n_states, n_states)
    H_tilde = kron(sparse(param["P_SS3"]'), I_sparse)
    P_SE_dist = param["P_SS2"]

    mu_dist, _, TT = find_dist(mu_dist, H, AProb, b_n_star, b_a_star, a_a_star, H_tilde, P_SE_dist, param, grid)
    mu_dist = reshape(mu_dist, (Int(grid["nb"]), Int(grid["na"]), Int(grid["nse"])))

    mu_dist_tilde = reshape(H_tilde * mu_dist[:], (Int(grid["nb"]), Int(grid["na"]), Int(grid["nse"])))

    K_NI_agg = sum(mu_dist_tilde[:] .* AProb[:] .* a_a_star[:] .+ mu_dist_tilde[:] .* (1 .- AProb[:]) .* meshes["a"][:])
    K_NI_agg = (1 - param["death_rate"]) * K_NI_agg

    B_NI_agg = sum(mu_dist_tilde[:] .* AProb[:] .* b_a_star[:] .+ mu_dist_tilde[:] .* (1 .- AProb[:]) .* b_n_star[:])
    B_NI_agg = (1 - param["death_rate"]) * B_NI_agg

    K_g_agg = param["A_g_ratio"] * Y
    B_g = K_g_agg

    K_I_agg = grid["K"] - K_NI_agg - K_g_agg
    param["Ab_ratio"] = K_I_agg / grid["K"]

    if K_I_agg < 0
        @warn "A negative equity holding by FIs was encountered."
    end

    R = param["R_cb"] / param["pi_cb"]
    R_a = (param["q"] + RR) / param["q"]
    Ab = K_I_agg
    THETA = param["THETA"]

    param["omega"] = (1 - param["theta_b"] * ((R_a - R) * THETA + R)) / THETA

    zz = (R_a - R) * THETA + R
    xx = zz
    ee = (1 - param["theta_b"]) * Lambda * R / (1 - param["theta_b"] * zz * Lambda)
    vv = ((1 - param["theta_b"]) * (R_a - R) * Lambda) / (1 - param["theta_b"] * xx * Lambda)
    param["DELTA"] = vv + ee / THETA

    NWb = Ab / THETA
    Profit_FI = (1 - param["theta_b"]) * ((R_a - R) * THETA + R) * NWb - param["omega"] * param["q"] * Ab
    B_FI = (THETA - 1) * NWb

    B_gov = B_NI_agg - B_g
    param["B_gov_ratio"] = B_gov / Y
    B_MMMF = B_FI
    param["B_MMMF_ratio"] = B_MMMF / max(B_FI + B_gov + B_g, eps())

    ns = Int(grid["ns"])
    tau_L = param["tau_L"]
    tau_P = param["tau_P"]
    xi = param["xi"]
    coeff = (1 / xi + tau_P) / (1 + 1 / xi) * param["GHH"]
    se = grid["se"]
    dist = grid["se_dist"]
    emp = 1:ns
    unemp = (ns + 1):(ns + ns)

    pretax_unit = fill(coeff * param["n_1"] * param["w_bar_1"], 2 * ns + 1)
    pretax_unit[(Int(grid["ns_1"]) + 1):ns] .= coeff * param["n_2"] * param["w_bar_2"]
    pretax_unit[(ns + Int(grid["ns_1"]) + 1):(ns + ns)] .= coeff * param["n_2"] * param["w_bar_2"]
    pretax_unit[end] = (param["Eratio"] * Profit) / param["Eshare"]

    aftertax_unit = fill((1 - tau_L) * coeff * (param["n_1"] * param["w_bar_1"])^(1 - tau_P), 2 * ns + 1)
    aftertax_unit[(Int(grid["ns_1"]) + 1):ns] .= (1 - tau_L) * coeff * (param["n_2"] * param["w_bar_2"])^(1 - tau_P)
    aftertax_unit[(ns + Int(grid["ns_1"]) + 1):(ns + ns)] .= (1 - tau_L) * coeff * (param["n_2"] * param["w_bar_2"])^(1 - tau_P)
    aftertax_unit[end] = (1 - tau_L) * ((param["Eratio"] * Profit) / param["Eshare"])^(1 - tau_P)

    Lincome_pretax = dot(pretax_unit[emp] .* se[emp], dist[emp])
    UB_pretax = dot(pretax_unit[unemp] .* se[unemp], dist[unemp])
    Bincome_pretax = param["Eratio"] * Profit
    Lincome_aftertax = dot(aftertax_unit[emp] .* se[emp].^(1 - tau_P), dist[emp])
    UB_aftertax = dot(aftertax_unit[unemp] .* se[unemp].^(1 - tau_P), dist[unemp])
    Bincome_aftertax = param["Eshare"] * aftertax_unit[end]
    Income_Tax = Lincome_pretax - Lincome_aftertax + UB_pretax - UB_aftertax + Bincome_pretax - Bincome_aftertax
    TAX_revenue = Income_Tax + param["tau_a"] * (1 - param["Eratio"]) * Profit
    UB = UB_pretax
    param["Income_Tax"] = Income_Tax
    MMMF_cont = param["MMMF_cont_ratio"] * TAX_revenue
    T_MMMF = MMMF_cont + (R - 1) * B_MMMF

    K_agg = K_NI_agg + K_I_agg + K_g_agg
    excess = grid["K"] - K_agg

    return excess, c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist, AProb_new, Value, mc, L_1,
           L_2, n_1, n_2, r_l_1, r_l_2, r_k, Profit, v, J_aux, u_1, u_2, V_1, V_2, M_1, M_2, f_1, f_2, w_1, w_2, WW,
           RR, RBRB, Y, param, Lambda, TT, grid, K_NI_agg, B_NI_agg, THETA, NWb, Profit_FI, Ab, B_FI, B_MMMF, T_MMMF,
           B_gov, B_g, K_g_agg, TAX_revenue, UB
end
