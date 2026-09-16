using LinearAlgebra
using NLsolve

"""
    update_ss_csc!(SS_stats, param, param_update, grid)

Update CSC steady-state statistics and parameters after estimated parameters
are changed. Mutates `SS_stats` and `param` in place and returns copies of the
pre-update dictionaries.
"""
function update_ss_csc!(SS_stats, param, param_update, grid)
    SS_stats_base = copy(SS_stats)
    param_base = copy(param)

    iota_1_update = haskey(param_update, "iota_1") ? param_update["iota_1"] : param_update["iota"]
    iota_2_update = haskey(param_update, "iota_2") ? param_update["iota_2"] : param_update["iota"]

    param["iota_1"] = iota_1_update
    param["iota_2"] = iota_2_update

    ns_1 = Int(grid["ns_1"])
    ns_2 = Int(grid["ns_2"])
    ns = Int(grid["ns"])

    r_l_1 = SS_stats["r_l_1"]
    r_l_2 = SS_stats["r_l_2"]
    V_1 = SS_stats["V_1"]
    V_2 = SS_stats["V_2"]
    M_1 = SS_stats["M_1"]
    M_2 = SS_stats["M_2"]
    s = vec(grid["s"])
    se_dist = vec(grid["se_dist"])

    lambda_aux = [
        (1 - param["lambda_1"]) * Matrix(1.0I, ns_1, ns_1) zeros(ns_1, ns_2);
        zeros(ns_2, ns_1) (1 - param["lambda_2"]) * Matrix(1.0I, ns_2, ns_2)
    ]
    A = Matrix(1.0I, ns, ns) -
        param["beta"] * (1 - param["death_rate"]) * (1 - param["in"]) *
        param["P_SS"] * lambda_aux

    function job_value_stats(fix_L_1, fix_L_2)
        J_inst = vcat(
            (r_l_1 - fix_L_1 - param["w_bar_1"]) * param["n_1"] .* s[1:ns_1],
            (r_l_2 - fix_L_2 - param["w_bar_2"]) * param["n_2"] .* s[(ns_1 + 1):ns],
        )
        J = A \ J_inst
        J_bar_1 = dot(J[1:ns_1], se_dist[1:ns_1]) / sum(se_dist[1:ns_1])
        J_bar_2 = dot(J[(ns_1 + 1):ns], se_dist[(ns_1 + 1):ns]) / sum(se_dist[(ns_1 + 1):ns])
        return J, J_bar_1, J_bar_2
    end

    function find_fix_Ls!(F, x)
        _, J_bar_1, J_bar_2 = job_value_stats(x[1], x[2])
        F[1] = param["iota_1"] - (M_1 / V_1) * J_bar_1
        F[2] = param["iota_2"] - (M_2 / V_2) * J_bar_2
        return nothing
    end

    sol = nlsolve(find_fix_Ls!, [0.05, 0.05]; ftol=1e-15, xtol=1e-15, show_trace=false)
    if !(sol.f_converged || sol.x_converged)
        error("update_ss_csc!: NLsolve failed to converge for fix_L_1/fix_L_2")
    end

    param["fix_L_1"] = sol.zero[1]
    param["fix_L_2"] = sol.zero[2]

    J, J_bar_1, J_bar_2 = job_value_stats(param["fix_L_1"], param["fix_L_2"])
    SS_stats["J"] = J
    SS_stats["J_bar_1"] = J_bar_1
    SS_stats["J_bar_2"] = J_bar_2

    SS_stats["Profit_L"] =
        (r_l_1 - param["fix_L_1"] - param["w_bar_1"]) * grid["L_1"] -
        param["iota_1"] * V_1 +
        (r_l_2 - param["fix_L_2"] - param["w_bar_2"]) * grid["L_2"] -
        param["iota_2"] * V_2

    param["fix"] =
        SS_stats["Y"] - SS_stats_base["mc"] * SS_stats_base["Y"] +
        SS_stats["Profit_L"] + SS_stats["Profit_K"] - SS_stats_base["Profit"]

    SS_stats["Profit"] =
        SS_stats["Y"] - SS_stats_base["mc"] * SS_stats_base["Y"] +
        SS_stats["Profit_L"] + SS_stats["Profit_K"] - param["fix"]

    SS_stats["Profit_int"] = (1 - SS_stats["mc"]) * SS_stats["Y"] - param["fix"]

    SS_stats["AvgC"] =
        (param["delta_ss"] * grid["K"] +
         param["w_bar_1"] * grid["L_1"] +
         param["w_bar_2"] * grid["L_2"] +
         param["iota_1"] * V_1 +
         param["iota_2"] * V_2 +
         param["fix"]) / SS_stats["Y"]

    param["fix_ratio"] = param["fix"] / SS_stats["Y"]

    R = get(SS_stats, "R", param["R_cb"] / param["pi_cb"])
    R_a = get(SS_stats, "R_a", (param["q"] + SS_stats["r_a"]) / param["q"])
    THETA = get(SS_stats, "leverage", param["THETA"])
    Lambda_b = get(SS_stats, "Lambda_b", param["Lambda_bank"])
    Ab = get(SS_stats, "A_b", SS_stats["NW_b"] * THETA)

    zz = (R_a - R) * THETA + R
    param["omega"] = (1 - param["theta_b"] * zz) / THETA
    xx = zz
    ee = (1 - param["theta_b"]) * Lambda_b * R / (1 - param["theta_b"] * zz * Lambda_b)
    vv = ((1 - param["theta_b"]) * (R_a - R) * Lambda_b) / (1 - param["theta_b"] * xx * Lambda_b)
    NWb = Ab / THETA
    Profit_FI = (1 - param["theta_b"]) * ((R_a - R) * THETA + R) * NWb - param["omega"] * param["q"] * Ab
    B_FI = (THETA - 1) * NWb
    B_MMMF = B_FI
    param["DELTA"] = vv + ee / THETA
    param["B_MMMF_ratio"] = B_MMMF / max(B_FI + get(SS_stats, "B_gov_ncp", 0.0) + get(SS_stats, "B_g", 0.0), eps())

    SS_stats["zz"] = zz
    SS_stats["xx"] = xx
    SS_stats["ee"] = ee
    SS_stats["vv"] = vv
    SS_stats["NW_b"] = NWb
    SS_stats["Profit_FI"] = Profit_FI
    SS_stats["B_FI"] = B_FI
    SS_stats["B_MMMF"] = B_MMMF
    SS_stats["B_b"] = B_MMMF
    SS_stats["market_clearing_diagnostics"] = Dict(
        "B_MMMF_minus_B_FI" => B_MMMF - B_FI,
        "B_b_minus_B_MMMF" => get(SS_stats, "B_b", B_MMMF) - B_MMMF,
    )

    return SS_stats_base, param_base
end
