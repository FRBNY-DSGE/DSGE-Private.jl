using LinearAlgebra

"""
    update_ss_v5!(SS_stats, m, param_update, grid)

Updates steady-state statistics and parameters based on estimated parameters.

This function updates various steady-state calculations including:
- Labor market parameters (iota, fix_L)
- Job matching values (J, J_bar)
- Profit calculations (Profit_L, Profit, Profit_int, Profit_FI)
- Financial intermediary parameters (theta_b, omega, DELTA)
- Fixed costs and other auxiliary parameters

# Arguments
- `SS_stats::Dict`: Dictionary containing steady-state statistics (modified in place)
- `m::mBBQ`: Model object containing parameters (modified in place)
- `param_update::Dict`: Dictionary containing parameter updates (iota, theta_b)
- `grid::Dict`: Dictionary containing grid specifications

# Returns
- `SS_stats_base::Dict`: Copy of original SS_stats before updates
- `param_base`: Placeholder (previously a copy of param dict; kept for call-site compatibility)

# Notes
The function modifies SS_stats and m in place, creating a base copy of SS_stats before modifications.
"""
function update_ss_v5!(SS_stats, m::mBBQ, param_update, grid)

    SS_stats_base = copy(SS_stats)
    param_base = nothing  # previously copy(param); retained for call-site compatibility

    # SS_stats_base["delta_ss"] = m[:δ_0]*SS_stats_base["v"]^m[:δ_1]

    # m[:β_b] = 0.9999

    m[:ι] = param_update["iota"]  # estimates parameter
    m[:θ_b] = param_update["theta_b"]
    # m[:ϕ_var] = param_update["varphi"]

    # 1) iota

    m[:fix_L] = (SS_stats["J_bar1"] - m[:ι]*SS_stats["V"]/SS_stats["M"])/SS_stats["J_bar2"]
    Main.xx[][:I] = I
    Main.xx[][:betaL] = m[:β_b]
    Main.xx[][:deathrate] = m[:dr]
    # TODO: SS/transition matrix object — handle separately
    Main.xx[][:lambda] = param_update["lambda"]
    Main.xx[][:in] = m[:in]
    # TODO: SS/transition matrix object — handle separately
    Main.xx[][:pss] = param_update["P_SS"]
    Main.xx[][:rl] = SS_stats["r_l"]
    Main.xx[][:fixL] = m[:fix_L]
    Main.xx[][:wbar] = m[:w_bar]
    # TODO: SS/transition matrix object — handle separately
    Main.xx[][:n] = param_update["n"]
    Main.xx[][:s] = grid["s"]
    # TODO: SS/transition matrix object — handle separately
    SS_stats["J"] = inv(I - m[:β_b]*(1-m[:dr])*(1-param_update["lambda"])*
                        (1-m[:in])*param_update["P_SS"]) * ((SS_stats["r_l"] - m[:fix_L] - m[:w_bar])
                                                          .* param_update["n"] .* grid["s"])  # value of matching
    SS_stats["J_bar"] = (SS_stats["J"]') * grid["s_dist"]

    SS_stats["Profit_L"] = (SS_stats["r_l"] - m[:fix_L] - m[:w_bar])*SS_stats["L"] - m[:ι]*SS_stats["V"]

    # # 2) varphi

    m[:fix] = SS_stats["Y"] - SS_stats_base["mc"]*SS_stats_base["Y"] + SS_stats["Profit_L"] + SS_stats["Profit_K"] - SS_stats_base["Profit"]

    SS_stats["Profit"] = SS_stats["Y"] - SS_stats_base["mc"]*SS_stats_base["Y"] + SS_stats["Profit_L"] + SS_stats["Profit_K"] - m[:fix]

    SS_stats["Profit_int"] = (1-SS_stats["mc"])*SS_stats["Y"] - m[:fix]

    # 3) theta_b

    m[:ω] = (1 - m[:θ_b]*((SS_stats_base["R_a"] - SS_stats_base["R"])*SS_stats_base["leverage"] + SS_stats_base["R"]))/SS_stats_base["leverage"]


    SS_stats["vv"] = ((1-m[:θ_b])*SS_stats["Lambda_b"]*(SS_stats["R_a"] - m[:b_a_aux2]*SS_stats["R"]))/(1 - m[:θ_b]*SS_stats["Lambda_b"]*SS_stats["xx"])
    SS_stats["ee"] = (1-m[:θ_b])*SS_stats["Lambda_b"]*m[:b_a_aux2]*SS_stats["R"]/(1 - m[:θ_b]*SS_stats["Lambda_b"]*SS_stats["zz"])

    # TODO: SS/transition matrix object — handle separately
    # param["DELTA"] = SS_stats["vv"] + SS_stats["ee"]/SS_stats_base["leverage"]

    # TODO: SS/transition matrix object — handle separately
    SS_stats["Profit_FI"] = (1-m[:θ_b])*((SS_stats_base["R_a"] - SS_stats_base["R"])*SS_stats_base["leverage"] + SS_stats_base["R"])*SS_stats_base["NW_b"] - m[:ω]*param_update["q"]*SS_stats_base["A_b"]  # Profits of FI distributed to the bankers family from existing bankers
    # B_FI = (THETA-1)*NWb                                                   # total deposits held at FIs
    #= TODO: SS/transition matrix object — handle separately
    # Bb = param["Bb_ratio"]*B_FI                                             # the family's deposits at FIs
    =#
    # Cb = (MMF_cont + (R-1)*Bb)                                          # the family's consumption level

    m[:fix2] = SS_stats["Profit_FI"]
    # TODO: SS/transition matrix object — handle separately
    SS_stats["AvgC"] = (param_update["delta_ss"]*grid["K"] + m[:w_bar]*grid["L"] + m[:ι]*SS_stats["V"] + m[:fix])/SS_stats["Y"]

    # TODO: SS/transition matrix object — handle separately
    # fix_ratio = m[:fix]/SS_stats["Y"]

    return SS_stats_base, param_base
end
