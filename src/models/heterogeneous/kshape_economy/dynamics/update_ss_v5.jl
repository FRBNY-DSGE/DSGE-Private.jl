using LinearAlgebra

"""
    update_ss_v5!(m)

Updates steady-state statistics

This function updates various steady-state calculations including:
- Labor market  (iota, fix_L)
- Job matching values (J, J_bar)
- Profit calculations (Profit_L, Profit, Profit_int, Profit_FI)
- Financial intermediary (theta_b, omega, DELTA)
- Fixed costs and other auxiliary 

"""
function update_ss_v5!(m::mBBQ)


    # 1) iota

    m[:fix_L] = (m.dicts[:SS_stats]["J_bar1"] - m[:ι]*m.dicts[:SS_stats]["V"]/m.dicts[:SS_stats]["M"])/m.dicts[:SS_stats]["J_bar2"]

    m.dicts[:SS_stats]["J"] = inv(I - m[:β_L]*(1-m[:dr])*(1-m.dicts[:param]["lambda"])*
                        (1-m[:in])*m.dicts[:param]["P_SS"]) * ((m.dicts[:SS_stats]["r_l"] - m[:fix_L] - m.dicts[:param]["w_bar"]) 
                                                          .* m.dicts[:param]["n"] .* m.dicts[:grid]["s"])  # value of matching
    m.dicts[:SS_stats]["J_bar"] = ((m.dicts[:SS_stats]["J"]') * m.dicts[:grid]["s_dist"])[1]

    m.dicts[:SS_stats]["Profit_L"] = (m.dicts[:SS_stats]["r_l"] - m[:fix_L] - m.dicts[:param]["w_bar"])*m.dicts[:SS_stats]["L"] - m[:ι]*m.dicts[:SS_stats]["V"]

    # # 2) varphi

    m[:fix] = m.dicts[:SS_stats]["Y"] - m.dicts[:SS_stats]["mc"]*m.dicts[:SS_stats]["Y"] + m.dicts[:SS_stats]["Profit_L"] + m.dicts[:SS_stats]["Profit_K"] - m.dicts[:SS_stats]["Profit"]

    m.dicts[:SS_stats]["Profit"] = m.dicts[:SS_stats]["Y"] - m.dicts[:SS_stats]["mc"]*m.dicts[:SS_stats]["Y"] + m.dicts[:SS_stats]["Profit_L"] + m.dicts[:SS_stats]["Profit_K"] - m[:fix]

    m.dicts[:SS_stats]["Profit_int"] = (1-m.dicts[:SS_stats]["mc"])*m.dicts[:SS_stats]["Y"] - m[:fix]

    # 3) theta_b

    m[:ω] = (1 - m[:θ_b]*((m.dicts[:SS_stats]["R_a"] - m.dicts[:SS_stats]["R"])*m.dicts[:SS_stats]["leverage"] + m.dicts[:SS_stats]["R"]))/m.dicts[:SS_stats]["leverage"]


    m.dicts[:SS_stats]["vv"] = ((1- m[:θ_b])*m.dicts[:SS_stats]["Lambda_b"]*(m.dicts[:SS_stats]["R_a"] - m[:b_a_aux2]*m.dicts[:SS_stats]["R"]))/(1 -  m[:θ_b]*m.dicts[:SS_stats]["Lambda_b"]*m.dicts[:SS_stats]["xx"])
    m.dicts[:SS_stats]["ee"] = (1- m[:θ_b])*m.dicts[:SS_stats]["Lambda_b"]*m[:b_a_aux2]*m.dicts[:SS_stats]["R"]/(1 -  m[:θ_b]*m.dicts[:SS_stats]["Lambda_b"]*m.dicts[:SS_stats]["zz"])


    m[:δ] = m.dicts[:SS_stats]["vv"] + m.dicts[:SS_stats]["ee"]/m.dicts[:SS_stats]["leverage"]

    m.dicts[:SS_stats]["Profit_FI"] = (1-m[:θ_b])*((m.dicts[:SS_stats]["R_a"] - m.dicts[:SS_stats]["R"])*m.dicts[:SS_stats]["leverage"] + m.dicts[:SS_stats]["R"])*m.dicts[:SS_stats]["NW_b"] - m[:ω]*m.dicts[:param]["q"]*m.dicts[:SS_stats]["A_b"]  # Profits of FI distributed to the bankers family from existing bankers
    # B_FI = (THETA-1)*NWb                                                   # total deposits held at FIs
    # Bb = m.dicts[:param]["Bb_ratio"]*B_FI                                             # the family's deposits at FIs
    # Cb = (MMF_cont + (R-1)*Bb)                                          # the family's consumption level

    m[:fix2] = m.dicts[:SS_stats]["Profit_FI"]
    m.dicts[:SS_stats]["AvgC"] = (m.dicts[:param]["delta_ss"]*m.dicts[:grid]["K"] + m.dicts[:param]["w_bar"]*m.dicts[:grid]["L"] + m[:ι]*m.dicts[:SS_stats]["V"] + m[:fix])/m.dicts[:SS_stats]["Y"]

    m.dicts[:param]["fix_ratio"] = m[:fix]/m.dicts[:SS_stats]["Y"]

    return m
end
