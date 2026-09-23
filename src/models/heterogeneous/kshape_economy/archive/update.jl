function update(SS_stats, param, param_update, grid)

    SS_stats_base = SS_stats
    param_base = param

    SS_stats_base.delta_ss = param_base.delta_0 * SS_stats_base.v^param.delta_1

    param.delta_1 = param_update.delta_1
    param.iota = param_update.iota
    param.theta_b = param_update.theta_b

    # value of matching
    J_aux2 = inv(Matrix{Float64}(I, grid.ns, grid.ns) - SS_stats_base.Lambda *
                 (1 - param.death_rate) * (1 - param.lambda) * (1 - param.in) * param.P_SS) *
             ones(grid.ns)
    param_base.J_bar2 = transpose(J_aux2) * grid.s_dist

    param.fix_L = (param_base.J_bar1 - param.iota * SS_stats_base.V / SS_stats_base.M) / param_base.J_bar2
    param.delta_0 = SS_stats_base.r_k / (param.delta_1 * SS_stats_base.v^(param.delta_1 - 1))

    # param.delta_0 = SS_stats_base.delta_ss/(SS_stats_base.v^param.delta_1)
    param.delta_ss = param.delta_0 * SS_stats_base.v^param.delta_1

    I_agg = K_tilde_agg + SS_stats.A_g + SS_stats.A_b - (1 - param.delta_0 * SS_stats.v^param.delta_1) * grid.K

    SS_stats.I2 = (1 - param.death_rate) * K_tilde_agg + SS_stats.A_g + SS_stats.A_b -
                  (1 - param.delta_0 * SS_stats.v^param.delta_1) * grid.K

    SS_stats.Profit_L = (SS_stats_base.r_l - param_base.w_bar) * SS_stats.L - param.fix_L - param.iota * SS_stats.V
    SS_stats.Profit_K = (SS_stats_base.r_k * SS_stats_base.v - param.delta_0 * SS_stats_base.v^param.delta_1) * grid.K

    param.fix = SS_stats_base.sale - SS_stats_base.mc * SS_stats_base.Y + SS_stats.Profit_L + SS_stats.Profit_K - SS_stats_base.Profit

    SS_stats.Profit = SS_stats_base.sale - SS_stats_base.mc * SS_stats_base.Y + SS_stats.Profit_L + SS_stats.Profit_K - param.fix

    param.omega = (1 - param.theta_b * ((SS_stats_base.R_a - SS_stats_base.R) * SS_stats_base.leverage + SS_stats_base.R)) / SS_stats_base.leverage

    SS_stats.ee = (1 - param.theta_b) / (1 - param.theta_b * SS_stats_base.zz / SS_stats_base.R)
    SS_stats.vv = ((1 - param.theta_b) * (SS_stats_base.R_a - SS_stats_base.R) / SS_stats_base.R) /
                  (1 - param.theta_b * SS_stats_base.xx / SS_stats_base.R)

    param.DELTA = SS_stats.vv + SS_stats.ee / SS_stats_base.leverage

    # NWb = Ab/THETA; aggregate net worth of financial intermediaries
    SS_stats.Profit_FI = (1 - param.theta_b) * ((SS_stats_base.R_a - SS_stats_base.R) * SS_stats_base.leverage + SS_stats_base.R) *
                         SS_stats_base.NW_b - param.omega * param.q * SS_stats_base.A_b
    # B_FI = (THETA-1)*NWb; total deposits held at FIs
    # Bb   = param.Bb_ratio*B_FI; family's deposits at FIs
    # Cb   = (MMF_cont + (R-1)*Bb); family's consumption

    param.fix2 = SS_stats.Profit_FI

    SS_stats.J = inv(Matrix{Float64}(I, grid.ns, grid.ns) - SS_stats_base.Lambda *
                     (1 - param.death_rate) * (1 - param.lambda) * (1 - param.in) * param.P_SS) *
                 ((SS_stats.r_l - param.w_bar) * param.n * transpose(grid.s) .- param.fix_L)
    SS_stats.J_bar = transpose(SS_stats.J) * grid.s_dist

    diff_mk_check = SS_stats.sale - C_agg - param.tau_cp * SS_stats.A_g - param.frac_b * SS_stats.C_b - I_agg -
                    G_agg - param.iota * SS_stats.V - Int_cost - param.fix - param.fix_L - absorp_annuity +
                    param.death_rate * K_tilde_agg - param.fix2 - (param.R_cb / param.pi_bar - 1) * SS_stats.B_F -
                    SS_stats.r_a * SS_stats.A_F

    return SS_stats_base, param_base, diff_mk_check
end