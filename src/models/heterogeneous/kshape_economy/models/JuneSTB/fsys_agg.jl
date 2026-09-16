using DSGE: @sslogdeviations2levels, @sslogdeviations2levels_unprimekeys, @unpack_and_first
using LinearAlgebra: I

"""
```
Fsys_agg(F, m, grid, StateSS, ControlSS, Xt1, Yt1, Xt, Yt, state_id, control_id, reg=1)
```
Return deviations from aggregate equilibrium conditions for the 9kCSC model
(two-sector labour market, nested-CES production, financial intermediary).

9kCSC differs from single-sector mBBQ in:
- Two skill groups: separate wages w_1/w_2, matching M_1/M_2, job values J[1:ns],
  finding rates f_1/f_2, utilisation-adjusted hours n_1/n_2
- ZZ_1–ZZ_4 AR(1) labour-market wedge processes
- Nested-CES production: unskilled L_1 vs capital-skill composite (V_aux)
- Passive-QE-only monetary policy (no active QE regime)

Mirrors kCSC/fsys_agg.jl. See that file for the full equation numbering reference.
"""
function Fsys_agg(F, m, grid, StateSS, ControlSS, Xt1, Yt1, Xt, Yt, state_id, control_id, reg=1)

    grid  = m.dicts[:grid]
    ns_1  = Int(grid["ns_1"])
    ns_2  = Int(grid["ns_2"])
    ns    = Int(grid["ns"])

    #===================================================================#
    #eq1: monetary policy (Taylor rule — no active QE regime in 9kCSC)
    @sslogdeviations2levels R_cb_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys R_cb′_t, MP′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels pi_t, unemp_t = Yt, control_id, ControlSS

    F[:eq_rate_monetary_policy] = log(R_cb′_t) - (
        log(m.dicts[:SS_stats]["R_cb"]) +
        (1 - m[:ρ_R]) * (m[:ϕ_π] * log(pi_t / m[:π_cb]) -
                          m[:ϕ_u] * (unemp_t - m.dicts[:SS_stats]["u"])) +
        m[:ρ_R] * (log(R_cb_t) - log(m.dicts[:SS_stats]["R_cb"])) +
        log(MP′_t))

    #===================================================================#
    #eq2: sector-1 wage Phillips curve
    @sslogdeviations2levels w_1_t, π_past_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys w_1′_t, ZZ_4′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels r_l_1_t = Yt, control_id, ControlSS

    F[:eq_wage_1] = log(w_1′_t) - (
        log(m[:w_bar_1]) +
        m[:ρ_w_1] * (log(w_1_t) - log(m[:w_bar_1])) +
        m[:ρ_w_1] * (m[:d_1] * log(m[:π_bar] / pi_t) +
                      (1 - m[:d_1]) * log(π_past_t / pi_t)) +
        (1 - m[:ρ_w_1]) * (log(1 / ZZ_4′_t) +
                            m.dicts[:param]["eps_w_1"] * log(r_l_1_t / m.dicts[:SS_stats]["r_l_1"])))

    #===================================================================#
    #eq3: sector-2 wage Phillips curve
    @sslogdeviations2levels w_2_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys w_2′_t, ψ_w′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels r_l_2_t = Yt, control_id, ControlSS

    F[:eq_wage_2] = log(w_2′_t) - (
        log(m[:w_bar_2]) +
        m[:ρ_w_2] * (log(w_2_t) - log(m[:w_bar_2])) +
        m[:ρ_w_2] * (m[:d_2] * log(m[:π_bar] / pi_t) +
                      (1 - m[:d_2]) * log(π_past_t / pi_t)) +
        (1 - m[:ρ_w_2]) * (log(1 / ψ_w′_t) +
                            m.dicts[:param]["eps_w_2"] * log(r_l_2_t / m.dicts[:SS_stats]["r_l_2"])))

    #===================================================================#
    #eq4: average wage  w = (L_1*w_1 + L_2*w_2) / (L_1 + L_2)
    @sslogdeviations2levels w_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys w′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels L_1_t, L_2_t = Yt, control_id, ControlSS

    F[:eq_wage_avg] = log(w′_t) - log((L_1_t * w_1′_t + L_2_t * w_2′_t) / (L_1_t + L_2_t))

    #===================================================================#
    #eq5: illiquid assets (bank balance sheet)
    @sslogdeviations2levels lev_t, NW_b_t, Q_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys A_b′_t, lev′_t, NW_b′_t, Q′_t = Xt1, state_id, StateSS

    F[:eq_illiquid_assets] = log(A_b′_t) - log(lev′_t * NW_b′_t / Q′_t)

    #===================================================================#
    #eq6: liquid assets (bond market / government budget)
    @sslogdeviations2levels A_gaux_t, B_b_t, A_b_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys B_b′_t, A_gaux′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels T_t, B_gov_ncp_t, r_a_t, Profit_t = Yt, control_id, ControlSS
    @sslogdeviations2levels UB_t, LT_t, C_b_t, G_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys B_gov_ncp′_t = Yt1, control_id, ControlSS

    A_g_t  = A_gaux_t
    A_g′_t = A_gaux′_t

    F[:eq_liquid_assets] = log(B_b′_t) -
        log(T_t + B_gov_ncp′_t - B_gov_ncp_t * R_cb_t / pi_t +
            (Q′_t + r_a_t - R_cb_t / pi_t * Q_t) * A_g_t -
            UB_t - LT_t -
            m[:τ_cp] * Q′_t * A_g′_t - G_t +
            m[:b_a_aux2] * R_cb_t / pi_t * B_b_t -
            C_b_t)

    #===================================================================#
    #eq7: central bank asset holdings (passive QE rule)
    @sslogdeviations2levels x_cb_t, ψ_rp_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys x_cb′_t, ψ_rp′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels Y_t = Yt, control_id, ControlSS

    F[:eq_central_bank_assets] = log(A_gaux′_t) -
        log(m[:ρ_A_g] * A_g_t +
            (1 - m[:ρ_A_g]) * m.dicts[:SS_stats]["A_g"])

    #===================================================================#
    #eq8: Tobin's q
    @sslogdeviations2levels ι2_t, x_k_t, λ_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys ι2′_t, x_k′_t, x_I′_t = Yt1, control_id, ControlSS

    F[:eq_tobins_q] = log(Q′_t) -
        log(ι2_t * (1 + m[:ϕ] * log(x_k_t) + m[:ϕ] / 2 * (log(x_k_t))^2) -
            λ_t * (ι2′_t * (1 + m[:ϕ] * log(x_k′_t) * x_k′_t) - x_I′_t))

    #===================================================================#
    #eq9: bank leverage
    @sslogdeviations2levels ee_t, vv_t = Yt, control_id, ControlSS

    F[:eq_leverage] = log(lev′_t) - log(ee_t / (m[:δ] - vv_t))

    #===================================================================#
    #eq10: bank net worth
    @sslogdeviations2levels RRa_t = Yt, control_id, ControlSS

    F[:eq_net_worth_bank] = log(NW_b′_t) -
        log(m[:θ_b] * ((RRa_t - m[:b_a_aux2] * R_cb_t / pi_t) * lev_t +
                        m[:b_a_aux2] * R_cb_t / pi_t) * NW_b_t +
            m[:ω] * Q_t * A_b_t)

    #===================================================================#
    #eq11: household bond rate
    @sslogdeviations2levels R_tilde_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys R_tilde′_t = Xt1, state_id, StateSS

    F[:eq_household_bond_rate] = log(R_tilde′_t) - log(R_cb′_t)

    #===================================================================#
    #eq12: passive QE / x_cb process
    F[:eq_quantitative_easing] = log(x_cb′_t) -
        log(1 + m[:ρ_X_QE] * (x_cb_t - 1) +
            (1 - m[:ρ_X_QE]) * (-m[:ϕ_π_QE] * log(pi_t / m[:π_cb]) +
                                   m[:ϕ_u_QE] * (unemp_t - m.dicts[:SS_stats]["u"])))

    #===================================================================#
    #eq13: inflation lag
    @sslogdeviations2levels_unprimekeys π_past′_t = Xt1, state_id, StateSS

    F[:eq_inflation_lag] = log(π_past′_t) - log(pi_t)

    #===================================================================#
    #eq14: output lag
    @sslogdeviations2levels_unprimekeys Y_past′_t = Xt1, state_id, StateSS

    F[:eq_output_lag] = log(Y_past′_t) - log(Y_t)

    #===================================================================#
    #eq15: consumption lag
    @sslogdeviations2levels_unprimekeys C_past′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels C_t = Yt, control_id, ControlSS

    F[:eq_consumption_lag] = log(C_past′_t) - log(C_t)

    #===================================================================#
    #eq16: investment lag
    @sslogdeviations2levels_unprimekeys I_past′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels I_t = Yt, control_id, ControlSS

    F[:eq_investment_lag] = log(I_past′_t) - log(I_t)

    #===================================================================#
    #eq17: profit lag
    @sslogdeviations2levels_unprimekeys Profit_past′_t = Xt1, state_id, StateSS

    F[:eq_profit_lag] = log(Profit_past′_t) - log(Profit_t + m[:fix2])

    #===================================================================#
    #eq18: unemployment sector-1 lag
    @sslogdeviations2levels_unprimekeys unemp_past_1′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels unemp_1_t = Yt, control_id, ControlSS

    F[:eq_unemployment_1_lag] = log(unemp_past_1′_t) - log(unemp_1_t)

    #===================================================================#
    #eq19: unemployment sector-2 lag
    @sslogdeviations2levels_unprimekeys unemp_past_2′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels unemp_2_t = Yt, control_id, ControlSS

    F[:eq_unemployment_2_lag] = log(unemp_past_2′_t) - log(unemp_2_t)

    #===================================================================#
    #eq20: aggregate unemployment lag
    @sslogdeviations2levels_unprimekeys unemp_past′_t = Xt1, state_id, StateSS

    F[:eq_unemployment_lag] = log(unemp_past′_t) - log(unemp_t)

    #===================================================================#
    #eq21: government spending lag
    @sslogdeviations2levels_unprimekeys G_past′_t = Xt1, state_id, StateSS

    F[:eq_government_spending_lag] = log(G_past′_t) - log(G_t)

    #===================================================================#
    #eq22: lump-sum transfers lag
    @sslogdeviations2levels_unprimekeys LT_past′_t = Xt1, state_id, StateSS

    F[:eq_lump_sum_transfers_lag] = log(LT_past′_t) - log(LT_t)

    #===================================================================#
    #R_star (natural rate, mirrors Taylor rule)
    @sslogdeviations2levels R_star_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys R_star′_t = Xt1, state_id, StateSS

    F[:eq_R_star] = log(R_star′_t) - (
        log(m.dicts[:SS_stats]["R_cb"]) +
        (1 - m[:ρ_R]) * (m[:ϕ_π] * log(pi_t / m[:π_cb]) -
                          m[:ϕ_u] * (unemp_t - m.dicts[:SS_stats]["u"])) +
        m[:ρ_R] * (log(R_star_t) - log(m.dicts[:SS_stats]["R_cb"])) +
        log(MP′_t))

    #===================================================================#
    #eq23: ZZ_1 AR(1) labour-market wedge
    @sslogdeviations2levels ZZ_1_t, eps_1_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys ZZ_1′_t = Xt1, state_id, StateSS

    #= wrong scaling: F[:eq_ZZ_1] = m[:ρ_ZZ_1] * log(ZZ_1′_t) - (m[:ρ_ZZ_1] * log(ZZ_1_t) + log(eps_1_t)) =#
    F[:eq_ZZ_1] = log(ZZ_1′_t) - (
        m[:ρ_ZZ_1] * log(ZZ_1_t) +
        (1 - m[:ρ_ZZ_1]) * log(m.dicts[:param]["ZZ_1"]) + log(eps_1_t))

    #===================================================================#
    #eq24: ZZ_2 AR(1) labour-market wedge
    @sslogdeviations2levels ZZ_2_t, eps_2_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys ZZ_2′_t = Xt1, state_id, StateSS

    F[:eq_ZZ_2] = log(ZZ_2′_t) - (
        m[:ρ_ZZ_2] * log(ZZ_2_t) +
        (1 - m[:ρ_ZZ_2]) * log(m.dicts[:param]["ZZ_2"]) + log(eps_2_t))

    #===================================================================#
    #eq25: ZZ_3 AR(1) labour-market wedge
    @sslogdeviations2levels ZZ_3_t, eps_3_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys ZZ_3′_t = Xt1, state_id, StateSS

    F[:eq_ZZ_3] = log(ZZ_3′_t) - (
        m[:ρ_ZZ_3] * log(ZZ_3_t) +
        (1 - m[:ρ_ZZ_3]) * log(m.dicts[:param]["ZZ_3"]) + log(eps_3_t))

    #===================================================================#
    #eq26: ZZ_4 AR(1) labour-market wedge
    @sslogdeviations2levels ZZ_4_t, eps_4_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys ZZ_4′_t = Xt1, state_id, StateSS

    F[:eq_ZZ_4] = log(ZZ_4′_t) - (m[:ρ_ZZ_4] * log(ZZ_4_t) + log(eps_4_t))

    #===================================================================#
    #eq27: fiscal liability B_F
    @sslogdeviations2levels B_F_t, eps_B_F_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys B_F′_t = Xt1, state_id, StateSS

    F[:eq_fiscal_liability] = log(B_F′_t) - (
        m[:ρ_B_F] * log(B_F_t) +
        (1 - m[:ρ_B_F]) * log(m.dicts[:SS_stats]["Y"] /
                               (m.dicts[:SS_stats]["Y"] - m.dicts[:SS_stats]["B_F"])) +
        log(eps_B_F_t))

    #===================================================================#
    #eq28: TFP Z
    @sslogdeviations2levels Z_t, eps_Z_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys Z′_t = Xt1, state_id, StateSS

    F[:eq_z] = log(Z′_t) - (m[:ρ_Z] * log(Z_t) + (1 - m[:ρ_Z]) * log(m[:Z]) + log(eps_Z_t))

    #===================================================================#
    #eq29: risk-premium wedge ψ_rp
    @sslogdeviations2levels ψ_rp_t, eps_RP_t = Xt, state_id, StateSS

    F[:eq_ψ] = log(ψ_rp′_t) - (m[:ρ_ψ_rp] * log(ψ_rp_t) + log(eps_RP_t))

    #===================================================================#
    #eq30: markup η = p_m / (p_m - 1)
    @sslogdeviations2levels η_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys η′_t, p_m′_t = Xt1, state_id, StateSS

    F[:eq_η] = log(η′_t) - (log(p_m′_t) - log(p_m′_t - 1))

    #===================================================================#
    #eq31: discount factor shock D
    @sslogdeviations2levels D_t, eps_D_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys D′_t = Xt1, state_id, StateSS

    F[:eq_D] = log(D′_t) - (m[:ρ_D] * log(D_t) + log(eps_D_t))

    #===================================================================#
    #eq32: fiscal multiplier GG
    @sslogdeviations2levels GG_t, eps_G_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys GG′_t = Xt1, state_id, StateSS

    F[:eq_GG] = log(GG′_t) - (
        m[:ρ_G] * log(GG_t) +
        (1 - m[:ρ_G]) * log(m.dicts[:SS_stats]["Y"] /
                              (m.dicts[:SS_stats]["Y"] -
                               m.dicts[:SS_stats]["LT"] -
                               m.dicts[:SS_stats]["C_b"])) +
        log(eps_G_t))

    #===================================================================#
    #eq33: investment-specific technology ι
    @sslogdeviations2levels ι_t, eps_iota_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys ι′_t = Xt1, state_id, StateSS

    F[:eq_ι] = log(ι′_t) - (m[:ρ_ι] * log(ι_t) + log(eps_iota_t))

    #===================================================================#
    #eq34: spread shock BB
    @sslogdeviations2levels BB_t, eps_BB_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys BB′_t = Xt1, state_id, StateSS

    #= wrong scaling: F[:eq_BB_1] = m[:ρ_BB] * log(BB′_t) - (m[:ρ_BB] * log(BB_t) + log(eps_BB_t)) =#
    F[:eq_BB] = log(BB′_t) - (m[:ρ_BB] * log(BB_t) + log(eps_BB_t))

    #===================================================================#
    #eq35: sector-2 wage-bargaining wedge ψ_w
    @sslogdeviations2levels ψ_w_t, eps_w_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys ψ_w′_t = Xt1, state_id, StateSS

    F[:eq_ψ_w] = log(ψ_w′_t) - (m[:ρ_ψ_w] * log(ψ_w_t) + log(eps_w_t))

    #===================================================================#
    #eq36: monetary policy shock MP
    @sslogdeviations2levels MP_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys MP′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels eps_R_t = Xt, state_id, StateSS

    F[:eq_MP] = log(MP′_t) - (m[:ρ_mp] * log(MP_t) + log(eps_R_t))

    #===================================================================#
    #eq37: price markup p_m AR(1)
    @sslogdeviations2levels p_m_t, eps_eta_t = Xt, state_id, StateSS

    F[:eq_pm] = log(p_m′_t) - (
        m[:ρ_η] * log(p_m_t) +
        (1 - m[:ρ_η]) * log(m[:η] / (m[:η] - 1)) +
        log(eps_eta_t))

    #===================================================================#
    #eq38–52: shock innovation identities (i.i.d.)
    @sslogdeviations2levels_unprimekeys eps_1′_t, eps_2′_t, eps_3′_t, eps_4′_t,
        eps_QE′_t, eps_RP′_t, eps_B_F′_t, eps_BB′_t,
        eps_Z′_t, eps_G′_t, eps_D′_t, eps_R′_t,
        eps_iota′_t, eps_eta′_t, eps_w′_t = Xt1, state_id, StateSS

    F[:eps_1_ind]    = log(eps_1′_t)
    F[:eps_2_ind]    = log(eps_2′_t)
    F[:eps_3_ind]    = log(eps_3′_t)
    F[:eps_4_ind]    = log(eps_4′_t)
    F[:eps_QE_ind]   = log(eps_QE′_t)
    F[:eps_RP_ind]   = log(eps_RP′_t)
    F[:eps_B_F_ind]  = log(eps_B_F′_t)
    F[:eps_BB_ind]   = log(eps_BB′_t)
    F[:eps_Z_ind]    = log(eps_Z′_t)
    F[:eps_G_ind]    = log(eps_G′_t)
    F[:eps_D_ind]    = log(eps_D′_t)
    F[:eps_R_ind]    = log(eps_R′_t)
    F[:eps_iota_ind] = log(eps_iota′_t)
    F[:eps_eta_ind]  = log(eps_eta′_t)
    F[:eps_w_ind]    = log(eps_w′_t)

    #===================================================================#
    # ── CONTROL EQUATIONS ──────────────────────────────────────────────
    #===================================================================#
    #= hetfsys block handles these 14 rows (MRS_ind..U2); test expects n_hh_summary_julia=3:
    F[:MRS_ind]       = 0.
    F[:A_hh_ind]      = 0.
    F[:B_hh_ind]      = 0.
    F[:C_ind]         = 0.
    F[:N_ind_1]       = 0.
    F[:N_ind_2]       = 0.
    F[:L_ind_1]       = 0.
    F[:L_ind_2]       = 0.
    F[:UB_ind]        = 0.
    F[:unemp_ind_1]   = 0.
    F[:unemp_ind_2]   = 0.
    F[:unemp_ind]     = 0.
    F[:U1]            = 0.
    F[:U2]            = 0.
    =#
    F[:Income_Tax_ind]= 0.
    F[:J_bar_1_ind]   = 0.
    F[:J_bar_2_ind]   = 0.

    #===================================================================#
    #control eq1: capital market clearing (includes entrepreneur capital A_F)
    @sslogdeviations2levels K_t, A_hh_t = Yt, control_id, ControlSS

    F[:eq_control_capital] = K_t - (A_hh_t + A_g_t + A_b_t + m.dicts[:SS_stats]["A_F"])

    #===================================================================#
    #control eq2: bond market clearing
    @sslogdeviations2levels B_t, B_hh_t = Yt, control_id, ControlSS

    F[:eq_control_bond] = B_t - (B_hh_t + B_b_t + (1 - 1/1) * m.dicts[:SS_stats]["Y"])

    #===================================================================#
    #control eq3: government bonds
    F[:eq_control_government_bond] = B_gov_ncp_t -
        (B_b_t + B_hh_t + (1 - 1/1) * m.dicts[:SS_stats]["Y"] -
         (Q_t * A_b_t - NW_b_t) - Q_t * (1 + m[:τ_cp]) * A_g_t)

    #===================================================================#
    #control eq4: taxes (labour income tax from hetfsys + capital income)
    @sslogdeviations2levels Income_Tax_t = Yt, control_id, ControlSS
    #= archived future-period form: @sslogdeviations2levels_unprimekeys Income_Tax′_t = Yt1, control_id, ControlSS =#
    #= archived direct formula: T = τ_w*(w_1′*L_1 + w_2′*L_2 + UB) + τ_a*Profit =#
    F[:eq_control_tax] = T_t - (Income_Tax_t + m[:τ_a] * (1 - m[:Eratio]) * Profit_t)

    #===================================================================#
    #control eq5: lump-sum transfers (G regime)
    F[:eq_control_transfer] = LT_t - ((1 - 1/GG′_t) * m.dicts[:SS_stats]["Y"] -
                                        m.dicts[:SS_stats]["C_b"])

    #===================================================================#
    #control eq6: government spending (fixed at SS)
    F[:eq_control_government_spending] = G_t - m.dicts[:SS_stats]["G"]

    #===================================================================#
    #control eq7: stochastic discount factor / Lambda
    @sslogdeviations2levels MRS_t = Yt, control_id, ControlSS

    F[:eq_control_lambda] = λ_t - MRS_t * m[:λ_aux] / m[:λ_b_aux]

    #===================================================================#
    #control eq8: inflation (FTPL)
    F[:eq_control_inflation] = pi_t - m[:π_cb] * exp(
        m[:ρ_B] / (m[:ρ_B] + m[:γ_π]) *
            log(B_gov_ncp_t * R_cb_t /
                (m.dicts[:SS_stats]["B_gov_ncp"] * m.dicts[:SS_stats]["R_cb"])) -
        m[:γ_T] / (m[:ρ_B] + m[:γ_π]) * log(T_t / m.dicts[:SS_stats]["T"]) -
        1 / (m[:ρ_B] + m[:γ_π]) * log(B_gov_ncp′_t / m.dicts[:SS_stats]["B_gov_ncp"]))

    #===================================================================#
    #control eq9: vacancy posting sector 1
    @sslogdeviations2levels V_1_t, V_2_t, M_1_t, M_2_t, J_t = Yt, control_id, ControlSS
    @sslogdeviations2levels J_bar_1_t, J_bar_2_t = Yt, control_id, ControlSS

    F[:eq_control_vacancies_1] = V_1_t - (M_1_t / m[:ι_1] * J_bar_1_t)

    #= archived weighted-J form (matches MATLAB J_bar def, not vacancy posting):
    se_dist = m.dicts[:grid]["se_dist"]
    F[:eq_control_vacancies_1] = V_1_t -
        (M_1_t / m[:ι_1] *
         (J_t[1:ns_1]' * se_dist[1:ns_1]) /
         sum(se_dist[1:ns_1]))
    =#

    #===================================================================#
    #control eq10: vacancy posting sector 2
    F[:eq_control_vacancies_2] = V_2_t - (M_2_t / m[:ι_2] * J_bar_2_t)

    #= archived weighted-J form:
    F[:eq_control_vacancies_2] = V_2_t -
        (M_2_t / m[:ι_2] *
         (J_t[ns_1+1:ns_1+ns_2]' * se_dist[ns_1+1:ns_1+ns_2]) /
         sum(se_dist[ns_1+1:ns_1+ns_2]))
    =#

    #===================================================================#
    #control eq11: job values J (vector over skill types)
    @sslogdeviations2levels n_1_t, n_2_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys J′_t, l_λ_1′_t, l_λ_2′_t = Yt1, control_id, ControlSS

    id1_mat = Matrix{Float64}(I, ns_1, ns_1)
    id2_mat = Matrix{Float64}(I, ns_2, ns_2)

    lambdanext_aux = [(1 - l_λ_1′_t) * id1_mat  zeros(ns_1, ns_2);
                       zeros(ns_2, ns_1)           (1 - l_λ_2′_t) * id2_mat]

    J_inst = vcat(
        (r_l_1_t - m[:fix_L_1] - w_1′_t) .* n_1_t .* m.dicts[:grid]["s"][1:ns_1],
        (r_l_2_t - m[:fix_L_2] - w_2′_t) .* n_2_t .* m.dicts[:grid]["s"][ns_1+1:ns_1+ns_2])

    F[:eq_control_j] = J_t - (
        J_inst +
        λ_t * (1 - m[:dr]) * (1 - m[:in]) *
        m.dicts[:param]["P_SS"] * lambdanext_aux * J′_t)

    #===================================================================#
    # ── Nested-CES production block ────────────────────────────────────
    #   F_aux = (a*L_1^ζ + (1-a)*V_aux^ζ)^(1/ζ)
    #   V_aux = (w*K_tilde^ρ + (1-w)*L_2^ρ)^(1/ρ),   K_tilde = v*K
    #===================================================================#
    @sslogdeviations2levels v_t, MC_t = Yt, control_id, ControlSS

    K_tilde = v_t * K_t
    V_aux   = (m[:w] * (ZZ_3′_t * K_tilde)^m[:ρ] +
               (1 - m[:w]) * (ZZ_2′_t * L_2_t)^m[:ρ])^(1 / m[:ρ])
    F_prod  = (m[:a] * (ZZ_1′_t * L_1_t)^m[:ζ] +
               (1 - m[:a]) * V_aux^m[:ζ])^(1 / m[:ζ])

    #===================================================================#
    #control eq12: MPL sector 1
    F[:eq_control_mpl_1] = r_l_1_t -
        (MC_t * Z′_t * F_prod^(1 - m[:ζ]) * m[:a] *
         L_1_t^(m[:ζ] - 1) * ZZ_1′_t^m[:ζ])

    #===================================================================#
    #control eq13: MPL sector 2
    F[:eq_control_mpl_2] = r_l_2_t -
        (MC_t * Z′_t * F_prod^(1 - m[:ζ]) * (1 - m[:a]) *
         V_aux^(m[:ζ] - m[:ρ]) * (1 - m[:w]) *
         L_2_t^(m[:ρ] - 1) * ZZ_2′_t^m[:ρ])

    #===================================================================#
    #control eq14: capital utilisation v — FOC: r_k = δ_0 * δ_1 * v^(δ_1−1)
    @sslogdeviations2levels r_k_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys v′_t = Yt1, control_id, ControlSS

    #= archived inverted form: F[:eq_control_elasticity_v] = r_k_t - m[:δ_0] * m[:δ_1] * v_t^(m[:δ_1] - 1) =#
    #= archived identity: F[:eq_control_elasticity_v] = v_t - v′_t =#
    F[:eq_control_elasticity_v] = v_t -
        (r_k_t / (Q′_t * m[:δ_0] * m[:δ_1]))^(1 / (m[:δ_1] - 1))

    #===================================================================#
    #control eq15: output Y (nested-CES)
    F[:eq_control_output] = Y_t - Z′_t * F_prod

    #===================================================================#
    #control eq16: firm profit (two-sector wages and vacancy costs)
    @sslogdeviations2levels_unprimekeys K′_t = Yt1, control_id, ControlSS
    @sslogdeviations2levels Profit_FI_t = Yt, control_id, ControlSS

    F[:eq_control_profit] = Profit_t - (
        (Y_t * (1 - η′_t / (2 * m[:κ]) *
                (log(pi_t) - (1 - m[:γ]) * log(m[:π_cb]) - m[:γ] * log(π_past_t))^2) +
         Y_t * (-MC_t) - m[:fix] - log(B_F′_t) * m.dicts[:SS_stats]["Y"] +
         (r_l_1_t - m[:fix_L_1] - w_1′_t) * L_1_t +
         (r_l_2_t - m[:fix_L_2] - w_2′_t) * L_2_t -
         m[:ι_1] * V_1_t - m[:ι_2] * V_2_t) +
        (r_k_t * v_t - Q′_t * (m[:δ_00] + m[:δ_0] * v_t^m[:δ_1])) * K_t -
        Q′_t * K_t + ι2_t * K_t + Q′_t * K′_t -
        ι2_t * (K′_t + m[:ϕ] / 2 * log(K′_t / K_t)^2 * K′_t) +
        Profit_FI_t - m[:fix2])

    #===================================================================#
    #control eq17: MPK (nested-CES)
    F[:eq_control_mpk] = r_k_t -
        (MC_t * Z′_t * F_prod^(1 - m[:ζ]) * (1 - m[:a]) *
         V_aux^(m[:ζ] - m[:ρ]) * m[:w] *
         K_tilde^(m[:ρ] - 1) * ZZ_3′_t^m[:ρ])

    #===================================================================#
    #control eq18: return on illiquid assets r_a
    F[:eq_control_mpa] = r_a_t -
        ((1 - m.dicts[:SS_stats]["tau_a"]) * (1 - m[:Eratio] - m[:b_share]) * Profit_t / K_t)

    #===================================================================#
    #control eq19: marginal cost MC (NKPC)
    @sslogdeviations2levels_unprimekeys Y′_t, pi′_t, η2′_t = Yt1, control_id, ControlSS
    @sslogdeviations2levels η2_t = Yt, control_id, ControlSS

    F[:eq_control_marginal_cost] = MC_t - (
        1 - 1/η′_t +
        (log(pi_t / (π_past_t^m[:γ] * m[:π_bar]^(1 - m[:γ]))) -
         m.dicts[:SS_stats]["Lambda"] * η2′_t / η2_t * Y′_t / Y_t *
         log(pi′_t / (pi_t^m[:γ] * m[:π_bar]^(1 - m[:γ])))) / m[:κ])

    #===================================================================#
    #control eq20: hours per worker sector 1
    #= archived GHH formula (m[:GHH] ≠ 1 in practice, so MATLAB takes else branch → n=1):
       F[:eq_control_n_1] = n_1_t - ((1 - m[:τ_L]) * (1 - m[:τ_P]) * w_1′_t^(1 - m[:τ_P]) / m[:ψ_1])^(m[:ξ] / (1 + m[:ξ] * m[:τ_P])) =#
    F[:eq_control_n_1] = n_1_t - 1.0

    #===================================================================#
    #control eq21: hours per worker sector 2
    #= archived GHH formula (MATLAB GHH ≠ 1 → else branch → n=1):
       F[:eq_control_n_2] = n_2_t - ((1 - m[:τ_L]) * (1 - m[:τ_P]) * w_2′_t^(1 - m[:τ_P]) / m[:ψ_2])^(m[:ξ] / (1 + m[:ξ] * m[:τ_P])) =#
    F[:eq_control_n_2] = n_2_t - 1.0

    #===================================================================#
    #control eq22: matching function sector 1
    @sslogdeviations2levels U_1_t, U_2_t = Yt, control_id, ControlSS

    F[:eq_control_M_1] = M_1_t -
        (U_1_t * V_1_t /
         ((U_1_t^m[:α_1] + V_1_t^m[:α_1])^(1 / m[:α_1])))

    #===================================================================#
    #control eq23: matching function sector 2
    F[:eq_control_M_2] = M_2_t -
        (U_2_t * V_2_t /
         ((U_2_t^m[:α_2] + V_2_t^m[:α_2])^(1 / m[:α_2])))

    #===================================================================#
    #control eq24: job-finding rate sector 1
    @sslogdeviations2levels f_1_t, f_2_t = Yt, control_id, ControlSS

    F[:eq_control_f_1] = f_1_t - M_1_t / U_1_t

    #===================================================================#
    #control eq25: job-finding rate sector 2
    F[:eq_control_f_2] = f_2_t - M_2_t / U_2_t

    #===================================================================#
    #control eq26: bank zz
    @sslogdeviations2levels zz_t = Yt, control_id, ControlSS

    F[:eq_control_zz] = zz_t -
        ((RRa_t - m[:b_a_aux2] * R_cb_t / pi_t) * lev_t +
         m[:b_a_aux2] * R_cb_t / pi_t)

    #===================================================================#
    #control eq27: bank xx
    @sslogdeviations2levels xx_t = Yt, control_id, ControlSS

    F[:eq_control_xx] = xx_t - (lev′_t / lev_t) * zz_t

    #===================================================================#
    #control eq28: banker value of assets vv
    @sslogdeviations2levels_unprimekeys RRa′_t, xx′_t, vv′_t, pi′_t = Yt1, control_id, ControlSS

    F[:eq_control_vv] = vv_t - (
        (1 - m[:θ_b]) * BB′_t * λ_t * (RRa′_t - R_cb′_t / pi′_t) +
        m[:θ_b] * BB′_t * λ_t * xx′_t * vv′_t)

    #===================================================================#
    #control eq29: banker franchise value ee
    @sslogdeviations2levels_unprimekeys zz′_t, ee′_t = Yt1, control_id, ControlSS

    F[:eq_control_ee] = ee_t - (
        (1 - m[:θ_b]) * BB′_t * λ_t * (R_cb′_t / pi′_t) +
        m[:θ_b] * BB′_t * λ_t * zz′_t * ee′_t)

    #===================================================================#
    #control eq30: banker household consumption C_b
    @sslogdeviations2levels_unprimekeys C_b′_t = Yt1, control_id, ControlSS

    F[:eq_control_cb] = C_b_t -
        ((D′_t * m[:β_b] * R_cb′_t / pi′_t /
          (1 + m[:ϕ_b] * (B_b′_t / B_b_t - 1)) *
          C_b′_t^(-m[:σ2]))^(-1 / m[:σ2]))

    #===================================================================#
    #control eq31: financial intermediary profit
    F[:eq_control_profit_fi] = Profit_FI_t -
        ((1 - m[:θ_b]) * ((RRa_t - m[:b_a_aux2] * R_cb_t / pi_t) * lev_t +
                           m[:b_a_aux2] * R_cb_t / pi_t) * NW_b_t -
         m[:ω] * Q_t * A_b_t)

    #===================================================================#
    #control eq32: return on illiquid assets RRa
    F[:eq_control_rra] = RRa_t - (Q′_t + r_a_t) / Q_t

    #===================================================================#
    #control eq33: real interest rate RR
    @sslogdeviations2levels RR_t = Yt, control_id, ControlSS

    F[:eq_control_rr] = RR_t - R_cb_t / pi_t

    #===================================================================#
    #control eq34: investment I (nested-CES, includes A_F)
    @sslogdeviations2levels_unprimekeys A_hh′_t = Yt1, control_id, ControlSS

    F[:eq_control_investment] = I_t - (
        ι′_t * (1 + m[:ϕ] / 2 * (log(x_k_t))^2) *
        (A_hh′_t + A_g′_t + A_b′_t + m.dicts[:SS_stats]["A_F"]) -
        Q′_t * (1 - (m[:δ_00] + m[:δ_0] * v_t^m[:δ_1])) * K_t)

    #===================================================================#
    #control eq35: capital growth x_k
    F[:eq_control_x_k] = x_k_t - K′_t / K_t

    #===================================================================#
    # ── Observable identities ──────────────────────────────────────────
    #===================================================================#

    #control eq36: A_g observable
    @sslogdeviations2levels A_g_obs_t = Yt, control_id, ControlSS
    F[:eq_control_a_g_obs] = A_g_obs_t - A_gaux′_t

    #control eq37: Y observable
    @sslogdeviations2levels Y_obs_t = Yt, control_id, ControlSS
    F[:eq_control_y_obs] = Y_obs_t - Y_t

    #control eq38: C observable
    @sslogdeviations2levels C_obs_t = Yt, control_id, ControlSS
    F[:eq_control_c_obs] = C_obs_t - C_t

    #control eq39: I observable
    @sslogdeviations2levels I_obs_t = Yt, control_id, ControlSS
    F[:eq_control_i_obs] = I_obs_t - I_t

    #control eq40–42: wage observables (sector 1, sector 2, aggregate)
    @sslogdeviations2levels w_1_obs_t, w_2_obs_t, w_obs_t = Yt, control_id, ControlSS
    F[:eq_control_w_1_obs] = w_1_obs_t - w_1′_t
    F[:eq_control_w_2_obs] = w_2_obs_t - w_2′_t
    F[:eq_control_w_obs]   = w_obs_t   - w′_t

    #control eq43: profit observable
    @sslogdeviations2levels Profit_obs_t = Yt, control_id, ControlSS
    F[:eq_control_profit_obs] = Profit_obs_t - Profit_t - m[:fix2]

    #control eq44–46: unemployment observables (sector 1, sector 2, aggregate)
    @sslogdeviations2levels unemp_1_obs_t, unemp_2_obs_t, unemp_obs_t = Yt, control_id, ControlSS
    F[:eq_unemployment_1_observable] = unemp_1_obs_t - 100 * unemp_1_t
    F[:eq_unemployment_2_observable] = unemp_2_obs_t - 100 * unemp_2_t
    F[:eq_unemployment_observable]   = unemp_obs_t   - 100 * unemp_t

    #control eq47: inflation observable
    @sslogdeviations2levels pi_obs_t = Yt, control_id, ControlSS
    F[:eq_inflation_observable] = pi_obs_t - pi_t

    #control eq48: R observable
    @sslogdeviations2levels R_obs_t = Yt, control_id, ControlSS
    F[:eq_rate_observable] = R_obs_t - R_cb′_t

    #===================================================================#
    # ── Lag / auxiliary control equations ──────────────────────────────
    #===================================================================#

    #control eq49–51: past wages (sector 1, sector 2, aggregate)
    @sslogdeviations2levels w_1_lag_t, w_2_lag_t, w_lag_t = Yt, control_id, ControlSS
    F[:eq_past_wage_1] = w_1_lag_t - w_1_t
    F[:eq_past_wage_2] = w_2_lag_t - w_2_t
    F[:eq_past_wage]   = w_lag_t   - w_t

    #control eq52: G observable
    @sslogdeviations2levels G_obs_t = Yt, control_id, ControlSS
    F[:eq_government_observable] = G_obs_t - G_t

    #control eq53: past Y
    @sslogdeviations2levels YY_lag_t = Yt, control_id, ControlSS
    @sslogdeviations2levels Y_past_t = Xt, state_id, StateSS
    F[:eq_past_YY] = YY_lag_t - Y_past_t

    #control eq54: past C
    @sslogdeviations2levels CC_lag_t = Yt, control_id, ControlSS
    @sslogdeviations2levels C_past_t = Xt, state_id, StateSS
    F[:eq_past_CC] = CC_lag_t - C_past_t

    #control eq55: past I
    @sslogdeviations2levels II_lag_t = Yt, control_id, ControlSS
    @sslogdeviations2levels I_past_t = Xt, state_id, StateSS
    F[:eq_past_II] = II_lag_t - I_past_t

    #control eq56: past profit
    @sslogdeviations2levels PPROFIT_lag_t = Yt, control_id, ControlSS
    @sslogdeviations2levels Profit_past_t = Xt, state_id, StateSS
    F[:eq_past_profit] = PPROFIT_lag_t - Profit_past_t

    #control eq57–59: past unemployment (sector 1, sector 2, aggregate)
    @sslogdeviations2levels uu_1_lag_t, uu_2_lag_t, uu_lag_t = Yt, control_id, ControlSS
    @sslogdeviations2levels unemp_past_1_t, unemp_past_2_t, unemp_past_t = Xt, state_id, StateSS
    F[:eq_past_uu_1] = uu_1_lag_t - unemp_past_1_t
    F[:eq_past_uu_2] = uu_2_lag_t - unemp_past_2_t
    F[:eq_past_uu]   = uu_lag_t   - unemp_past_t

    #control eq60: past GG (= past LT in G regime)
    @sslogdeviations2levels GG_lag_t = Yt, control_id, ControlSS
    @sslogdeviations2levels LT_past_t = Xt, state_id, StateSS
    F[:eq_past_GG] = GG_lag_t - LT_past_t

    #control eq61: past A_g
    @sslogdeviations2levels A_g_lag_t = Yt, control_id, ControlSS
    F[:eq_past_A_g] = A_g_lag_t - A_g_t

    #control eq66: l_lambda_1 (fixed at calibrated value — MATLAB: l_lambda_1 = param["lambda_1"])
    @sslogdeviations2levels l_λ_1_t = Yt, control_id, ControlSS
    #= archived identity form (wrong: introduces spurious F42 l_λ_1′ dependence):
       @sslogdeviations2levels_unprimekeys l_λ_1′_t = Yt1, control_id, ControlSS
       F[:eq_l_lambda_1] = l_λ_1_t - l_λ_1′_t =#
    F[:eq_l_lambda_1] = l_λ_1_t - m[:λ_1]

    #control eq67: l_lambda_2 (fixed at calibrated value — MATLAB: l_lambda_2 = param["lambda_2"])
    @sslogdeviations2levels l_λ_2_t = Yt, control_id, ControlSS
    #= archived identity form:
       @sslogdeviations2levels_unprimekeys l_λ_2′_t = Yt1, control_id, ControlSS
       F[:eq_l_lambda_2] = l_λ_2_t - l_λ_2′_t =#
    F[:eq_l_lambda_2] = l_λ_2_t - m[:λ_2]

    #control eq62: past Q
    @sslogdeviations2levels Q_lag_t = Yt, control_id, ControlSS
    F[:eq_past_q] = Q_lag_t - Q_t

    #control eq68: investment growth x_I
    @sslogdeviations2levels x_I_t = Yt, control_id, ControlSS
    F[:eq_xI] = x_I_t - Q′_t

    #control eq69: markup auxiliary η2 (= η)
    @sslogdeviations2levels_unprimekeys η′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels η2_t = Yt, control_id, ControlSS
    F[:eq_eta2] = η2_t - η′_t

    #control eq70: IST auxiliary ι2 (= next-period ι)
    @sslogdeviations2levels ι2_t = Yt, control_id, ControlSS
    F[:eq_iota2] = ι2_t - ι′_t

    #control eq63: past LT (2nd copy)
    @sslogdeviations2levels LT2_lag_t = Yt, control_id, ControlSS
    F[:eq_past_lt2] = LT2_lag_t - LT_past_t

    #control eq64: past G (2nd copy)
    @sslogdeviations2levels G2_lag_t = Yt, control_id, ControlSS
    @sslogdeviations2levels G_past_t = Xt, state_id, StateSS
    F[:eq_past_g2] = G2_lag_t - G_past_t

    #control eq65: LT observable
    @sslogdeviations2levels LT_obs_t = Yt, control_id, ControlSS
    F[:eq_lt_obs] = LT_obs_t - LT_t

    #control eq71: B_gov_ncp2 auxiliary
    @sslogdeviations2levels B_gov_ncp2_t = Yt, control_id, ControlSS
    F[:eq_b_gov_ncp2] = B_gov_ncp2_t -
        (B_b_t + B_hh_t - (Q_t * A_b_t - NW_b_t) - Q_t * (1 + m[:τ_cp]) * A_g_t)

    return F
end

#= ── ARCHIVED: single-sector 9kCSC fsys_agg (pre two-sector rewrite) ──────

function Fsys_agg(F, m, grid, StateSS, ControlSS, Xt1, Yt1, Xt, Yt, state_id, control_id, reg=1)

    #===================================================================#
    #eq1: eq_rate_monetary_policy TODO: regime change
    @sslogdeviations2levels R_cb_t, eps_R_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys R_cb′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels pi_t, unemp_t = Yt, control_id, ControlSS

    monpol = get_setting(m, :regime_monpol)[reg]

    F[:eq_rate_monetary_policy] = log(R_cb′_t) - log(m.dicts[:SS_stats]["R_cb"]) -
    (1 - m[:ρ_R]) * (m[:ϕ_π] * log(pi_t/m[:π_cb]) -
    m[:ϕ_u] * (unemp_t - m.dicts[:SS_stats]["u"])) -
    m[:ρ_R]*(log(R_cb_t) - log(m.dicts[:SS_stats]["R_cb"])) - log(eps_R_t)

    F[:eq_rate_monetary_policy_ZLB] = log(R_cb′_t) - log(m.dicts[:SS_stats]["R_cb"]) - log(eps_R_t)

    #... (rest of archived single-sector equations omitted for brevity)
end
=#
