using Test
using JLD2

include(joinpath(@__DIR__, "../../../src/kshape_CSC/dynamics/index2.jl"))

input = matread(@__DIR__, "../../../../original/inputs/update_Jacob.mat")

param = input["param"]
grid = input["grid"]
SS_stats = input["SS_stats"]
Xss = input["Xss"]
Yss = input ["Yss"]


state_id, control_id = DSGE.build_indices(grid, length(StateSS), length(ControlSS))

F = OrderedDict{Symbol, Any}(
:eq_rate_monetary_policy => 0., 
        :eq_wage => 0., 
        :eq_illiquid_assets => 0., 
        :eq_liquid_assets => 0., 
        :eq_central_bank_assets => 0., 
        :eq_tobins_q => 0., 
        :eq_leverage => 0., 
        :eq_net_worth_bank => 0., 
        :eq_houshold_bond_rate => 0., 
        :eq_quantitative_easing => 0., 
        :eq_inflation_lag => 0., 
        :eq_output_lag => 0., 
        :eq_consumption_lag => 0., 
        :eq_investment_lag => 0., 
        :eq_profit_lag => 0., 
        :eq_unemployment_lag => 0., 
        :eq_government_spending_lag => 0., 
        :eq_lump_sum_transfers => 0., 
        :eq_R_star => 0., #TODO: need to add
        :eq_fiscal_liability => 0., 
        :eq_z => 0., 
        :eq_ψ => 0., 
        :eq_η => 0., 
        :eq_D => 0., 
        :eq_GG => 0., 
        :eq_ι => 0., 
        :eq_BB => 0., 
        :eq_ψ_w => 0., 
        :eq_MP => 0., 
        :eq_pm => 0., 
        :eps_QE_ind => 0., 
        :eps_RP_ind => 0., 
        :eps_B_F_ind => 0., 
        :eps_BB_ind => 0., 
        :eps_Z_ind => 0., 
        :eps_G_ind => 0., 
        :eps_D_ind => 0.,
        :eps_R_ind => 0.,
        :eps_iota_ind => 0.,
        :eps_eta_ind => 0.,
        :eps_w_ind => 0.,
        #CONTROLS
        :MRS_ind => 0.,
        :A_hh_ind => 0.,
        :B_hh_ind => 0.,
        :C_ind => 0.,
        :N_ind => 0.,
        :L_ind => 0.,
        :UB_ind => 0.,
        :eq_control_capital => 0.,
        :eq_control_bond => 0.,
        :eq_control_government_bond => 0.,
        :eq_control_tax => 0.,
:eq_control_transfer => 0.,
        :eq_control_government_spending => 0.,
        :eq_control_lambda => 0.,
        :eq_control_inflation => 0.,
        :eq_control_vacancies => 0.,
        :eq_control_j => 0.,
        :eq_control_mpl => 0.,
        :eq_control_elasticity_labor => 0.,
        :eq_control_output => 0.,
        :eq_control_profit => 0.,
        :eq_control_mpk => 0.,
        :eq_control_mpa => 0.,
        :eq_control_marginal_cost => 0.,
        :eq_control_unemployment_rate => 0.,
        :eq_control_nn => 0.,
        :eq_control_M => 0.,
        :eq_control_f => 0.,
        :eq_control_zz => 0.,
        :eq_control_xx => 0.,
        :eq_control_vv => 0.,
        :eq_control_ee => 0.,
        :eq_control_cb => 0.,
        :eq_control_profit_fi => 0.,
        :eq_control_rra => 0.,
        :eq_control_rr => 0.,
        :eq_control_investment => 0.,
        :eq_control_x_k => 0.,
        :eq_control_a_g_obs => 0.,
        :eq_control_y_obs => 0.,
        :eq_control_c_obs => 0.,
        :eq_control_i_obs => 0.,
        :eq_control_w_obs => 0.,
        :eq_control_profit_obs => 0.,
        :eq_unemployment_observable => 0.,
        :eq_inflation_observable => 0.,
        :eq_rate_observable => 0.,
        :eq_past_wage => 0.,
        :eq_government_observable => 0.,
        :eq_past_YY => 0.,
        :eq_past_CC => 0.,
        :eq_past_II => 0.,
        :eq_past_profit => 0.,
        :eq_past_uu => 0.,
        :eq_past_GG => 0.,
        :eq_past_A_g => 0.,
        :eq_l_lambda => 0.,
        :eq_past_q => 0.,
        :eq_xI => 0.,
        :eq_eta2 => 0.,
        :eq_iota2 => 0.,
        :eq_past_lt2 => 0.,
        :eq_past_g2 => 0.,
        :eq_lt_obs => 0.,
        :eq_b_gov_ncp2 => 0.,
        :eq_rate_monetary_policy_ZLB => 0.,
        :eq_R_star_ZLB => 0.,
        :eq_central_bank_assets_ZLB => 0.,
        :eq_quantitative_easing_ZLB => 0.
    )




                             


