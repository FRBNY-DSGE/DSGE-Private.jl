using Revise
using LinearAlgebra
using OrderedCollections: OrderedDict
using ForwardDiff
using DataFrames
using DSGE



function jacobian!(m::mBBQ)
    grid = m.dicts[:grid] 
    SS_stats = m.dicts[:SS_stats] 
    param = m.dicts[:param] 
    StateSS = m.grids[:StateSS]
    ControlSS = m.grids[:ControlSS] 
    getgrid(g, k::Symbol) =
        haskey(g, k) ? g[k] :
        haskey(g, String(k)) ? g[String(k)] :
        error("Missing grid key: $(k)")


    #one time setup to move prev donggyu format to bbl format
    state_id, control_id = DSGE.build_indices(grid, length(StateSS), length(ControlSS))
    #ss = DSGE.build_ss(param, SS_stats)

    #manual fixes, jank but ygdwygd
    state_id[:A_g_t] = 17733
    push!(StateSS, 0.)
    StateSS[state_id[:A_g_t]] = log(exp(StateSS[state_id[:A_gaux_t]]) - 1)

    ControlSS[control_id[:J_t]] = ControlSS[control_id[:J_t]] .+ log(1.711583655983251)


    #test state and control controls of zero
    State_zero = zeros(length(StateSS))
    Control_zero = zeros(length(ControlSS))

    nState  = length(StateSS)
    nCtrl   = length(ControlSS)


    #F = Dict{Symbol,Any}() #TODO: offload to different file
    #must be same ordering as in idx of donggyu code
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

    zlb_equations = Set([
        (:eq_rate_monetary_policy, :eq_rate_monetary_policy_ZLB),
        (:eq_R_star, :eq_R_star_ZLB),
        (:eq_central_bank_assets, :eq_central_bank_assets_ZLB),
        (:eq_quantitative_easing, :eq_quantitative_easing_ZLB)
    ])

    #set regime for testing
    reg = 1
    F = DSGE.Fsys_agg(F, m, grid, StateSS, ControlSS, State_zero, Control_zero, State_zero, Control_zero, state_id, control_id, reg)

    eq_keys = collect(keys(F))

    # flatten f dict into a vector of scalar, issue with J, might cause indexing issue later on
    function flatten_F!(F_vec, F_dict, eq_keys)
        idx = 1
        for k in eq_keys
            v = get(F_dict, k, zero(eltype(F_vec)))
            if v isa AbstractArray
                n = length(v)
                F_vec[idx:idx+n-1] .= v
                idx += n
            else
                F_vec[idx] = v
                idx += 1
            end
        end
    end

    #determine flat output size
    n_eqs_flat = sum(k -> (F[k] isa AbstractArray ? length(F[k]) : 1), eq_keys)

    # x = [Xt1; Yt1; Xt; Yt] Xt1 is state_t, Yt1 is control_t+1, Xt is state_t-1, Yt is control_t
    n_x = 2*nState + 2*nCtrl
    BA_agg = zeros(n_eqs_flat, n_x)

    function obj_fnct_agg(F_vec, x)
        Xt1 = x[1:nState]
        Yt1 = x[nState+1:nState+nCtrl]
        Xt  = x[nState+nCtrl+1:2*nState+nCtrl]
        Yt  = x[2*nState+nCtrl+1:end]
        F_dict = OrderedDict{Symbol, Any}()
        DSGE.Fsys_agg(F_dict, m, grid, StateSS, ControlSS, Xt1, Yt1, Xt, Yt, state_id, control_id, reg)
        flatten_F!(F_vec, F_dict, eq_keys)
    end

    #autodiff jacobian
    ForwardDiff.jacobian!(BA_agg, obj_fnct_agg, zeros(n_eqs_flat), zeros(n_x))

    # aggregate-only columns (drop distribution block indices)
    dist_state_keys = Set([:marginal_pdf_b_t, :marginal_pdf_a_t, :marginal_pdf_se_t, :copula_t])
    dist_ctrl_keys  = Set([:Value_t, :mutil_c_t, :Va_t])

    agg_state_cols = vcat([collect(state_id[s])   for s in keys(state_id)   if s ∉ dist_state_keys]...)
    agg_ctrl_cols  = vcat([collect(control_id[s]) for s in keys(control_id) if s ∉ dist_ctrl_keys]...)



    #zlb equations 
    num_zlb_equations = length(zlb_equations)

    zlb_nonzlb_idx_pairs = [
        (
            findfirst(x -> x == nonzlb, eq_keys),
            findfirst(x -> x == zlb, eq_keys)
        )
        for (nonzlb, zlb) in zlb_equations
    ]

    #matching donggyu dims
    F1_ad = BA_agg[:, 1:nState][:, agg_state_cols]                        # ∂F/∂Xt1 (agg cols)
    F2_ad = BA_agg[:, nState+1:nState+nCtrl][:, agg_ctrl_cols]            # ∂F/∂Yt1 (agg cols)
    F3_ad = BA_agg[:, nState+nCtrl+1:2*nState+nCtrl][:, agg_state_cols]  # ∂F/∂Xt  (agg cols)
    F4_ad = BA_agg[:, 2*nState+nCtrl+1:end][:, agg_ctrl_cols]            # ∂F/∂Yt  (agg cols)

    os = Int(getgrid(grid, :os))
    oc = Int(getgrid(grid, :oc))
    F21_ad = F1_ad[1:os, :]
    F22_ad = F2_ad[1:os, :]
    F23_ad = F3_ad[1:os, :]
    F24_ad = F4_ad[1:os, :]

    F41_ad = F1_ad[os+1:end-num_zlb_equations, :]
    F42_ad = F2_ad[os+1:end-num_zlb_equations, :]
    F43_ad = F3_ad[os+1:end-num_zlb_equations, :]
    F44_ad = F4_ad[os+1:end-num_zlb_equations, :]

    # FZLB_ad = F1_ad[end-num_zlb_equations+1 : end, :]
    # FZLB_ad = F2_ad[end-num_zlb_equations+1 : end, :]
    # FZLB_ad = F3_ad[end-num_zlb_equations+1 : end, :]
    # FZLB_ad = F4_ad[end-num_zlb_equations+1 : end, :]



    #edit remove col 42
    let a_gaux_col = findfirst(==(state_id[:A_gaux_t]), agg_state_cols)
        replace_pairs = [(F23_ad, 5),
                        (F41_ad, 41), (F41_ad, 43),
                        (F43_ad, 8), (F43_ad, 10), (F43_ad, 60), (F43_ad, 69)]
        for (mat, i) in replace_pairs
            mat[i, a_gaux_col] = mat[i, end]
        end
    end
    F21_ad = F21_ad[:, 1:end-1]
    F23_ad = F23_ad[:, 1:end-1]
    F41_ad = F41_ad[:, 1:end-1]
    F43_ad = F43_ad[:, 1:end-1]



    F21_ad_zlb = copy(F21_ad)
    F22_ad_zlb = copy(F22_ad)
    F23_ad_zlb = copy(F23_ad)
    F24_ad_zlb = copy(F24_ad)

    # trim ag_t
    F1_ad_trim = F1_ad[:, 1:end-1]
    F3_ad_trim = F3_ad[:, 1:end-1]

    for (nonzlb_row, zlb_row) in zlb_nonzlb_idx_pairs
        F21_ad_zlb[nonzlb_row, :] = F1_ad_trim[zlb_row + 4, :] #TODO instead of maual +4 account for J better
        F22_ad_zlb[nonzlb_row, :] = F2_ad[zlb_row + 4, :]
        F23_ad_zlb[nonzlb_row, :] = F3_ad_trim[zlb_row + 4, :]
        F24_ad_zlb[nonzlb_row, :] = F4_ad[zlb_row + 4, :]
    end


    return F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad, F21_ad_zlb, F22_ad_zlb, F23_ad_zlb, F24_ad_zlb, F41_ad, F42_ad, F43_ad, F44_ad

end







