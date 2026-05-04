using Revise
using LinearAlgebra
using OrderedCollections: OrderedDict
using ForwardDiff
using DataFrames
using DSGE

include("helpers/index.jl")
include("fsys_agg.jl")

function jacobian!(m::kCSC)
    # t0 = time()

    grid = m.dicts[:grid] 
    SS_stats = m.dicts[:SS_stats] 
    param = m.dicts[:param] 
    StateSS = m.grids[:StateSS]
    ControlSS = m.grids[:ControlSS] 
    getgrid(g, k::Symbol) =
        haskey(g, k) ? g[k] :
        haskey(g, String(k)) ? g[String(k)] :
        error("Missing grid key: $(k)")
    
    # t1 = time()
    # println("setup: $(t1 - t0) seconds")

    #one time setup to move prev donggyu format to bbl format
    state_id, control_id = build_indices_csc(grid, length(StateSS), length(ControlSS))
    
    #ss = DSGE.build_ss(param, SS_stats)

    # t1 = time()
    # println("buildindices: $(t1 - t0) seconds")

    #manual fixes, jank but ygdwygd
    # state_id[:A_g_t] = 17733
    # push!(StateSS, 0.)
    # StateSS[state_id[:A_g_t]] = log(exp(StateSS[state_id[:A_gaux_t]]) - 1)

    # ControlSS[control_id[:J_t]] = ControlSS[control_id[:J_t]] .+ log(1.711583655983251)


    #test state and control controls of zero
    State_zero = zeros(length(StateSS))
    Control_zero = zeros(length(ControlSS))

    nState  = length(StateSS)
    nCtrl   = length(ControlSS)

    # t1 = time()
    # println("setup2: $(t1 - t0) seconds")

    #F = Dict{Symbol,Any}() #TODO: offload to different file
    #must be same ordering as in idx of donggyu code
    # F = OrderedDict{Symbol, Any}(
    #     #states
    #     :eq_rate_monetary_policy => 0.,
    #     :eq_wage_1 => 0.,
    #     :eq_wage_2 => 0.,
    #     :eq_wage_avg => 0.,
    #     :eq_illiquid_assets => 0.,
    #     :eq_liquid_assets => 0.,
    #     :eq_central_bank_assets => 0.,
    #     :eq_tobins_q => 0.,
    #     :eq_leverage => 0.,
    #     :eq_net_worth_bank => 0.,
    #     :eq_houshold_bond_rate => 0.,
    #     :eq_quantitative_easing => 0.,
    #     :eq_inflation_lag => 0.,
    #     :eq_output_lag => 0.,
    #     :eq_consumption_lag => 0.,
    #     :eq_investment_lag => 0.,
    #     :eq_profit_lag => 0.,
    #     :eq_unemployment_lag_1 => 0.,
    #     :eq_unemployment_lag_2 => 0.,
    #     :eq_unemployment_lag => 0.,
    #     :eq_government_spending_lag => 0.,
    #     :eq_lump_sum_transfers_lag => 0.,
    #     #shocks
    #     :eq_ZZ_1 => 0., 
    #     :eq_ZZ_2 => 0., 
    #     :eq_ZZ_3 => 0.,
    #     :eq_ZZ_4 => 0.,
    #     :eq_fiscal_liability => 0.,
    #     :eq_z => 0.,
    #     :eq_ψ => 0.,
    #     :eq_η => 0.,
    #     :eq_D => 0.,
    #     :eq_GG => 0.,
    #     :eq_ι => 0.,
    #     :eq_BB => 0.,
    #     :eq_ψ_w => 0.,
    #     :eq_MP => 0.,
    #     :eq_pm => 0.,
    #     :eq_eps_1_ind => 0.,
    #     :eq_eps_2_ind => 0.,
    #     :eq_eps_3_ind => 0.,
    #     :eq_eps_4_ind => 0.,
    #     :eps_QE_ind => 0.,
    #     :eps_RP_ind => 0.,
    #     :eps_B_F_ind => 0.,
    #     :eps_BB_ind => 0.,
    #     :eps_Z_ind => 0.,
    #     :eps_G_ind => 0.,
    #     :eps_D_ind => 0.,
    #     :eps_R_ind => 0.,
    #     :eps_iota_ind => 0.,
    #     :eps_eta_ind => 0.,
    #     :eps_w_ind => 0.,
    #     #CONTROLS
    #     #summary
    #     :MRS_ind => 0.,
    #     :A_hh_ind => 0.,
    #     :B_hh_ind => 0.,
    #     :C_ind => 0.,
    #     :N_ind_1 => 0.,
    #     :N_ind_2 => 0.,
    #     :L_ind_1 => 0.,
    #     :L_ind_2 => 0.,
    #     :UB_ind => 0.,
    #     :unemp_ind_1 => 0.,
    #     :unemp_ind_2 => 0.,
    #     :unemp_ind => 0.,
    #     :U1 => 0.,
    #     :U2 => 0.,
    #     #controls
    #     :eq_control_capital => 0.,
    #     :eq_control_bond => 0.,
    #     :eq_control_government_bond => 0.,
    #     :eq_control_tax => 0.,
    #     :eq_control_transfer => 0.,
    #     :eq_control_government_spending => 0.,
    #     :eq_control_lambda => 0.,
    #     :eq_control_inflation => 0.,
    #     :eq_control_vacancies_1 => 0.,
    #     :eq_control_vacancies_2 => 0.,
    #     :eq_control_j => 0.,
    #     :eq_control_mpl_1 => 0.,
    #     :eq_control_mpl_2 => 0.,
    #     :eq_control_elasticity_v => 0.,
    #     :eq_control_output => 0.,
    #     :eq_control_profit => 0.,
    #     :eq_control_mpk => 0.,
    #     :eq_control_mpa => 0.,
    #     :eq_control_marginal_cost => 0.,
    #     :eq_control_n_1 => 0.,
    #     :eq_control_n_2 => 0.,
    #     :eq_control_M_1 => 0.,
    #     :eq_control_M_2 => 0.,
    #     :eq_control_f_1 => 0.,
    #     :eq_control_f_2 => 0.,
    #     :eq_control_zz => 0.,
    #     :eq_control_xx => 0.,
    #     :eq_control_vv => 0.,
    #     :eq_control_ee => 0.,
    #     :eq_control_cb => 0.,
    #     :eq_control_profit_fi => 0.,
    #     :eq_control_rra => 0.,
    #     :eq_control_rr => 0.,
    #     :eq_control_investment => 0.,
    #     :eq_control_x_k => 0.,
    #     #observables
    #     :eq_control_a_g_obs => 0.,
    #     :eq_control_y_obs => 0.,
    #     :eq_control_c_obs => 0.,
    #     :eq_control_i_obs => 0.,
    #     :eq_control_w1_obs => 0.,
    #     :eq_control_w2_obs => 0.,
    #     :eq_control_w_obs => 0.,
    #     :eq_control_profit_obs => 0.,
    #     :eq_unemployment_1_observable => 0.,
    #     :eq_unemployment_2_observable => 0.,
    #     :eq_unemployment_observable => 0.,
    #     :eq_inflation_observable => 0.,
    #     :eq_rate_observable => 0.,
    #     :eq_past_wage_1 => 0.,
    #     :eq_past_wage_2 => 0.,
    #     :eq_past_wage => 0.,
    #     :eq_government_observable => 0.,
    #     :eq_past_YY => 0.,
    #     :eq_past_CC => 0.,
    #     :eq_past_II => 0.,
    #     :eq_past_profit => 0.,
    #     :eq_past_uu_1 => 0.,
    #     :eq_past_uu_2 => 0.,
    #     :eq_past_uu => 0.,
    #     :eq_past_GG => 0.,
    #     :eq_past_A_g => 0.,
    #     :eq_l_lambda_1 => 0.,
    #     :eq_l_lambda_2 => 0.,
    #     :eq_past_q => 0.,
    #     :eq_xI => 0.,
    #     :eq_eta2 => 0.,
    #     :eq_iota2 => 0.,
    #     :eq_past_lt2 => 0.,
    #     :eq_past_g2 => 0.,
    #     :eq_lt_obs => 0.,
    #     :eq_b_gov_ncp2 => 0.,
    # )

    # zlb_equations = Set([
    #     (:eq_rate_monetary_policy, :eq_rate_monetary_policy_ZLB),
    #     (:eq_R_star, :eq_R_star_ZLB),
    #     (:eq_central_bank_assets, :eq_central_bank_assets_ZLB),
    #     (:eq_quantitative_easing, :eq_quantitative_easing_ZLB)
    # ])

    # t1 = time()
    # println("state the f eqn: $(t1 - t0) seconds")

    #set regime for testing
    reg = 1
    F = Fsys_agg(OrderedDict{Symbol, Any}(), m, grid, StateSS, ControlSS, State_zero, Control_zero, State_zero, Control_zero, state_id, control_id, reg)

    # t1 = time()
    # println("call fsys: $(t1 - t0) seconds")

    eq_keys = collect(keys(F))


    # t1 = time()
    # println("collect keys: $(t1 - t0) seconds")

    # flatten f dict into a vector of scalar, issue with J, might cause indexing issue later on
    function flatten_F!(F_vec::AbstractVector{T}, F_dict::OrderedDict{Symbol, Any}, eq_keys::AbstractVector{Symbol}) where {T}
        idx = 1
        @inbounds for k in eq_keys
            v = get(F_dict, k, zero(T))
            if v isa AbstractArray
                n = length(v)
                copyto!(view(F_vec, idx:(idx + n - 1)), vec(v))
                idx += n
            else
                F_vec[idx] = v
                idx += 1
            end
        end
    end

    #determine flat output size
    n_eqs_flat = sum(k -> (F[k] isa AbstractArray ? length(F[k]) : 1), eq_keys)


    # t1 = time()
    # println("flatten: $(t1 - t0) seconds")

    # x = [Xt1; Yt1; Xt; Yt] Xt1 is state_t, Yt1 is control_t+1, Xt is state_t-1, Yt is control_t
    n_x = 2*nState + 2*nCtrl
    BA_agg = zeros(Float64, n_eqs_flat, n_x)
    F_dict = OrderedDict{Symbol, Any}()

    function obj_fnct_agg(F_vec, x)
        
        # t2 = time()

        @views Xt1 = x[1:nState]
        @views Yt1 = x[nState+1:nState+nCtrl]
        @views Xt  = x[nState+nCtrl+1:2*nState+nCtrl]
        @views Yt  = x[2*nState+nCtrl+1:end]
        empty!(F_dict)
        
        # t3 = time()
        # println("flaobjfcn 1: $(t3 - t2) seconds")
        Fsys_agg(F_dict, m, grid, StateSS, ControlSS, Xt1, Yt1, Xt, Yt, state_id, control_id, reg)
        
        # t4 = time()
        # println("flaobjfcn 2: $(t4 - t3) seconds")

        flatten_F!(F_vec, F_dict, eq_keys)

        # t5 = time()
        # println("flaobjfcn 3: $(t5 - t4) seconds")
    end

    #autodiff jacobian
    #ForwardDiff.jacobian!(BA_agg, obj_fnct_agg, zeros(n_eqs_flat), zeros(n_x))
    # autodiff jacobian with preallocated buffers/config
    f0 = zeros(Float64, n_eqs_flat)
    x0 = zeros(Float64, n_x)
    cfg = ForwardDiff.JacobianConfig(obj_fnct_agg, f0, x0)
    ForwardDiff.jacobian!(BA_agg, obj_fnct_agg, f0, x0, cfg)

    
    # t1 = time()
    # println("forwarddiff: $(t1 - t0) seconds")

    # aggregate-only columns (drop distribution block indices)
    dist_state_keys = Set([:marginal_pdf_b_t, :marginal_pdf_a_t, :marginal_pdf_se_t, :copula_t])
    dist_ctrl_keys  = Set([:Value_t, :mutil_c_t, :Va_t])

    agg_state_cols = vcat([collect(state_id[s])   for s in keys(state_id)   if s ∉ dist_state_keys]...)
    agg_ctrl_cols  = vcat([collect(control_id[s]) for s in keys(control_id) if s ∉ dist_ctrl_keys]...)

    # t1 = time()
    # println("drop dist: $(t1 - t0) seconds")

    #zlb equations 
    # num_zlb_equations = length(zlb_equations)

    # eq_index = Dict{Symbol, Int}(k => i for (i, k) in pairs(eq_keys))
    # zlb_nonzlb_idx_pairs = Tuple{Int, Int}[
    #     (eq_index[nonzlb], eq_index[zlb]) for (nonzlb, zlb) in zlb_equations
    # ]

    # t1 = time()
    # println("zlb sorting: $(t1 - t0) seconds")


    #matching donggyu dims
    F1_ad = BA_agg[:, 1:nState][:, agg_state_cols]                        # ∂F/∂Xt1 (agg cols)
    F2_ad = BA_agg[:, nState+1:nState+nCtrl][:, agg_ctrl_cols]            # ∂F/∂Yt1 (agg cols)
    F3_ad = BA_agg[:, nState+nCtrl+1:2*nState+nCtrl][:, agg_state_cols]  # ∂F/∂Xt  (agg cols)
    F4_ad = BA_agg[:, 2*nState+nCtrl+1:end][:, agg_ctrl_cols]            # ∂F/∂Yt  (agg cols)

    os = Int(getgrid(grid, :os))
    oc = Int(getgrid(grid, :oc))
    F21_ad = F1_ad[1:os, end-os+1:end]
    F22_ad = F2_ad[1:os, end-oc+1:end]
    F23_ad = F3_ad[1:os, end-os+1:end]
    F24_ad = F4_ad[1:os, end-oc+1:end]

    F41_ad = F1_ad[os+1:end, end-os+1:end]
    F42_ad = F2_ad[os+1:end, end-oc+1:end]
    F43_ad = F3_ad[os+1:end, end-os+1:end]
    F44_ad = F4_ad[os+1:end, end-oc+1:end]



    # FZLB_ad = F1_ad[end-num_zlb_equations+1 : end, :]
    # FZLB_ad = F2_ad[end-num_zlb_equations+1 : end, :]
    # FZLB_ad = F3_ad[end-num_zlb_equations+1 : end, :]
    # FZLB_ad = F4_ad[end-num_zlb_equations+1 : end, :]



    #TODO ADD THIS
    # let a_gaux_col = findfirst(==(state_id[:A_gaux_t]), agg_state_cols)
    #     replace_pairs = [(F23_ad, 5),
    #                     (F41_ad, 41), (F41_ad, 43),
    #                     (F43_ad, 8), (F43_ad, 10), (F43_ad, 60), (F43_ad, 69)]
    #     for (mat, i) in replace_pairs
    #         mat[i, a_gaux_col] = mat[i, end]
    #     end
    # end
    # F21_ad = F21_ad[:, 1:end-1]
    # F23_ad = F23_ad[:, 1:end-1]
    # F41_ad = F41_ad[:, 1:end-1]
    # F43_ad = F43_ad[:, 1:end-1]



    # F21_ad_zlb = copy(F21_ad)
    # F22_ad_zlb = copy(F22_ad)
    # F23_ad_zlb = copy(F23_ad)
    # F24_ad_zlb = copy(F24_ad)

    # trim ag_t
    F1_ad_trim = F1_ad[:, 1:end-1]
    F3_ad_trim = F3_ad[:, 1:end-1]

    # for (nonzlb_row, zlb_row) in zlb_nonzlb_idx_pairs
    #     F21_ad_zlb[nonzlb_row, :] = F1_ad_trim[zlb_row + 4, :] #TODO instead of maual +4 account for J better
    #     F22_ad_zlb[nonzlb_row, :] = F2_ad[zlb_row + 4, :]
    #     F23_ad_zlb[nonzlb_row, :] = F3_ad_trim[zlb_row + 4, :]
    #     F24_ad_zlb[nonzlb_row, :] = F4_ad[zlb_row + 4, :]
    # end                                 

    # t1 = time()
    # println("trimming: $(t1 - t0) seconds")


    return F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad

end







