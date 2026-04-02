using Revise
using LinearAlgebra
using OrderedCollections: OrderedDict
using DSGE
using JLD2
using ForwardDiff
using MAT
using CSV
using DataFrames

#=
include("jacobian.jl")
include("helpers/jachelpers/index2.jl")    
include("helpers/jachelpers/macros2.jl")   
=#

jld2file = "data/XssYss.jld2"
@load jld2file StateSS ControlSS SS_stats grid param

m = mBBQ()

#one time setup to move prev donggyu format to bbl format
grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "nCOP", "oc", "os", "numstates", "s_dist", "s", "K"])
state_id, control_id = DSGE.build_indices(grid, length(StateSS), length(ControlSS))
ss = DSGE.build_ss(param, SS_stats)

#test state and control controls of zero
State_zero = zeros(length(StateSS))
Control_zero = zeros(length(ControlSS))

nState  = length(StateSS)
nCtrl   = length(ControlSS)



#F = Dict{Symbol,Any}() #TODO: need to change typing of this
#offload to different file, must be same ordering as in idx of donggyu code
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
    :eps_B_F_ind => 0.,
    :eps_RP_ind => 0.,
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
    :unemp_obs_ind => 0., #TODO: add rest
    :inf_obs_ind => 0.,
    :R_obs_ind => 0.,
    :pastw_ind => 0.,
    :G_obs_ind => 0.,
    :pastYY_ind => 0.,
    :pastCC_ind => 0.,
    :pastII_ind => 0.,
    :pastPPROFIT_ind => 0.,
    :pastuu_ind => 0.,
    :pastGG_ind => 0.,
    :pastA_g_ind => 0.,
    :l_lambda_ind => 0.,
    :pastQ_ind => 0.,
    :x_I_ind => 0.,
    :eta2_ind => 0.,
    :iota2_ind => 0.,
    :pastLT2_ind => 0.,
    :pastG2_ind => 0.,
    :LT_obs_ind => 0.,
    :B_gov_ncp2_ind => 0.
)

#set regime for testing
reg = 1
F = DSGE.Fsys_agg(F, m, grid, StateSS, ControlSS, State_zero, Control_zero, State_zero, Control_zero, ss, state_id, control_id, reg)

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
    DSGE.Fsys_agg(F_dict, m, grid, StateSS, ControlSS, Xt1, Yt1, Xt, Yt, ss, state_id, control_id, reg)
    flatten_F!(F_vec, F_dict, eq_keys)
end

#autodiff jacobian
ForwardDiff.jacobian!(BA_agg, obj_fnct_agg, zeros(n_eqs_flat), zeros(n_x))

# aggregate-only columns (drop distribution block indices)
dist_state_keys = Set([:marginal_b_t, :marginal_a_t, :marginal_se_t, :copula_t])
dist_ctrl_keys  = Set([:Value_t, :mutil_c_t, :Va_t])

agg_state_cols = vcat([collect(state_id[s])   for s in keys(state_id)   if s ∉ dist_state_keys]...)
agg_ctrl_cols  = vcat([collect(control_id[s]) for s in keys(control_id) if s ∉ dist_ctrl_keys]...)


#matching donggyu dims
F1_ad = BA_agg[:, 1:nState][:, agg_state_cols]                        # ∂F/∂Xt1 (agg cols)
F2_ad = BA_agg[:, nState+1:nState+nCtrl][:, agg_ctrl_cols]            # ∂F/∂Yt1 (agg cols)
F3_ad = BA_agg[:, nState+nCtrl+1:2*nState+nCtrl][:, agg_state_cols]  # ∂F/∂Xt  (agg cols)
F4_ad = BA_agg[:, 2*nState+nCtrl+1:end][:, agg_ctrl_cols]            # ∂F/∂Yt  (agg cols)

os = grid[:os]
oc = grid[:oc]
F21_ad = F1_ad[1:os, :]
F22_ad = F2_ad[1:os, :]
F23_ad = F3_ad[1:os, :]
F24_ad = F4_ad[1:os, :]

F41_ad = F1_ad[os+1:end, :]
F42_ad = F2_ad[os+1:end, :]
F43_ad = F3_ad[os+1:end, :]
F44_ad = F4_ad[os+1:end, :]


#load in nonQE donggyu original jacobians
mat_contents = matread("data/TESTJACOB.mat")
out_jacob = mat_contents["out_Jacob"]
F21_aux = out_jacob["F21_aux"]
F22_aux = out_jacob["F22_aux"]
F23_aux = out_jacob["F23_aux"]
F24_aux = out_jacob["F24_aux"]
F41_aux = out_jacob["F41_aux"]
F42_aux = out_jacob["F42_aux"]
F43_aux = out_jacob["F43_aux"]
F44_aux = out_jacob["F44_aux"]

#r1_start, r1_end = 89, 118
#r2_start, r2_end = 337, 405

F21_aux_trim = F21_aux[: , end-os+1:end]
F23_aux_trim = F23_aux[: , end-os+1:end]
F41_aux_trim = F41_aux[: , end-os+1:end]
F43_aux_trim = F43_aux[: , end-os+1:end]

F22_aux_trim = F22_aux[: , end-oc+1:end]
F24_aux_trim = F24_aux[: , end-oc+1:end]
F42_aux_trim = F42_aux[: , end-oc+1:end]
F44_aux_trim = F44_aux[: , end-oc+1:end]


#save as csv for viewing 
mkpath("csv")
for (name, mat) in [("F21_ad", F21_ad), ("F22_ad", F22_ad), ("F23_ad", F23_ad), ("F24_ad", F24_ad),
                    ("F41_ad", F41_ad), ("F42_ad", F42_ad), ("F43_ad", F43_ad), ("F44_ad", F44_ad),
                    ("F21_aux", F21_aux_trim), ("F23_aux", F23_aux_trim),
                    ("F41_aux", F41_aux_trim), ("F43_aux", F43_aux_trim),
                    ("F22_aux", F22_aux_trim), ("F24_aux", F24_aux_trim),
                    ("F42_aux", F42_aux_trim), ("F44_aux", F44_aux_trim)]
    CSV.write("csv/$name.csv", DataFrame(mat, :auto))
end





