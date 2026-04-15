using Revise
using Test
using JLD2
using DSGE
using MAT
using OrderedCollections: OrderedDict


jld2file = "../data/XssYss.jld2"
@load jld2file StateSS ControlSS SS_stats grid param
grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "nCOP", "oc", "os", "numstates", "s_dist", "s", "K"])

m = mBBQ()

F21_ad, F22_ad, F23_ad, F24_ad, F41_ad, F42_ad, F43_ad, F44_ad, F21_ad_zlb, F22_ad_zlb, F23_ad_zlb, F24_ad_zlb, F41_ad_zlb, F42_ad_zlb, F43_ad_zlb, F44_ad_zlb = DSGE.jacobian(m, StateSS, ControlSS, SS_stats, grid, param)


#LOAD IN OLD JACOB

#load in nonQE donggyu original jacobians
mat_contents = matread("../data/TESTJACOB.mat")
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
os = grid[:os]
oc = grid[:oc]

F21_aux_trim = F21_aux[: , end-os+1:end]
F23_aux_trim = F23_aux[: , end-os+1:end]
F41_aux_trim = F41_aux[: , end-os+1:end]
F43_aux_trim = F43_aux[: , end-os+1:end]

F22_aux_trim = F22_aux[: , end-oc+1:end]
F24_aux_trim = F24_aux[: , end-oc+1:end]
F42_aux_trim = F42_aux[: , end-oc+1:end]
F44_aux_trim = F44_aux[: , end-oc+1:end]



mat_contents = matread("../data/TESTJACOBZLB.mat")
out_jacob_zlb = mat_contents["out_Jacob_ZLB"]
F21_aux_zlb = out_jacob_zlb["F21_aux"]
F22_aux_zlb = out_jacob_zlb["F22_aux"]
F23_aux_zlb = out_jacob_zlb["F23_aux"]
F24_aux_zlb = out_jacob_zlb["F24_aux"]
F41_aux_zlb = out_jacob_zlb["F41_aux"]
F42_aux_zlb = out_jacob_zlb["F42_aux"]
F43_aux_zlb = out_jacob_zlb["F43_aux"]
F44_aux_zlb = out_jacob_zlb["F44_aux"]

#r1_start, r1_end = 89, 118
#r2_start, r2_end = 337, 405
# os = grid[:os]
# oc = grid[:oc]

F21_aux_trim_zlb = F21_aux_zlb[: , end-os+1:end]
F23_aux_trim_zlb = F23_aux_zlb[: , end-os+1:end]
F41_aux_trim_zlb = F41_aux_zlb[: , end-os+1:end]
F43_aux_trim_zlb = F43_aux_zlb[: , end-os+1:end]

F22_aux_trim_zlb = F22_aux_zlb[: , end-oc+1:end]
F24_aux_trim_zlb = F24_aux_zlb[: , end-oc+1:end]
F42_aux_trim_zlb = F42_aux_zlb[: , end-oc+1:end]
F44_aux_trim_zlb = F44_aux_zlb[: , end-oc+1:end]


#save as csv for viewing 
# mkpath("csv")
# for (name, mat) in [("F21_ad", F21_ad), ("F22_ad", F22_ad), ("F23_ad", F23_ad), ("F24_ad", F24_ad),
#                     ("F41_ad", F41_ad), ("F42_ad", F42_ad), ("F43_ad", F43_ad), ("F44_ad", F44_ad),
#                     ("F21_aux", F21_aux_trim), ("F23_aux", F23_aux_trim),
#                     ("F41_aux", F41_aux_trim), ("F43_aux", F43_aux_trim),
#                     ("F22_aux", F22_aux_trim), ("F24_aux", F24_aux_trim),
#                     ("F42_aux", F42_aux_trim), ("F44_aux", F44_aux_trim)]
#     CSV.write("csv/$name.csv", DataFrame(mat, :auto))
# end

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

#indexing
names = collect(keys(F))
#state_eqn_names = vcat(fill("dont know name", 88), names[1:os])
state_eqn_names = names[1:os]
#control_eqn_names = names[os+1:end]
control_eqn_names = String[]
for k in names[os+1:end]
    v = F[k]
    n = v isa AbstractArray ? length(v) : 1
    if n > 1
        append!(control_eqn_names, ["$(k)[$(i)]" for i in 1:n])
    else
        push!(control_eqn_names, String(k))
    end
end

state_id, control_id = DSGE.build_indices(grid, length(StateSS), length(ControlSS)) #duplicate
names2 = collect(keys(state_id))
#state_var_names = vcat(fill("dont know name", 88), names2[5:end])
state_var_names = names2[5:end]

names3 = collect(keys(control_id))
control_var_names = names3[4:end]
new_syms = [:J_t2, :J_t3, :J_t4, :J_t5]
splice!(control_var_names, 18:17, new_syms)


control_var_name = []
#for i = 17692:17732

function compare_matrices(AD, FD, name::String, eqn_state::Bool, var_state; atol=1e-4,
                          exceptions=Vector{Vector{Int}}())
    exc_set = Set(Tuple(e) for e in exceptions)
    @testset "$name" begin
        @test size(AD) == size(FD)
        if size(AD) == size(FD)
            diffs = findall(i -> !isapprox(AD[i], FD[i]; atol=atol), eachindex(AD))
            filter!(i -> begin
                r, c = Tuple(CartesianIndices(AD)[i])
                (r, c) ∉ exc_set
            end, diffs)
            if !isempty(diffs)
                println("  $(length(diffs)) differing elements in $name:")
                for i in diffs
                    r, c = Tuple(CartesianIndices(AD)[i])
                    eqn = eqn_state ? state_eqn_names[r] : control_eqn_names[r]
                    var = var_state ? state_var_names[c] : control_var_names[c]
                    println("eqn (row) $eqn, var (col) $var") 
                    println("[$r,$c]  julia=$(AD[i])  original=$(FD[i])  ")
                    println("===============================================")
                end
            end
            #@test isempty(diffs)
        end
    end
end

println("TEST TAYLOR")

compare_matrices(F21_ad, F21_aux_trim, "F21 (state eqns wrt x_t)",   true,  true)#, exceptions=[[5,22]])
compare_matrices(F22_ad, F22_aux_trim, "F22 (state eqns wrt y_t+1)", true,  false)
compare_matrices(F23_ad, F23_aux_trim, "F23 (state eqns wrt x_t-1)", true,  true)
compare_matrices(F24_ad, F24_aux_trim, "F24 (state eqns wrt y_t)",   true,  false)
compare_matrices(F41_ad, F41_aux_trim, "F41 (ctrl eqns wrt x_t)",    false, true)#, exceptions = [[41,26]])
compare_matrices(F42_ad, F42_aux_trim, "F42 (ctrl eqns wrt y_t+1)",  false, false)
compare_matrices(F43_ad, F43_aux_trim, "F43 (ctrl eqns wrt x_t-1)",  false, true)
compare_matrices(F44_ad, F44_aux_trim, "F44 (ctrl eqns wrt y_t)",    false, false)#, exceptions=[[41,8], [41,42]])


println("TEST ZLB")
compare_matrices(F21_ad_zlb, F21_aux_trim_zlb, "F21 (state eqns wrt x_t)",   true,  true)#, exceptions=[[5,22]])
compare_matrices(F22_ad_zlb, F22_aux_trim_zlb, "F22 (state eqns wrt y_t+1)", true,  false)
compare_matrices(F23_ad_zlb, F23_aux_trim_zlb, "F23 (state eqns wrt x_t-1)", true,  true)
compare_matrices(F24_ad_zlb, F24_aux_trim_zlb, "F24 (state eqns wrt y_t)",   true,  false)
compare_matrices(F41_ad_zlb, F41_aux_trim_zlb, "F41 (ctrl eqns wrt x_t)",    false, true)#, exceptions = [[41,26]])
compare_matrices(F42_ad_zlb, F42_aux_trim_zlb, "F42 (ctrl eqns wrt y_t+1)",  false, false)
compare_matrices(F43_ad_zlb, F43_aux_trim_zlb, "F43 (ctrl eqns wrt x_t-1)",  false, true)
compare_matrices(F44_ad_zlb, F44_aux_trim_zlb, "F44 (ctrl eqns wrt y_t)",    false, false)#, exceptions=[[41,8], [41,42]])
