using Revise
using DSGE, ModelConstructors, SMC
using Test
using MAT
using OrderedCollections: OrderedDict
using ForwardDiff
using CSV
using DataFrames
using DSGE: @sslogdeviations2levels, @sslogdeviations2levels_unprimekeys, @unpack_and_first

include("../helpers/index.jl")
include("../fsys_agg.jl")

input = matread("../input_data/update_Jacob.mat")

param = input["param"]
grid = input["grid"]
SS_stats = input["SS_stats"]
Xss = input["Xss"]
Yss = input["Yss"]

StateSS = vec(Xss)
ControlSS = vec(Yss)

ss = SS_stats

m = kCSC9()
m.dicts[:SS_stats] = ss
m.dicts[:grid] = grid
m.dicts[:param] = param

grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "ns_1", "ns_2", "nCOP", "oc", "os", "numstates", "s_dist", "s", "K"])
state_id, control_id = build_indices_kCSC9(grid, length(StateSS), length(ControlSS))

#test state and control controls of zero
State_zero = zeros(length(StateSS))
Control_zero = zeros(length(ControlSS))

nState  = length(StateSS)
nCtrl   = length(ControlSS)

reg = 1
F = Fsys_agg(OrderedDict{Symbol, Any}(), m, grid, StateSS, ControlSS, State_zero, Control_zero, State_zero, Control_zero, state_id, control_id)

eq_keys = collect(keys(F))

#=
@testset "fsys_agg key sanity" begin
    @test :eq_wage_1 in eq_keys
    @test :eq_wage_2 in eq_keys
    @test :eq_wage_avg in eq_keys
    @test :eq_control_vacancies_1 in eq_keys
    @test :eq_control_vacancies_2 in eq_keys
    @test :eq_control_mpl_1 in eq_keys
    @test :eq_control_mpl_2 in eq_keys
    @test :eq_control_M_1 in eq_keys
    @test :eq_control_M_2 in eq_keys
    @test :eq_control_f_1 in eq_keys
    @test :eq_control_f_2 in eq_keys
    @test :eq_unemployment_1_observable in eq_keys
    @test :eq_unemployment_2_observable in eq_keys
end
=#

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
    Fsys_agg(F_dict, m, grid, StateSS, ControlSS, Xt1, Yt1, Xt, Yt, state_id, control_id, reg)
    flatten_F!(F_vec, F_dict, eq_keys)
end

#autodiff jacobian
ForwardDiff.jacobian!(BA_agg, obj_fnct_agg, zeros(n_eqs_flat), zeros(n_x))

# aggregate-only columns (drop distribution block indices)
dist_state_keys = Set([:marginal_b′_t, :marginal_a′_t, :marginal_se′_t, :COP])
dist_ctrl_keys  = Set([:VALUE_t, :mutil_c_t, :Va_t])

agg_state_cols = vcat([collect(state_id[s])   for s in keys(state_id)   if s ∉ dist_state_keys]...)
agg_ctrl_cols  = vcat([collect(control_id[s]) for s in keys(control_id) if s ∉ dist_ctrl_keys]...)


#matching donggyu dims
F1_ad = BA_agg[:, 1:nState][:, agg_state_cols]                        # ∂F/∂Xt1 (agg cols)
F2_ad = BA_agg[:, nState+1:nState+nCtrl][:, agg_ctrl_cols]            # ∂F/∂Yt1 (agg cols)
F3_ad = BA_agg[:, nState+nCtrl+1:2*nState+nCtrl][:, agg_state_cols]  # ∂F/∂Xt  (agg cols)
F4_ad = BA_agg[:, 2*nState+nCtrl+1:end][:, agg_ctrl_cols]            # ∂F/∂Yt  (agg cols)

os = Int(grid[:os])
oc = Int(grid[:oc])
n_hh_summary_julia  = 3   # Income_Tax, J_bar_1, J_bar_2 remain in fsys_agg
n_hh_summary_matlab = 17  # MATLAB still has all 17 (MRS..J_bar_2)

F21_ad = F1_ad[1:os, :]
F22_ad = F2_ad[1:os, :]
F23_ad = F3_ad[1:os, :]
F24_ad = F4_ad[1:os, :]

F41_ad = F1_ad[os+n_hh_summary_julia+1:end, :]
F42_ad = F2_ad[os+n_hh_summary_julia+1:end, :]
F43_ad = F3_ad[os+n_hh_summary_julia+1:end, :]
F44_ad = F4_ad[os+n_hh_summary_julia+1:end, :]


#=
##edit remove col 42
let a_gaux_col = findfirst(==(state_id[:A_gaux_t]), agg_state_cols)
    replace_pairs = [(F23_ad, 5),
                     (F41_ad, 41), (F41_ad, 43),
                     (F43_ad, 8), (F43_ad, 10), (F43_ad, 60), (F43_ad, 69)]
    for (mat, i) in replace_pairs
        mat[i, a_gaux_col] = mat[i, end]
    end
end=#
#= removed: was paired with commented-out let block that moved A_gaux_t to last col before dropping
F21_ad = F21_ad[:, 1:end-1]
F23_ad = F23_ad[:, 1:end-1]
F41_ad = F41_ad[:, 1:end-1]
F43_ad = F43_ad[:, 1:end-1]
=#


#load in nonQE donggyu original jacobians
output = matread("../output_data/update_Jacob.mat")
out_jacob = output["out_Jacob"]
F21_aux = out_jacob["F21_aux"]
F22_aux = out_jacob["F22_aux"]
F23_aux = out_jacob["F23_aux"]
F24_aux = out_jacob["F24_aux"]
F41_aux = out_jacob["F41_aux"]
F42_aux = out_jacob["F42_aux"]
F43_aux = out_jacob["F43_aux"]
F44_aux = out_jacob["F44_aux"]

F21_aux_trim = F21_aux[:, end-os+1:end]
F23_aux_trim = F23_aux[:, end-os+1:end]
F41_aux_trim = F41_aux[n_hh_summary_matlab+1:end, end-os+1:end]
F43_aux_trim = F43_aux[n_hh_summary_matlab+1:end, end-os+1:end]

F22_aux_trim = F22_aux[:, end-oc+1:end]
F24_aux_trim = F24_aux[:, end-oc+1:end]
F42_aux_trim = F42_aux[n_hh_summary_matlab+1:end, end-oc+1:end]
F44_aux_trim = F44_aux[n_hh_summary_matlab+1:end, end-oc+1:end]


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





