using Revise
using Test
using JLD2
using DSGE
using MAT
using OrderedCollections: OrderedDict
using CSV
using StateSpaceRoutines

input_path = joinpath(@__DIR__, "../models/mBBQ/data/inputs/state_reduc_tvcopula.mat")
println("\nLoading input fixtures from $input_path")
input_data = matread(input_path)
#TODO: THESE SHOULD BE JLD2 inSTEAD OF MAT
mu_dist = input_data["mu_dist"]
Value = input_data["Value"]
mutil_c = input_data["mutil_c"]
Va = input_data["Va"]

jld2file = "../models/mBBQ/data/sgufx.jld2"
@load jld2file F1_aux_rep F2_aux_rep F3_aux_rep F4_aux_rep

jld2file = "../models/mBBQ/data/XssYss.jld2"
@load jld2file StateSS ControlSS SS_stats grid param
# grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "nCOP", "oc", "os", "numstates", "s_dist", "s", "K"])

# mat_contents = matread("../models/mBBQ/data/pq_in2.mat")
# ZLB_duration_1 = mat_contents["ZLB_duration_1"]
# Fss_ZLB        = mat_contents["Fss_ZLB"]


m = mBBQ()


m.dicts[:grid] = grid
m.dicts[:SS_stats] = SS_stats
m.dicts[:param] = param
m.grids[:StateSS] = StateSS
m.grids[:ControlSS] = ControlSS

m.grids[:mu_dist] = mu_dist
m.grids[:Value] = Value
m.grids[:mutil_c] = mutil_c
m.grids[:Va] = Va
# m.dicts[:ZLB_duration_1] = ZLB_duration_1

Jacob_base2 = DSGE.save_jacob_base(F1_aux_rep, F2_aux_rep, F3_aux_rep, F4_aux_rep, m)


#load in basejacob
mat_contents = matread("../models/mBBQ/data/TESTJACOBBASE.mat")
Jacob_base = mat_contents["Jacob_base"] #for testing use, this but in prod use save_jacob_base based on tvcopula

m.dicts[:Jacob_base] = Jacob_base2

# TESTING!, need to make call to F_sys_ZLB_QE
# Fss_ZLB = zeros(632, 1)
# m.dicts[:Fss_ZLB] = Fss_ZLB

regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es, flag = DSGE.compute_system_nozlb!(m)

if flag == true
    loglh, _s_pred, _P_pred, _s_filt, _P_filt, _s0, _P0, _sT, _PT =
    StateSpaceRoutines.kalman_filter(regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es;
                  outputs = [:loglh, :pred, :filt])

else
    loglh = Inf

end