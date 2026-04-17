using Revise
using Test
using JLD2
using DSGE
using MAT
using OrderedCollections: OrderedDict
using CSV

input_path = joinpath(@__DIR__, "../data/inputs/state_reduc_tvcopula.mat")
println("\nLoading input fixtures from $input_path")
input_data = matread(input_path)
#TODO: THESE SHOULD BE JLD2 inSTEAD OF MAT
mu_dist = input_data["mu_dist"]
Value = input_data["Value"]
mutil_c = input_data["mutil_c"]
Va = input_data["Va"]


jld2file = "../data/XssYss.jld2"
@load jld2file StateSS ControlSS SS_stats grid param
# grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "nCOP", "oc", "os", "numstates", "s_dist", "s", "K"])

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

#load in basejacob
mat_contents = matread("../data/TESTJACOBBASE.mat")
Jacob_base = mat_contents["Jacob_base"]

compute_system(m, Jacob_base)