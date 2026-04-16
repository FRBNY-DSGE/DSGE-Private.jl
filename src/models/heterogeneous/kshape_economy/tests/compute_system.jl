using Revise
using Test
using JLD2
using DSGE
using MAT
using OrderedCollections: OrderedDict
using CSV


jld2file = "../data/XssYss.jld2"
@load jld2file StateSS ControlSS SS_stats grid param
grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "nCOP", "oc", "os", "numstates", "s_dist", "s", "K"])

m = mBBQ()



#load in basejacob
mat_contents = matread("../data/TESTJACOBBASE.mat")
Jacob_base = mat_contents["Jacob_base"]

compute_system(m, StateSS, ControlSS, SS_stats, grid, param, Jacob_base)