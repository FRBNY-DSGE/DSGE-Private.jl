import Pkg
Pkg.activate("/home/rcemzp01/.julia/environments/v1.5")

using DSGE
using JLD2
using MAT: matread

# Load the current 9kCSC dynamics method into the DSGE module.  The package
# currently includes the earlier kCSC driver by default.
Base.include(DSGE, joinpath(@__DIR__, "dynamics", "dyn_CSC_tvcopula.jl"))

fixture_root = "/data/dsge_data_dir/k-shaped consumption/dynamics/kshape_economy/models/kCSC"
input = matread(joinpath(fixture_root, "input_data", "dyn_ZLB_tvcopula_new.mat"))
ss = load(joinpath(fixture_root, "tests", "ss_CSC_result.jld2"), "result")

param = ss["param"]
grid = ss["grid"]
SS_stats = ss["SS_stats"]
mu_dist = ss["mu_dist"]
Value = ss["Value"]
mutil_c = ss["mutil_c"]
Va = ss["Va"]

result = DSGE.dyn_CSC_tvcopula(input["final_table"], param, grid, SS_stats,
                               mu_dist, Value, mutil_c, Va; pool=nothing)
ssinfo = load(joinpath(@__DIR__, "SSInfo.jld2"))
irfs = DSGE.IRFs_CSC_tvcopula(result["hx"], result["gx"],
    ssinfo["Xss"], ssinfo["Yss"], ssinfo["Gamma_state"],
    ssinfo["Gamma_control"], result["param"], grid, SS_stats)

@save joinpath(@__DIR__, "irfs_csc.jld2") result irfs
println("Saved: ", joinpath(@__DIR__, "irfs_csc.jld2"))
