using Revise
using DSGE
using MAT: matread
using JLD2



# Load MATLAB input fixtures
input_path = joinpath(@__DIR__, "../input_data/dyn_ZLB_tvcopula_new.mat")
println("\nLoading input fixtures from $input_path")
input_data = matread(input_path)
final_table = input_data["final_table"]

# result = DSGE.ss_CSC()
# @save "ss_CSC_result.jld2" result

@load "ss_CSC_result.jld2" result


param    = result["param"]
grid     = result["grid"]
SS_stats = result["SS_stats"]
# P_SE     = result["P_SE"]
mu_dist  = result["mu_dist"]
Value    = result["Value"]
mutil_c  = result["mutil_c"]
Va       = result["Va"]

# P_SE = result["param"]["P_SE"]

m = DSGE.kCSC()


result = DSGE.dyn_CSC_tvcopula(
    final_table,
    param, grid, SS_stats,
    mu_dist, Value, mutil_c, Va;
    pool=nothing
)