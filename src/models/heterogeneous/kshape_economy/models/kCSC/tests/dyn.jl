using Revise
using DSGE
using MAT: matread



# Load MATLAB input fixtures
input_path = joinpath(@__DIR__, "../input_data/dyn_ZLB_tvcopula_new.mat")
println("\nLoading input fixtures from $input_path")
input_data = matread(input_path)
final_table = input_data["final_table"]

result = DSGE.ss_CSC()

param    = result["param"]
grid     = result["grid"]
SS_stats = result["SS_stats"]
# P_SE     = result["P_SE"]
mu_dist  = result["mu_dist"]
Value    = result["Value"]
mutil_c  = result["mutil_c"]
Va       = result["Va"]

P_SE = result["param"]["P_SE"]


result = DSGE.dyn_ZLB_tvcopula_new(
    final_table,
    P_SE, grid, SS_stats,
    mu_dist, Value, mutil_c, Va
)