using DSGE
using HDF5, Test, JLD, BenchmarkTools

path =  dirname(@__FILE__)

### Move old results if they exist
results_old = try
    load("results.jld", "results")
    catch
    load("speedtests_old/results.jld", "results")
end
try
    mv("results.jld", "speedtests_old/results.jld", force=true)
catch
end

### Model
m = GHLS()
h5 = h5open("$path/params.h5")
params = read(h5, "params")
update!(m, params)
close(h5)
m.approx.nshockgrid = [7,2,2,2,2,1]

### Set up benchmarking test suite
suite = BenchmarkGroup()

### Test shock details
suite["shocks"] = @benchmarkable get_shockdetails($m.approx.nexogcont, $m.approx.number_shock_values, $m.approx.nshockgrid, $m.approx.exogvarinfo, $m.approx.nexogshock, $m.approx.ns, $m.approx.nexog,$m.parameters, $m.keys)

h5 = h5open("$path/shockdetails.h5")
exoggrid_ref = read(h5, "exoggrid")
shockbounds_ref = read(h5, "shockbounds")
shockdistance_ref = read(h5, "shockdistance")
close(h5)

### Test decr_euler
m.approx.exoggrid, m.approx.shockbounds, m.approx.shockdistance = exoggrid_ref, shockbounds_ref, shockdistance_ref

h5 = h5open("$path/simulatelinear.h5")
endog_emean_ref = read(h5, "endog_emean")
zlbfrequency_ref = read(h5, "zlbfrequency")
msvbounds_ref = read(h5, "msvbounds")
statezlbinfo_ref = read(h5, "statezlbinfo")
close(h5)

m.approx.endog_emean, m.approx.zlbfrequency, m.approx.msvbounds, m.approx.statezlbinfo, m.approx.convergence = endog_emean_ref, zlbfrequency_ref, msvbounds_ref, statezlbinfo_ref, true

h5 = h5open("$path/slopes.h5")
slopeconmsv_ref = read(h5, "slopeconmsv")
slopeconxx_ref = read(h5, "slopeconxx")
close(h5)

m.approx.slopeconmsv, m.approx.slopeconxx = slopeconmsv_ref, slopeconxx_ref

h5 = h5open("$path/initialalphas.h5")
α_initial_ref = read(h5, "initialalphas")
close(h5)


suite["decr_euler"] = @benchmarkable decr_euler($m[:rkss].value, $m.approx, $1, $1, $m.parameters, $m.keys, $α_initial_ref, $m[:labss].value, $m.exogenous_shocks, $m.endogenous_states)

#=
suite["decr_euler"] = @benchmarkable decr_euler($m[:rkss].value, $m.approx.ninter, $m.approx.nexogshock, $m.approx.nfunc, $m.approx.nexog, $m.approx.nvars, $m.approx.nexogcont, $m.approx.nmsv, $m.approx.xgrid, $m.approx.slopeconxx, $m.approx.exoggrid, $1, $1, $m.approx.ngrid, $m.approx.nshockgrid, $m.approx.bbt, $m.approx.statezlbinfo, $m.approx.zlbswitch, $m.approx.nquad, $m.approx.ghweights, $m.approx.ghnodes, $m.approx.shockbounds, $m.approx.shockdistance, $m.approx.interpolatemat, $m.approx.slopeconmsv, $m.approx.nindplus, $m.approx.indplus, $m.approx.ns, $m.parameters, $m.keys, $α_initial_ref, $m[:labss].value, $m.exogenous_shocks, $m.endogenous_states)
=#

### Test intermediatedec
h5 = h5open("$path/intermediatedec.h5")
endogvarm1_ref = read(h5, "endogvarm1")
currentshocks_ref = read(h5, "currentshocks")
polyvar_ref = read(h5, "polyvar")
endogvar_ref = read(h5, "endogvar")
close(h5)

endogvar = Array{Float64}(undef, m.approx.nvars+m.approx.nexog)

suite["intermediatedec"] = @benchmarkable intermediatedec!($endogvar, $m.approx.nvars, $m.approx.nexog, $m[:labss].value, $endogvarm1_ref, $currentshocks_ref, $polyvar_ref, $1.0, $polyvar_ref, $false, $m.exogenous_shocks, $m.parameters, $m.keys)

### Test decr
h5 = h5open("$path/decr.h5")
innovations_ref = read(h5, "innovations")
close(h5)

endogvarp = Array{Float64}(undef, m.approx.nvars+m.approx.nexog)
#=
suite["decr"] = @benchmarkable decr!($endogvarp, $m.approx.nvars, $m.approx.nexog, $m.approx.nexogcont, $m.approx.nexogshock, $m.approx.nfunc, $m.approx.ninter, $m.approx.nmsv, $m.approx.ngrid, $m.approx.nshockgrid, $m.approx.shockbounds, $m.approx.shockdistance, $m.approx.interpolatemat, $m.approx.slopeconmsv, $m.approx.nindplus, $m.approx.indplus, $m.approx.exoggrid, $m.approx.ns, $m.approx.zlbswitch, $endogvar_ref, $innovations_ref, $m.parameters, $m.keys, $m[:labss].value, $α_initial_ref, $m.exogenous_shocks, $m.endogenous_states)
=#

suite["decr"] = @benchmarkable decr!($endogvarp, $m.approx, $endogvar_ref, $innovations_ref, $m.parameters, $m.keys, $m[:labss].value, $α_initial_ref, $m.exogenous_shocks, $m.endogenous_states)

### Test conversion between msv and xx domains
h5 = h5open("$path/msv2xx.h5")
lmsv_ref = read(h5, "lmsv")
close(h5)

suite["msv2xx"] = @benchmarkable msv2xx($lmsv_ref, $7, $slopeconmsv_ref)


### Test exogposition
suite["exogpos"] = @benchmarkable exogposition($ones(Int64, 6), $[7,2,2,2,2,1], $6)

### Test decrlin
h5 = h5open("$path/decrlin.h5")
endogvarm1_ref = read(h5, "endogvarm1")
innovations_ref = read(h5, "innovations")
sigma_ref = read(h5, "sigma")
pp_ref = read(h5, "pp")
close(h5)
steady_states = [i.value for i in m.steady_state]

suite["decrlin"] = @benchmarkable decrlin($endogvarm1_ref, $innovations_ref, $m.approx.nvars, $m.approx.nexog, $sigma_ref, $pp_ref, $steady_states)

### Test finite_grid
#suite["finite_grid"] = @benchmarkable finite_grid($7, $m[:ρ_η].value, $m[:σ_η].scaledvalue)

### Run and save benchmarks
tune!(suite)
results = run(suite, verbose = true)
save("results.jld", "results", results)

### Compare to previous
io = IOContext(stdout, :compact => false)
for trial in keys(results)
    println("$trial:")
    show(io, judge(BenchmarkTools.median(results[trial]), BenchmarkTools.median(results_old[trial]), time_tolerance = 0.06))
    println("")
end

nothing
