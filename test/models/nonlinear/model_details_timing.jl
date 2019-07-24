using DSGE
using HDF5, Test, JLD, BenchmarkTools

path =  dirname(@__FILE__)

### Model
m = GHLS()
h5 = h5open("$path/params.h5")
params = read(h5, "params")
update!(m, params)
close(h5)
m.approx.nshockgrid = [7 2 2 2 2 1]

### Set up benchmarking test suite
suite = BenchmarkGroup()

### Test shock details
suite["shocks"] = @benchmarkable get_shockdetails($m)

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

suite["decr_euler"] = @benchmarkable decr_euler($m, $1, $1, $α_initial_ref)

### Test intermediatedec
h5 = h5open("$path/intermediatedec.h5")
endogvarm1_ref = read(h5, "endogvarm1")
currentshocks_ref = read(h5, "currentshocks")
polyvar_ref = read(h5, "polyvar")
endogvar_ref = read(h5, "endogvar")
close(h5)

suite["intermediatedec"] = @benchmarkable intermediatedec($m, $endogvarm1_ref, $currentshocks_ref, $polyvar_ref, $1.0, $polyvar_ref, $false)

### Test decr
h5 = h5open("$path/decr.h5")
innovations_ref = read(h5, "innovations")
close(h5)

suite["decr"] = @benchmarkable decr($m, $endogvar_ref, $innovations_ref, $α_initial_ref)

### Test conversion between msv and xx domains
h5 = h5open("$path/msv2xx.h5")
lmsv_ref = read(h5, "lmsv")
close(h5)

suite["msv2xx"] = @benchmarkable msv2xx($lmsv_ref, $7, $slopeconmsv_ref)


### Test exogposition
suite["exogpos"] = @benchmarkable exogposition($ones(Int64, 6), $[7 2 2 2 2 1], $6)

### Test decrlin
h5 = h5open("$path/decrlin.h5")
endogvarm1_ref = read(h5, "endogvarm1")
innovations_ref = read(h5, "innovations")
sigma_ref = read(h5, "sigma")
pp_ref = read(h5, "pp")
close(h5)

suite["decrlin"] = @benchmarkable decrlin($endogvarm1_ref, $innovations_ref, $m, $sigma_ref, $pp_ref)

### Test finite_grid
suite["finite_grid"] = @benchmarkable finite_grid($7, $m[:ρ_η].value, $m[:σ_η].scaledvalue)

### Run and save benchmarks
tune!(suite)
results = run(suite, verbose = true)
save("results.jld", "results", results)

nothing
