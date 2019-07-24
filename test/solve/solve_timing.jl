using DSGE
using BenchmarkTools, HDF5, JLD

path = dirname(@__FILE__)

#Initialize model
m = GHLS()
h5 = h5open("$path/params.h5")
params = read(h5, "params")
close(h5)
update!(m, params)
m.approx.nshockgrid = [7 2 2 2 2 1]

### Set up benchmarking test suite
suite = BenchmarkGroup()

### Add results of linear model simulation
h5 = h5open("$path/shockdetails.h5")
exoggrid_ref = read(h5, "exoggrid")
shockbounds_ref = read(h5, "shockbounds")
shockdistance_ref = read(h5, "shockdistance")
close(h5)

m.approx.exoggrid, m.approx.shockbounds, m.approx.shockdistance = exoggrid_ref, shockbounds_ref, shockdistance_ref

h5 = h5open("$path/simulatelinear.h5")
endog_emean_ref = read(h5, "endog_emean")
zlbfrequency_ref = read(h5, "zlbfrequency")
msvbounds_ref = read(h5, "msvbounds")
statezlbinfo_ref = read(h5, "statezlbinfo")
close(h5)

### Test slopes
m.approx.endog_emean, m.approx.zlbfrequency, m.approx.msvbounds, m.approx.statezlbinfo, m.approx.convergence = endog_emean_ref, zlbfrequency_ref, msvbounds_ref, statezlbinfo_ref, true

suite["slopes"] = @benchmarkable create_slopes($m)

h5 = h5open("$path/slopes.h5")
slopeconmsv_ref = read(h5, "slopeconmsv")
slopeconxx_ref = read(h5, "slopeconxx")
close(h5)


## Test lindecrule_markov
suite["markov"] = @benchmarkable lindecrule_markov($m)

h5 = h5open("$path/lindecrule_markov.h5")
aalin_ref = read(h5, "aalin")
bblin_ref = read(h5, "bblin")
close(h5)

### Test initial alphas
m.approx.slopeconmsv, m.approx.slopeconxx = slopeconmsv_ref, slopeconxx_ref
suite["initial"] = @benchmarkable initial_α($m)

h5 = h5open("$path/initialalphas.h5")
α_initial_ref = read(h5, "initialalphas")
close(h5)

### Tune before adding long functions
tune!(suite)

### Test linear model simulation
suite["linear"] = @benchmarkable simulate_linear($m)

### Test fixed point output against reference output
suite["fixedpoint"] = @benchmarkable fixedpoint($m, $α_initial_ref) samples = 1 evals = 1

### Run benchmarks
results = run(suite, verbose = true)

### Save benchmarks
save("results.jld", "results", results)
