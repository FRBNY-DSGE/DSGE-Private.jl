using DSGE
#addprocs_frbny(5)
@everywhere using DSGE, BenchmarkTools, HDF5, JLD

path = dirname(@__FILE__)

#Initialize model
m = GHLS()
#h5 = h5open("$path/params.h5")
#params = read(h5, "params")
#close(h5)

pars = vec(rand(m.parameters,1))
update!(m, pars)
#update!(m,params)
#m.approx.nshockgrid = [7,2,2,2,2,1]

### Set up benchmarking test suite
suite = BenchmarkGroup()


### Add results of linear model simulation
h5 = h5open("$path/shockdetails.h5")
exoggrid_ref = read(h5, "exoggrid")
shockbounds_ref = read(h5, "shockbounds")
shockdistance_ref = read(h5, "shockdistance")
close(h5)

m.approx.exoggrid, m.approx.shockbounds, m.approx.shockdistance = exoggrid_ref, shockbounds_ref, shockdistance_ref

# Getting pp and sigma by running gensys
Γ0, Γ1, C, Ψ, Π = eqcond(m)
TTT_gensys, CCC_gensys, RRR_gensys, eu = gensys(Γ0, Γ1, C, Ψ, Π, 1+1e-6, verbose = :high)

# Check for LAPACK exception, existence and uniqueness
if eu[1] != 1 || eu[2] != 1
    throw(GensysError())
end

TTT = real(TTT_gensys)
RRR = real(RRR_gensys)

QQQ = zeros(8, 8)
QQQ[1, 1] = m[:σ_η]^2.0
QQQ[2, 2] = m[:σ_μ]^2.0
QQQ[3, 3] = m[:σ_Z]^2.0
QQQ[4, 4] = m[:σ_R]^2.0
QQQ[5, 5] = m[:σ_g]^2.0
QQQ[6, 6] = m[:σ_elast]^2.0
QQQ[7, 7] = m[:σ_elastw]^2.0
QQQ[8, 8] = 0.

RRR = RRR*sqrt.(QQQ)
pp = TTT[1:28, 1:28]
sigma = RRR[1:28, 1:6]


h5 = h5open("$path/simulatelinear.h5")
endog_emean_ref = read(h5, "endog_emean")
zlbfrequency_ref = read(h5, "zlbfrequency")
msvbounds_ref = read(h5, "msvbounds")
statezlbinfo_ref = read(h5, "statezlbinfo")
close(h5)

### Test slopes
m.approx.endog_emean, m.approx.zlbfrequency, m.approx.msvbounds, m.approx.statezlbinfo, m.approx.convergence = endog_emean_ref, zlbfrequency_ref, msvbounds_ref, statezlbinfo_ref, true

suite["slopes"] = @benchmarkable create_slopes($m.approx.nmsv, $m.approx.nexogcont, $m.approx.msvbounds)

h5 = h5open("$path/slopes.h5")
slopeconmsv_ref = read(h5, "slopeconmsv")
slopeconxx_ref = read(h5, "slopeconxx")
close(h5)


## Test lindecrule_markov
suite["markov"] = @benchmarkable lindecrule_markov($pp, $sigma, $m.approx.nvars, $m.approx.nexog, $m.approx.nexogcont, $m.approx.nexogshock)

h5 = h5open("$path/lindecrule_markov.h5")
aalin_ref = read(h5, "aalin")
bblin_ref = read(h5, "bblin")
close(h5)

steady_states = [i.value for i in m.steady_state]

### Test initial alphas
m.approx.slopeconmsv, m.approx.slopeconxx = slopeconmsv_ref, slopeconxx_ref
suite["initial"] = @benchmarkable initial_α($m.approx.nvars, $m.approx.nexog, $m.approx.nexogshock, $m.approx.nmsv, $m.approx.nexogcont, $m.approx.ns, $m.approx.ngrid, $m.approx.exoggrid, steady_states, $m.approx.slopeconxx, $m.approx.xgrid, $m.approx.nfunc, $m.approx.bbtinv, aalin_ref, bblin_ref)

h5 = h5open("$path/initialalphas.h5")
α_initial_ref = read(h5, "initialalphas")
close(h5)

### Tune before adding long functions
tune!(suite)

### Test linear model simulation
suite["linear"] = @benchmarkable simulate_linear($m.approx.ns, $m.approx.nvars, $m.approx.nexog, $m.approx.nmsv, $m.approx.nexogcont, $m.approx.nexogshock, steady_states, $m.approx.nshockgrid, $m.approx.shockbounds, $m.approx.shockdistance, pp, sigma)

@sync @distributed (hcat) for i in 1:4
    @btime fixedpoint($m[:rkss].value, $m.approx.ninter, $m.approx.nexogshock, $m.approx.nfunc, $m.approx.nexog, $m.approx.nvars, $m.approx.nexogcont, $m.approx.nmsv, $m.approx.xgrid, $m.approx.slopeconxx, $m.approx.exoggrid, $m.approx.ngrid, $m.approx.nshockgrid, $m.approx.bbt, $m.approx.statezlbinfo, $m.approx.zlbswitch, $m.approx.nquad, $m.approx.ghweights, $m.approx.ghnodes, $m.approx.shockbounds, $m.approx.shockdistance, $m.approx.interpolatemat, $m.approx.slopeconmsv, $m.approx.nindplus, $m.approx.indplus, $m.approx.ns, $m.parameters, $m.keys, $m[:labss].value, $m.exogenous_shocks, $m.endogenous_states, $m.approx.bbtinv, $α_initial_ref)
end

### Test fixed point output against reference output

suite["fixedpoint"] = @benchmarkable fixedpoint($m[:rkss].value, $m.approx.ninter, $m.approx.nexogshock, $m.approx.nfunc, $m.approx.nexog, $m.approx.nvars, $m.approx.nexogcont, $m.approx.nmsv, $m.approx.xgrid, $m.approx.slopeconxx, $m.approx.exoggrid, $m.approx.ngrid, $m.approx.nshockgrid, $m.approx.bbt, $m.approx.statezlbinfo, $m.approx.zlbswitch, $m.approx.nquad, $m.approx.ghweights, $m.approx.ghnodes, $m.approx.shockbounds, $m.approx.shockdistance, $m.approx.interpolatemat, $m.approx.slopeconmsv, $m.approx.nindplus, $m.approx.indplus, $m.approx.ns, $m.parameters, $m.keys, $m[:labss].value, $m.exogenous_shocks, $m.endogenous_states, $m.approx.bbtinv, $α_initial_ref) samples = 1 evals = 1
#=
suite["fixed_parallel2"] = @benchmarkable fixedpoint_parallel($m[:rkss].value, $m.approx.ninter, $m.approx.nexogshock, $m.approx.nfunc, $m.approx.nexog, $m.approx.nvars, $m.approx.nexogcont, $m.approx.nmsv, $m.approx.xgrid, $m.approx.slopeconxx, $m.approx.exoggrid, $m.approx.ngrid, $m.approx.nshockgrid, $m.approx.bbt, $m.approx.statezlbinfo, $m.approx.zlbswitch, $m.approx.nquad, $m.approx.ghweights, $m.approx.ghnodes, $m.approx.shockbounds, $m.approx.shockdistance, $m.approx.interpolatemat, $m.approx.slopeconmsv, $m.approx.nindplus, $m.approx.indplus, $m.approx.ns, $m.parameters, $m.keys, $m[:labss].value, $m.exogenous_shocks, $m.endogenous_states, $m.approx.bbtinv, $α_initial_ref) samples = 100 evals = 15
=#
### Run benchmarks
results = run(suite, verbose = true)
@show results

### Save benchmarks
save("results8.jld", "results", results)
