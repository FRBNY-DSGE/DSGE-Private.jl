using HDF5
using Test
using DSGE

path = dirname(@__FILE__)

### Model to test nonlinear solve
m = GHLS()
#h5 = h5open("$path/params.h5")
#model_params = read(h5, "params")
#close(h5)
#update!(m, model_params)
#m.approx.nshockgrid = [7,3,3,3,3,1]

suite = BenchmarkGroup()

# Get_shockdetails
m.approx.exoggrid, m.approx.shockbounds, m.approx.shockdistance = get_shockdetails(m.approx.nexogcont, m.approx.number_shock_values, m.approx.nshockgrid, m.approx.exogvarinfo, m.approx.nexogshock, m.approx.ns, m.approx.nexog,m.parameters,m.keys)

# Getting pp and sigma for tests
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

steady_states = [i.value for i in m.steady_state]

m.approx.endog_emean, m.approx.zlbfrequency, m.approx.msvbounds, m.approx.statezlbinfo, m.approx.convergence = simulate_linear(m.approx.ns, m.approx.nvars, m.approx.nexog, m.approx.nmsv, m.approx.nexogcont, m.approx.nexogshock, steady_states, m.approx.nshockgrid, m.approx.shockbounds, m.approx.shockdistance, pp, sigma)

m.approx.slopeconmsv, m.approx.slopeconxx = create_slopes(m.approx.nmsv, m.approx.nexogcont, m.approx.msvbounds)


## Test lindecrule_markov against reference output
aalin, bblin = lindecrule_markov(pp, sigma, m.approx.nvars, m.approx.nexog, m.approx.nexogcont, m.approx.nexogshock)

### Test initial alphas against reference output
α_initial = initial_α(m.approx.nvars, m.approx.nexog, m.approx.nexogshock, m.approx.nmsv,m.approx.nexogcont,m.approx.ns,m.approx.ngrid, m.approx.exoggrid, steady_states, m.approx.slopeconxx, m.approx.xgrid, m.approx.nfunc, m.approx.bbtinv, aalin, bblin)


### Test fixed point output against reference output

suite["fixedpoint"] = @benchmarkable fixedpoint($m[:rkss].value, $m.approx, $m.parameters, $m.keys, $m[:labss].value, $m.exogenous_shocks, $m.endogenous_states, $α_initial)

#=
suite["fixedpoint"] = @benchmarkable fixedpoint($m[:rkss].value, $m.approx.ninter, $m.approx.nexogshock, $m.approx.nfunc, $m.approx.nexog, $m.approx.nvars, $m.approx.nexogcont, $m.approx.nmsv, $m.approx.xgrid, $m.approx.slopeconxx, $m.approx.exoggrid, $m.approx.ngrid, $m.approx.nshockgrid, $m.approx.bbt, $m.approx.statezlbinfo, $m.approx.zlbswitch, $m.approx.nquad, $m.approx.ghweights, $m.approx.ghnodes, $m.approx.shockbounds, $m.approx.shockdistance, $m.approx.interpolatemat, $m.approx.slopeconmsv, $m.approx.nindplus, $m.approx.indplus, $m.approx.ns, $m.parameters, $m.keys, $m[:labss].value, $m.exogenous_shocks, $m.endogenous_states, $m.approx.bbtinv, $α_initial) samples = 3 evals = 3
=#
### Test fixedpoint_parallel output against reference output
#=
α_star_parallel, convergence = fixedpoint_parallel(m[:rkss].value, m.approx.ninter, m.approx.nexogshock, m.approx.nfunc, m.approx.nexog, m.approx.nvars, m.approx.nexogcont, m.approx.nmsv, m.approx.xgrid, m.approx.slopeconxx, m.approx.exoggrid, m.approx.ngrid, m.approx.nshockgrid, m.approx.bbt, m.approx.statezlbinfo, m.approx.zlbswitch, m.approx.nquad, m.approx.ghweights, m.approx.ghnodes, m.approx.shockbounds, m.approx.shockdistance, m.approx.interpolatemat, m.approx.slopeconmsv, m.approx.nindplus, m.approx.indplus, m.approx.ns, m.parameters, m.keys, m[:labss].value, m.exogenous_shocks, m.endogenous_states, m.approx.bbtinv, α_initial_ref)
=#

results = run(suite, verbose=true)
@show results

nothing
