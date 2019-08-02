using HDF5
using Test
using DSGE

path = dirname(@__FILE__)

### Test linear solver
file = "$path/../reference/solve.h5"
TTT_expected = h5read(file, "TTT")
CCC_expected = h5read(file, "CCC")
RRR_expected = h5read(file, "RRR")

m = AnSchorfheide()
TTT, RRR, CCC = solve(m)

@testset "Check state-space system matches reference" begin
    @test @test_matrix_approx_eq TTT_expected TTT
    @test @test_matrix_approx_eq RRR_expected RRR
    @test @test_matrix_approx_eq CCC_expected CCC
end

### Model to test nonlinear solve
m = GHLS()
h5 = h5open("$path/params.h5")
model_params = read(h5, "params")
close(h5)
update!(m, model_params)
m.approx.nshockgrid = [7,2,2,2,2,1]

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

### Test linear model simulation against reference output
h5 = h5open("$path/shockdetails.h5")
exoggrid_ref = read(h5, "exoggrid")
shockbounds_ref = read(h5, "shockbounds")
shockdistance_ref = read(h5, "shockdistance")
close(h5)
m.approx.exoggrid, m.approx.shockbounds, m.approx.shockdistance = exoggrid_ref, shockbounds_ref, shockdistance_ref

endog_emean, zlbfrequency, msvbounds, statezlbinfo, convergence = simulate_linear(m.approx.ns, m.approx.nvars, m.approx.nexog, m.approx.nmsv, m.approx.nexogcont, m.approx.nexogshock, steady_states, m.approx.nshockgrid, m.approx.shockbounds, m.approx.shockdistance, pp, sigma)

h5 = h5open("$path/simulatelinear.h5")
endog_emean_ref = read(h5, "endog_emean")
zlbfrequency_ref = read(h5, "zlbfrequency")
msvbounds_ref = read(h5, "msvbounds")
statezlbinfo_ref = read(h5, "statezlbinfo")
close(h5)

@testset "Compare linear simulation to reference output" begin
    @test endog_emean_ref ≈ endog_emean
    @test zlbfrequency_ref ≈ zlbfrequency
    @test msvbounds_ref ≈ msvbounds
    @test statezlbinfo_ref ≈ statezlbinfo
    @test convergence == true
end

### Test slopes against reference output
m.approx.endog_emean, m.approx.zlbfrequency, m.approx.msvbounds, m.approx.statezlbinfo, m.approx.convergence = endog_emean_ref, zlbfrequency_ref, msvbounds_ref, statezlbinfo_ref, true

slopeconmsv, slopeconxx = create_slopes(m.approx.nmsv, m.approx.nexogcont, m.approx.msvbounds)

h5 = h5open("$path/slopes.h5")
slopeconmsv_ref = read(h5, "slopeconmsv")
slopeconxx_ref = read(h5, "slopeconxx")
close(h5)

@testset "Compare slope conversions to reference output" begin
    @test slopeconmsv_ref ≈ slopeconmsv
    @test slopeconxx_ref ≈ slopeconxx
end

## Test lindecrule_markov against reference output
aalin, bblin = lindecrule_markov(pp, sigma, m.approx.nvars, m.approx.nexog, m.approx.nexogcont, m.approx.nexogshock)

h5 = h5open("$path/lindecrule_markov.h5")
aalin_ref = read(h5, "aalin")
bblin_ref = read(h5, "bblin")
close(h5)

@testset "Compare linear matrices to reference output" begin
    @test aalin_ref ≈ aalin
    @test bblin_ref ≈ bblin
end

### Test initial alphas against reference output
m.approx.slopeconmsv, m.approx.slopeconxx = slopeconmsv_ref, slopeconxx_ref
α_initial = initial_α(m.approx.nvars, m.approx.nexog, m.approx.nexogshock, m.approx.nmsv,m.approx.nexogcont,m.approx.ns,m.approx.ngrid, m.approx.exoggrid, steady_states, m.approx.slopeconxx, m.approx.xgrid, m.approx.nfunc, m.approx.bbtinv, aalin, bblin)

h5 = h5open("$path/initialalphas.h5")
α_initial_ref = read(h5, "initialalphas")
close(h5)

@testset "Compare initial alphas to reference output" begin
    @test α_initial_ref ≈ α_initial
end

h5 = h5open("$path/alphastar.h5")
α_star_ref = read(h5, "alphastar")
close(h5)

### Test fixed point output against reference output
α_star, convergence = fixedpoint(m[:rkss].value, m.approx.ninter, m.approx.nexogshock, m.approx.nfunc, m.approx.nexog, m.approx.nvars, m.approx.nexogcont, m.approx.nmsv, m.approx.xgrid, m.approx.slopeconxx, m.approx.exoggrid, m.approx.ngrid, m.approx.nshockgrid, m.approx.bbt, m.approx.statezlbinfo, m.approx.zlbswitch, m.approx.nquad, m.approx.ghweights, m.approx.ghnodes, m.approx.shockbounds, m.approx.shockdistance, m.approx.interpolatemat, m.approx.slopeconmsv, m.approx.nindplus, m.approx.indplus, m.approx.ns, m.parameters, m.keys, m[:labss].value, m.exogenous_shocks, m.endogenous_states, m.approx.bbtinv, α_initial_ref)

@testset "Compare alpha star to reference output" begin
    @test α_star_ref ≈ α_star
end

### Test fixedpoint_parallel output against reference output
α_star_parallel, convergence = fixedpoint_parallel(m[:rkss].value, m.approx.ninter, m.approx.nexogshock, m.approx.nfunc, m.approx.nexog, m.approx.nvars, m.approx.nexogcont, m.approx.nmsv, m.approx.xgrid, m.approx.slopeconxx, m.approx.exoggrid, m.approx.ngrid, m.approx.nshockgrid, m.approx.bbt, m.approx.statezlbinfo, m.approx.zlbswitch, m.approx.nquad, m.approx.ghweights, m.approx.ghnodes, m.approx.shockbounds, m.approx.shockdistance, m.approx.interpolatemat, m.approx.slopeconmsv, m.approx.nindplus, m.approx.indplus, m.approx.ns, m.parameters, m.keys, m[:labss].value, m.exogenous_shocks, m.endogenous_states, m.approx.bbtinv, α_initial_ref)

@testset "fixedpoint_parallel" begin
    @test α_star_ref ≈ α_star_parallel
end

nothing
