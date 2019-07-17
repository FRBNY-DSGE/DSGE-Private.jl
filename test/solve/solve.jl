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
params = read(h5, "params")
close(h5)
update!(m, params)
m.approx.nshockgrid = [7 2 2 2 2 1]

### Test linear model simulation against reference output
h5 = h5open("$path/shockdetails.h5")
exoggrid_ref = read(h5, "exoggrid")
shockbounds_ref = read(h5, "shockbounds")
shockdistance_ref = read(h5, "shockdistance")
close(h5)
m.approx.exoggrid, m.approx.shockbounds, m.approx.shockdistance = exoggrid_ref, shockbounds_ref, shockdistance_ref

endog_emean, zlbfrequency, msvbounds, statezlbinfo, convergence = simulate_linear(m)

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

slopeconmsv, slopeconxx = create_slopes(m)

h5 = h5open("$path/slopes.h5")
slopeconmsv_ref = read(h5, "slopeconmsv")
slopeconxx_ref = read(h5, "slopeconxx")
close(h5)

@testset "Compare slope conversions to reference output" begin
    @test slopeconmsv_ref ≈ slopeconmsv
    @test slopeconxx_ref ≈ slopeconxx
end

## Test lindecrule_markov against reference output
aalin, bblin = lindecrule_markov(m)

h5 = h5open("$path/lindecrule_markov.h5")
aalin_ref = read(h5, "aalin")
bblin_ref = read(h5, "bblin")
close(h5)

@testset "Compare linear matrices to reference output" begin
    @test aalin_ref ≈ aalin
    @test bblin ≈ bblin
end

### Test initial alphas against reference output
m.approx.slopeconmsv, m.approx.slopeconxx = slopeconmsv_ref, slopeconxx_ref
α_initial = initial_α(m)

h5 = h5open("$path/initialalphas.h5")
α_initial_ref = read(h5, "initialalphas")
close(h5)

@testset "Compare initial alphas to reference output" begin
    @test α_initial_ref ≈ α_initial
end

### Test fixedpoint_parallel output against reference output
α_star_parallel, convergence = fixedpoint_parallel(m, α_initial_ref)

h5 = h5open("$path/alphastar.h5")
α_star_ref = read(h5, "alphastar")
close(h5)

@testset "fixedpoint_parallel" begin
    @test α_star_ref ≈ α_star_parallel
end

### Test fixed point output against reference output
α_star = fixedpoint(m, α_initial_ref)

@testset "Compare alpha star to reference output" begin
    @test α_star_ref ≈ α_star
end

nothing
