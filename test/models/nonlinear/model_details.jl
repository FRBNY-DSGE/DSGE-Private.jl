using DSGE
using HDF5, Test

path =  dirname(@__FILE__)

### Model
m = GHLS()
h5 = h5open("$path/params.h5")
params = read(h5, "params")
update!(m, params)
close(h5)
m.approx.nshockgrid = [7,2,2,2,2,1]

### Test shock details against reference output
exoggrid, shockbounds, shockdistance = get_shockdetails(m.approx.nexogcont, m.approx.number_shock_values,m.approx.nshockgrid, m.approx.exogvarinfo, m.approx.nexogshock, m.approx.ns, m.approx.nexog, m.parameters, m.keys)

h5 = h5open("$path/shockdetails.h5")
exoggrid_ref = read(h5, "exoggrid")
shockbounds_ref = read(h5, "shockbounds")
shockdistance_ref = read(h5, "shockdistance")
close(h5)

@testset "All tests" begin

@testset "Compare shock details to reference output" begin
    @test exoggrid_ref ≈ exoggrid
    @test shockbounds_ref ≈ shockbounds
    @test shockdistance_ref ≈ shockdistance
end

### Test decr_euler output against reference output
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

updated_approx_polynomials, err = decr_euler(m[:rkss].value, m.approx.ninter, m.approx.nexogshock, m.approx.nfunc, m.approx.nexog, m.approx.nvars, m.approx.nexogcont, m.approx.nmsv, m.approx.xgrid, m.approx.slopeconxx, m.approx.exoggrid, 1, 1, m.approx.ngrid, m.approx.nshockgrid, m.approx.bbt, m.approx.statezlbinfo, m.approx.zlbswitch, m.approx.nquad, m.approx.ghweights, m.approx.ghnodes, m.approx.shockbounds, m.approx.shockdistance, m.approx.interpolatemat, m.approx.slopeconmsv, m.approx.nindplus, m.approx.indplus, m.approx.ns, m.parameters, m.keys, α_initial_ref, m[:labss].value, m.exogenous_shocks, m.endogenous_states)

h5 = h5open("$path/decr_euler.h5")
updated_approx_polynomials_ref = read(h5, "updated_approx_polynomials")
close(h5)

@testset "Compare updated polynomials to reference output" begin
    @test updated_approx_polynomials_ref ≈ updated_approx_polynomials
end

### Test intermediatedec output against reference output
h5 = h5open("$path/intermediatedec.h5")
endogvarm1_ref = read(h5, "endogvarm1")
currentshocks_ref = read(h5, "currentshocks")
polyvar_ref = read(h5, "polyvar")
endogvar_ref = read(h5, "endogvar")
close(h5)

endogvar = Array{Float64}(undef, m.approx.nvars+m.approx.nexog)

intermediatedec!(endogvar, m.approx.nvars, m.approx.nexog, m[:labss].value, endogvarm1_ref, currentshocks_ref, polyvar_ref, 1.0, polyvar_ref, false, m.exogenous_shocks, m.parameters, m.keys)

@testset "Compare intermediate endogenous variables to reference output" begin
    @test endogvar_ref ≈ endogvar
end

### Test decr output against reference output
h5 = h5open("$path/decr.h5")
innovations_ref = read(h5, "innovations")
endogvarp_ref = read(h5, "endogvarp")
close(h5)

endogvarp = Array{Float64}(undef, m.approx.nvars+m.approx.nexog)

decr!(endogvarp, m.approx.nvars, m.approx.nexog, m.approx.nexogcont, m.approx.nexogshock, m.approx.nfunc, m.approx.ninter, m.approx.nmsv, m.approx.ngrid, m.approx.nshockgrid, m.approx.shockbounds, m.approx.shockdistance, m.approx.interpolatemat, m.approx.slopeconmsv, m.approx.nindplus, m.approx.indplus, m.approx.exoggrid, m.approx.ns, m.approx.zlbswitch, endogvar_ref, innovations_ref, m.parameters, m.keys, m[:labss].value, α_initial_ref, m.exogenous_shocks, m.endogenous_states)

@testset "Compare decision rule output to reference output" begin
    @test endogvarp_ref ≈ endogvarp
end

### Test conversion between msv and xx domains against reference output
h5 = h5open("$path/msv2xx.h5")
lmsv_ref = read(h5, "lmsv")
xx_ref = read(h5, "xx")
close(h5)

xx = msv2xx(lmsv_ref, 7, slopeconmsv_ref)

@testset "Compare domain conversion to reference output" begin
    @test xx_ref ≈ xx
end

### Test exogposition against reference output
@testset "Compare exogposition to reference output" begin
    @test 1 == exogposition(ones(Int64, 6), [7,2,2,2,2,1], 6)
end

### Test decrlin against reference output
h5 = h5open("$path/decrlin.h5")
endogvarm1_ref = read(h5, "endogvarm1")
innovations_ref = read(h5, "innovations")
sigma_ref = read(h5, "sigma")
pp_ref = read(h5, "pp")
endogvarlin_ref = read(h5, "endogvarlin")
close(h5)
steady_states = [i.value for i in m.steady_state]

endogvarlin = decrlin(endogvarm1_ref, innovations_ref, m.approx.nvars, m.approx.nexog, sigma_ref, pp_ref, steady_states)

@testset "Compare linearized decision rule to reference output" begin
    @test endogvarlin_ref ≈ endogvarlin
end


### Test finite_grid against reference output
shockbounds = zeros(5,2)
shockdistance = zeros(5)
shockvalues = zeros(15)

@views finite_grid!(shockvalues[1:7], shockdistance[1:1], shockbounds[1, :], 7, m[:ρ_η].value, m[:σ_η].scaledvalue)

h5 = h5open("$path/finite_grid.h5")
shockbounds_ref = read(h5, "shockbounds")
shockdistance_ref = read(h5, "shockdistance")
shockvalues_ref = read(h5, "shockvalues")
close(h5)

@testset "Compare grid to reference output" begin
    @test shockbounds_ref ≈ shockbounds
    @test shockdistance_ref ≈ shockdistance
    @test shockvalues_ref ≈ shockvalues
end

end
