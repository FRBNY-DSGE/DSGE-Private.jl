using DSGE
using HDF5, Test

path =  dirname(@__FILE__)

### Model
m = GHLS()
h5 = h5open("$path/params.h5")
params = read(h5, "params")
update!(m, params)
close(h5)

### Testing requires smaller shock grid
if (m.approx.nshockgrid != [7,2,2,2,2,1])
    println("Testing requires nshockgrid = [7,2,2,2,2,1]")
end

### Test without zlb
# SET SETTING HERE

### Test shock grid against reference output
exoggrid, shockbounds, shockdistance = gen_shockgrid(m.approx.nshockgrid, m.approx.nexogshocks, m.approx.ns, m.approx.nexogvars, m.parameters, m.keys)

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

m.approx.endog_emean, m.approx.zlbfrequency, m.approx.msvbounds, m.approx.statezlbinfo, convergence = endog_emean_ref, zlbfrequency_ref, msvbounds_ref, statezlbinfo_ref, true

h5 = h5open("$path/slopes.h5")
slopeconmsv_ref = read(h5, "slopeconmsv")
slopeconxx_ref = read(h5, "slopeconxx")
close(h5)

m.approx.slopeconmsv, m.approx.slopeconxx = slopeconmsv_ref, slopeconxx_ref

h5 = h5open("$path/initialalphas.h5")
α_initial_ref = read(h5, "initialalphas")
close(h5)

updated_approx_functions, err = decr_euler(m[:rkss].value, m.approx, 1, 1, m.parameters, m.keys, α_initial_ref, m[:labss].value, m.exogenous_shocks, m.endogenous_states, get_setting(m, :zero_lower_bound))

h5 = h5open("$path/decr_euler.h5")
updated_approx_functions_ref = read(h5, "updated_approx_polynomials") #updated_approx_polynomials was previous name for this variable
close(h5)

@testset "Compare updated functions to reference output" begin
    @test updated_approx_functions_ref ≈ updated_approx_functions
end

### Test intermediatedec output against reference output
h5 = h5open("$path/intermediatedec.h5")
endogvarm1_ref = read(h5, "endogvarm1")
currentshocks_ref = read(h5, "currentshocks")
funcvals_ref = read(h5, "polyvar") #polyvar was previous name for this variable
endogvar_ref = read(h5, "endogvar")
close(h5)

endogvar = Array{Float64}(undef, m.approx.nendogvars+m.approx.nexogvars)

intermediatedec!(endogvar, m.approx.nendogvars, m.approx.nexogvars, m[:labss].value, endogvarm1_ref, currentshocks_ref, funcvals_ref, 1.0, funcvals_ref, false, m.endogenous_states, m.exogenous_shocks, m.parameters, m.keys)

@testset "Compare intermediate endogenous variables to reference output" begin
    @test endogvar_ref ≈ endogvar
end

### Test decr output against reference output
h5 = h5open("$path/decr.h5")
innovations_ref = read(h5, "innovations")
endogvarp_ref = read(h5, "endogvarp")
close(h5)

endogvarp = Array{Float64}(undef, m.approx.nendogvars+m.approx.nexogvars)

decr!(endogvarp, m.approx, endogvar_ref, innovations_ref, m.parameters, m.keys, m[:labss].value, α_initial_ref, m.exogenous_shocks, m.endogenous_states, get_setting(m, :zero_lower_bound))

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

### Test exogstate against reference output
@testset "Compare exogstate to reference output" begin
    @test 1 == exogstate(ones(Int64, 6), [7,2,2,2,2,1], 6)
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

endogvarlin = decrlin(endogvarm1_ref, innovations_ref, m.approx.nendogvars, m.approx.nexogvars, sigma_ref, pp_ref, steady_states)

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
