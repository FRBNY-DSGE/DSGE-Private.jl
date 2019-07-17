using DSGE
using HDF5, Test

path = dirname(@__FILE__)

### Model
m = GHLS()
h5 = h5open("$path/params.h5")
close(h5)
params = read(h5, "params")
update!(m, params)

### Parameters

@testset "Test parameter bounds checking" begin
    for θ in m.parameters
        !isa(θ,AbstractParameter) && error()
        θ.fixed && continue

        (left, right) = θ.valuebounds
        @test left < θ.value < right
    end
end

### Model indices

@testset "Checking model indices" begin
    # Endogenous states
    endo = m.endogenous_states
    @test length(endo) == 42
    @test endo[:Eλ_c] == 42

    # Exogenous shocks
    exo = m.exogenous_shocks
    @test length(exo) == 8
    @test exo[:unk_sh] == 8

    # Expectation shocks
    ex = m.expected_shocks
    @test length(ex) == 8
    @test ex[:Eλc_sh] == 8

    # Equations
    eq = m.equilibrium_conditions
    @test length(eq) == 42
    @test eq[:eq_Eλc] == 42

    # Additional states
    endo_new = m.endogenous_states_augmented
    @test length(endo_new) == 7
    @test endo_new[:y_t1] == 43

    # Observables
    obs = m.observables
    @test length(obs) == 7
    @test obs[:obs_investment] == 7
end

### Linear equilibrium conditions
Γ0, Γ1, C, Ψ, Π = eqcond(m)

@testset "Check eqcond and state-space dimensions" begin
    # Matrices are of expected dimensions
    @test size(Γ0) == (42, 42)
    @test size(Γ1) == (42, 42)
    @test size(C) == (42,)
    @test size(Ψ) == (42, 8)
    @test size(Π) == (42, 8)
end

# Check output matrices against reference output (ϵ = 1e-4)
h5 = h5open("$path/eqcond.h5")
Γ0_ref = read(h5, "G0")
Γ1_ref = read(h5, "G1")
C_ref  = reshape(read(h5, "C"), 42, 1)
Ψ_ref  = read(h5, "PSI")
Π_ref  = read(h5, "PIE")
close(h5)

@testset "Checking eqconds against reference output" begin
    @test Γ0_ref ≈ Γ0
    @test Γ1_ref ≈ Γ1
    @test C_ref ≈ C
    @test Ψ_ref ≈ Ψ
    @test Π_ref ≈ Π
end

### Measurement equation

### Custom settings
