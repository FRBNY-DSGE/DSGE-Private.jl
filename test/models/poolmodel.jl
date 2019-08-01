using DSGEModels, CSV

###########################################################################
# Set up for testing PoolModel instantiation
###########################################################################
# filepath = pwd()
filepath = dirname(@__FILE__)

@testset "Check constructors" begin
    pm = PoolModel()
    # pm <= Setting(:saveroot, "$(filepath)/../reference/")
    pm <= Setting(:dataroot, "$(filepath)/../reference/")
    @test typeof(pm) == PoolModel{Float64}
    mstatic = PoolModel(static = true)
    @test mstatic[:ρ].value == 1 && mstatic[:ρ].fixed
end

# Check solve and statespace functions apply to PoolModel
@testset "Check solve and statespace functions apply to PoolModel" begin
    Φ1, Ψ2 = compute_system(pm)
    Φ2, Ψ2 = solve(pm)
    @test Φ1 == Φ2
    @test Ψ1 == Ψ2
end

nothing
