module TestSMCExtraKwargs

using DSGE, ModelConstructors, HDF5, Test

const data = h5read(joinpath(@__DIR__, "reference", "smc.h5"), "data")

function model(name)
    m = AnSchorfheide()
    m <= Setting(:saveroot, joinpath(mktempdir(), name))
    m <= Setting(:n_particles, 40)
    m <= Setting(:use_parallel_workers, false)
    m <= Setting(:smc_seed, 1113)
    return m
end

@testset "smc_extra_kwargs reaches SMC.smc" begin
    # SMC requires n_parts::Int. A type error proves that the extra value reached
    # SMC and overrode the valid :n_particles setting without sampling noise.
    named = model("named")
    named <= Setting(:smc_extra_kwargs, (n_parts = "invalid",))
    @test_throws TypeError smc2(named, data; verbose = :none, run_csminwel = false)

    dict = model("dict")
    dict <= Setting(:smc_extra_kwargs, Dict(:n_parts => "invalid"))
    @test_throws TypeError smc2(dict, data; verbose = :none, run_csminwel = false)
end

end # module
