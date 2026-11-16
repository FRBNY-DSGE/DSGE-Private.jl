module TestSMCExtraKwargs

using DSGE, ModelConstructors, SMC, HDF5, JLD2, Test

const data = h5read(joinpath(@__DIR__, "reference", "smc.h5"), "data")

function model(name)
    m = AnSchorfheide()
    m <= Setting(:saveroot, joinpath(mktempdir(), name))
    m <= Setting(:n_particles, 40)
    m <= Setting(:n_Φ, 20)
    m <= Setting(:use_fixed_schedule, true)
    m <= Setting(:use_parallel_workers, false)
    m <= Setting(:smc_seed, 1113)
    return m
end

@testset "smc_extra_kwargs reaches SMC.smc" begin
    m = model("esjd")
    m <= Setting(:smc_extra_kwargs, (scale_adaptation = :esjd,
                                     esjd_pilot_particles = 4,
                                     esjd_multipliers = [0.5, 1.0, 2.0]))
    smc2(m, data; verbose = :none, run_csminwel = false)
    health = JLD2.load(rawpath(m, "estimate", "smc_cloud.jld2"), "stage_health")
    @test !isempty(health["esjd_pilot_scores"])

    # A Dict is accepted too. The downstream SMC validation proves the option was received.
    bad = model("bad")
    bad <= Setting(:smc_extra_kwargs, Dict(:scale_adaptation => :invalid))
    @test_throws ArgumentError smc2(bad, data; verbose = :none, run_csminwel = false)
end

end # module
