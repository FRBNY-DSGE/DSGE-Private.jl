using DSGE, ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test

# The setting smc_initial_c_blocks reaches SMC.smc's initial_c_blocks through smc2.
# SMC.jl's own tests cover the keyword; this covers the DSGE passthrough.

path = dirname(@__FILE__)
icb_data = h5read(joinpath(path, "reference/smc.h5"), "data")
icb_dir  = mktempdir()

function icb_model(name)
    m = AnSchorfheide()
    m <= Setting(:saveroot, joinpath(icb_dir, name))
    m <= Setting(:n_particles, 100)
    m <= Setting(:n_Φ, 20)
    m <= Setting(:λ, 2.0)
    m <= Setting(:n_smc_blocks, 1)
    m <= Setting(:use_parallel_workers, false)
    m <= Setting(:step_size_smc, 0.5)
    m <= Setting(:n_mh_steps_smc, 1)
    m <= Setting(:resampler_smc, :systematic)
    m <= Setting(:target_accept, 0.25)
    m <= Setting(:mixture_proportion, 0.9)
    m <= Setting(:use_fixed_schedule, true)
    m <= Setting(:resampling_threshold, 0.5)
    m <= Setting(:smc_iteration, 0)
    m <= Setting(:use_chand_recursion, true)
    m <= Setting(:smc_seed, 3)
    m <= Setting(:smc_param_blocks, [[:ρ_z], [:σ_R, :σ_g]])
    m <= Setting(:smc_blockwise_c, true)
    return m
end
function icb_run(m)
    smc2(m, icb_data; verbose = :none, run_csminwel = false)
    return load(rawpath(m, "estimate", "smc_cloud.jld2"))
end

@testset "smc_initial_c_blocks reaches SMC.smc" begin
    m = icb_model("default")
    @test get_setting(m, :smc_initial_c_blocks) === nothing
    base = icb_run(m)
    h0 = base["stage_health"]
    @test h0["block_labels"] == ["ρ_z", "block2", "rest"]
    @test all(h0["c_blocks"][:, 1] .== SMC.update_c(0.5, 0.25, 0.25))

    c0 = Dict("ρ_z" => 1.3, "block2" => 0.2, "rest" => 0.7)
    m1 = icb_model("dict")
    m1 <= Setting(:smc_initial_c_blocks, c0)                 # a Dict replaces the default nothing
    r1 = icb_run(m1)
    @test r1["stage_health"]["c_blocks"][:, 1] == SMC.update_c.([1.3, 0.2, 0.7], 0.25, 0.25)
    @test r1["cloud"].particles != base["cloud"].particles

    # The vector form (group order) is the Dict form, bitwise
    m2 = icb_model("vector")
    m2 <= Setting(:smc_initial_c_blocks, [1.3, 0.2, 0.7])
    r2 = icb_run(m2)
    @test r2["cloud"].particles == r1["cloud"].particles
    @test r2["stage_health"]["c_blocks"] == r1["stage_health"]["c_blocks"]

    # Setting it back to nothing is the default run, bitwise
    m3 = icb_model("reset")
    m3 <= Setting(:smc_initial_c_blocks, c0)
    m3 <= Setting(:smc_initial_c_blocks, nothing)
    r3 = icb_run(m3)
    @test r3["cloud"].particles == base["cloud"].particles
    @test r3["w"] == base["w"] && r3["W"] == base["W"]
    @test r3["stage_health"]["c_blocks"] == h0["c_blocks"]

    # SMC's input checks apply
    m4 = icb_model("bad")
    m4 <= Setting(:smc_initial_c_blocks, Dict("ρ_z" => 1.3, "rest" => 0.7))
    @test_throws ArgumentError icb_run(m4)
    m4 <= Setting(:smc_initial_c_blocks, [1.3, 0.2, 0.7])
    m4 <= Setting(:smc_blockwise_c, false)
    @test_throws ArgumentError icb_run(m4)
end

rm(icb_dir; recursive = true, force = true)
