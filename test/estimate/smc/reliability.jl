using DSGE, ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test

# SMC reliability tools (reliability plan, step 3 part C) on a small An–Schorfheide run

path = dirname(@__FILE__)
rel_data = h5read(joinpath(path, "reference/smc.h5"), "data")
rel_dir = mktempdir()

function rel_model(name; n_particles = 400, n_smc_groups = 1, seed = 1)
    m = AnSchorfheide()
    m <= Setting(:saveroot, joinpath(rel_dir, name))
    m <= Setting(:n_particles, n_particles)
    m <= Setting(:n_Φ, 100)
    m <= Setting(:λ, 2.0)
    m <= Setting(:n_smc_blocks, 1)
    m <= Setting(:use_parallel_workers, false)
    m <= Setting(:step_size_smc, 0.5)
    m <= Setting(:n_mh_steps_smc, 1)
    m <= Setting(:resampler_smc, :systematic)
    m <= Setting(:target_accept, 0.25)
    m <= Setting(:mixture_proportion, 0.9)
    m <= Setting(:use_fixed_schedule, false)
    m <= Setting(:adaptive_tempering_target_smc, 0.95)
    m <= Setting(:resampling_threshold, 0.5)
    m <= Setting(:smc_iteration, 0)
    m <= Setting(:use_chand_recursion, true)
    m <= Setting(:n_smc_groups, n_smc_groups)
    m <= Setting(:smc_bridge_sampling, true)
    m <= Setting(:smc_seed, seed)
    return m
end

@testset "smc2 with particle groups" begin
    m = rel_model("groups"; n_smc_groups = 4)
    cloud = DSGE.smc2(m, rel_data; verbose = :none, run_csminwel = false)
    file = rawpath(m, "estimate", "smc_cloud.jld2")
    @test length(cloud) == 400
    @test load(file, "n_smc_groups") == 4
    @test isfile(replace(file, ".jld2" => "_group=4.jld2"))
    @test isfile(replace(file, ".jld2" => "_group_summary.csv"))

    # The pooled file has no weight history: the MDD and the draws come from the pooled cloud
    est = load(file, "log_mdd_estimates")
    @test marginal_data_density(m, rel_data) == est["smc"]
    @test isfinite(est["bridge_sampling"]) && isfinite(est["mhm"])
    m <= Setting(:sampling_method, :SMC)
    @test size(load_draws(m, :full), 1) == 400

    labels, x, nse = smc_group_estimates(file)
    @test length(labels) == 4 * 13 + 1
    @test labels[end] == "log_mdd"
    @test all(nse .> 0)

    p = plot_smc_stage_health(replace(file, ".jld2" => "_group=1.jld2"))
    @test isfile(replace(file, ".jld2" => "_group=1_stage_health.pdf"))
end

@testset "Comparing runs and validating NSEs" begin
    files = map(1:3) do seed
        m = rel_model("seed$seed"; n_smc_groups = 4, seed = seed)
        DSGE.smc2(m, rel_data; verbose = :none, run_csminwel = false)
        rawpath(m, "estimate", "smc_cloud.jld2")
    end
    c = smc_compare_runs(files[1], files[2])
    @test length(c["z"]) == 4 * 13
    @test 0 <= c["share_above_2"] <= 1
    @test 0 <= c["ks_pvalue"] <= 1
    @test c["log_mdd_a"] == smc_group_estimates(files[1])[2][end]
    v = smc_nse_validation(files)
    @test v["n_runs"] == 3
    @test v["n_smc_groups"] == 4
    @test length(v["sd_ratio"]) == 4 * 13 + 1
end

@testset "Mode check" begin
    m = rel_model("mode")
    cloud = DSGE.smc2(m, rel_data; verbose = :none, run_csminwel = false)
    out = smc_mode_check(m, rel_data, cloud; n_starts = 2, max_hessians = 1, iterations = 50,
                         verbose = :none)
    @test length(out["modes"]) >= 1
    @test out["mode_logpost"][1] >= out["max_particle_logpost"] - 1e-6
    @test isfinite(out["laplace_log_mdd"][1])
    @test length(out["sd_ratio"]) == 13
    @test sum(out["cloud_share"]) ≈ 1
    @test out["flags"] isa Vector{String}
end

@testset "Simulation-based calibration" begin
    m = rel_model("sbc"; n_particles = 200)
    ranks, names = smc_sbc(m, 80; reps = 1:2, n_draws = 19, seed = 3)
    @test size(ranks) == (2, 13)
    @test all(0 .<= ranks .<= 19)
    again, _ = smc_sbc(m, 80; reps = 2:2, n_draws = 19, seed = 3)
    @test again[1, :] == ranks[2, :]   # replications don't depend on how they are split
    @test all(0 .<= smc_sbc_uniformity(ranks, 19; n_bins = 4) .<= 1)
end

rm(rel_dir; recursive = true, force = true)
