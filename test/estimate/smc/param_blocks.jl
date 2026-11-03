using DSGE, ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test, Dates, OrderedCollections

# The SMC mutation-block options (smc_param_blocks, smc_blockwise_c) through DSGE: SMC.jl's own
# tests cover the algorithm; these cover what DSGE adds. Block names must point at the right
# entries of a real (regime-switching) DSGE parameter vector, and a blocked run must resume
# from a DSGE intermediate save.

path = dirname(@__FILE__)
blk_data = h5read(joinpath(path, "reference/smc.h5"), "data")
blk_dir  = mktempdir()

function blk_model(name; n_particles = 100, seed = 1)
    m = AnSchorfheide()
    m <= Setting(:saveroot, joinpath(blk_dir, name))
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
    m <= Setting(:smc_bridge_sampling, false)
    m <= Setting(:smc_seed, seed)
    m <= Setting(:smc_param_blocks, [[:ρ_z], [:σ_R, :σ_g]])
    m <= Setting(:smc_blockwise_c, true)
    return m
end

@testset "A blocked run resumes bitwise from a DSGE intermediate save" begin
    m = blk_model("resume")
    smc2(m, blk_data; verbose = :none, run_csminwel = false, save_intermediate = true,
         intermediate_stage_increment = 5)
    final = rawpath(m, "estimate", "smc_cloud.jld2")
    full  = load(final)
    n_stages = length(full["stage_health"]["stage"])
    k = 5 * div(n_stages - 1, 10)                       # a saved stage in the middle of the run
    @test k >= 5
    ckpt = rawpath(m, "estimate", "smc_cloud") * "_stage=$(k).jld2"
    @test isfile(ckpt)
    @test load(ckpt, "param_blocks") == full["param_blocks"]
    cp(final, final * ".full"; force = true)

    smc2(m, blk_data; verbose = :none, run_csminwel = false, continue_intermediate = true,
         intermediate_stage_start = k)
    resumed = load(final)
    @test resumed["cloud"].particles == full["cloud"].particles
    @test resumed["w"] == full["w"] && resumed["W"] == full["W"]
    @test resumed["stage_health"]["c_blocks"] == full["stage_health"]["c_blocks"]
    @test resumed["stage_health"]["accept_blocks"] == full["stage_health"]["accept_blocks"]

    # Resuming with other blocks, without blocks, or with a shared c is refused
    m2 = deepcopy(m); m2 <= Setting(:smc_param_blocks, [[:ρ_z]])
    @test_throws ArgumentError smc2(m2, blk_data; verbose = :none, run_csminwel = false,
                                    continue_intermediate = true, intermediate_stage_start = k)
    m3 = deepcopy(m); m3 <= Setting(:smc_param_blocks, Vector{Vector{Symbol}}()); m3 <= Setting(:smc_blockwise_c, false)
    @test_throws ArgumentError smc2(m3, blk_data; verbose = :none, run_csminwel = false,
                                    continue_intermediate = true, intermediate_stage_start = k)
    m4 = deepcopy(m); m4 <= Setting(:smc_blockwise_c, false)
    @test_throws ArgumentError smc2(m4, blk_data; verbose = :none, run_csminwel = false,
                                    continue_intermediate = true, intermediate_stage_start = k)
end

@testset "ss104: block names point at the right parameter and regime" begin
    # SMC names the extra regimes Symbol(key, "_reg<i>") in its own loop, while the particle's
    # entries follow ModelConstructors' ordering. If the two disagreed, [[:ρ_σ_w]] would mutate
    # another parameter. The likelihood is zero, so no model is solved.
    m = Model1002("ss104", custom_settings = [Setting(:date_forecast_start, DSGE.quartertodate("2026-Q3")),
                                              Setting(:date_conditional_end, DSGE.quartertodate("2026-Q3"))])
    for k in [:σ_r_m, :σ_r_m1, :σ_r_m2, :σ_r_m3, :σ_r_m4, :σ_r_m5, :σ_r_m6]   # as in the ss104 estimation scripts
        m[k].regimes[:fixed] = OrderedDict(1 => false, 2 => true)
    end
    params = m.parameters
    dir    = mktempdir()
    run_smc(name; kw...) = SMC.smc((p, d) -> 0.0, params, zeros(1, 1); n_parts = 40, use_fixed_schedule = true,
                                   n_Φ = 3, λ = 1.0, n_mh_steps = 1, regime_switching = true, verbose = :none,
                                   smc_seed = 1, savepath = joinpath(dir, "$name.jld2"),
                                   particle_store_path = joinpath(dir, "$name.h5"), kw...)

    # blockwise_c without blocks saves one group with every free parameter, in free-parameter order
    run_smc("all"; blockwise_c = true)
    names = only(load(joinpath(dir, "all.jld2"), "param_blocks"))
    free  = ModelConstructors.get_free_para_inds(params; regime_switching = true, toggle = true)
    @test length(names) == length(free)
    @test :ρ_σ_w in names && :σ_g_reg2 in names && :σ_ziid_reg3 in names
    @test !(:σ_r_m_reg2 in names) && :σ_r_m in names          # regime 2 fixed, regime 1 free

    # Moving entry k of the particle moves exactly the parameter and regime its name gives
    vals = ModelConstructors.get_values(params; regime_switching = true)
    regval(q, r) = isempty(q.regimes) ? q.value : q.regimes[:value][r]
    regs(q) = isempty(q.regimes) ? [1] : collect(keys(q.regimes[:value]))
    function moved_by(k)
        p = deepcopy(params); v = copy(vals); v[k] += 1e-4
        ModelConstructors.update!(p, v)
        return [(q.key, r) for (q, q0) in zip(p, params) for r in regs(q) if regval(q, r) != regval(q0, r)]
    end
    function named(nm)
        mt = match(r"^(.*)_reg(\d+)$", string(nm))
        return mt === nothing ? (nm, 1) : (Symbol(mt[1]), parse(Int, mt[2]))
    end
    @test all(moved_by(k) == [named(nm)] for (k, nm) in zip(free, names))

    # A run with ρ_σ_w in its own block: that block, then every other free parameter
    run_smc("blk"; param_blocks = [[:ρ_σ_w]], blockwise_c = true)
    out = load(joinpath(dir, "blk.jld2"))
    @test out["param_blocks"] == [[:ρ_σ_w], Base.filter(!=(:ρ_σ_w), names)]
    @test out["stage_health"]["block_labels"] == ["ρ_σ_w", "rest"]

    # Fixed parameters and fixed regimes cannot be blocks
    @test_throws ArgumentError run_smc("fix1"; param_blocks = [[:σ_r_m_reg2]])
    @test_throws ArgumentError run_smc("fix2"; param_blocks = [[:σ_φ_prop_reg2]])
    rm(dir; recursive = true, force = true)
end

@testset "smc_sbc passes the block settings to SMC" begin
    m = blk_model("sbc")
    s = DSGE.sbc_smc_settings(m)
    @test s.param_blocks == [[:ρ_z], [:σ_R, :σ_g]] && s.blockwise_c
end

rm(blk_dir; recursive = true, force = true)
