using DSGE, ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test

# :smc_reuse_system_new_params. In a same-model bridge (old_model === m) the old-data likelihood
# of every proposal is evaluated on the system just solved for its new-data likelihood. It must
# not change anything: the bridged run is bitwise the run that solves the model twice.

path = dirname(@__FILE__)
rs_data     = h5read(joinpath(path, "reference/smc.h5"), "data")
rs_old_data = rs_data[:, 1:(size(rs_data, 2) - 30)]          # old ≠ new: 30 fewer periods
rs_dir      = mktempdir()

function rs_model(name)
    m = AnSchorfheide()
    m <= Setting(:saveroot, joinpath(rs_dir, name))
    m <= Setting(:n_particles, 100)
    m <= Setting(:n_Φ, 20)
    m <= Setting(:λ, 2.0)
    m <= Setting(:n_smc_blocks, 1)
    m <= Setting(:use_parallel_workers, false)
    m <= Setting(:step_size_smc, 0.5)
    m <= Setting(:n_mh_steps_smc, 2)
    m <= Setting(:resampler_smc, :systematic)
    m <= Setting(:target_accept, 0.25)
    m <= Setting(:mixture_proportion, 0.9)
    m <= Setting(:use_fixed_schedule, true)
    m <= Setting(:resampling_threshold, 0.5)
    m <= Setting(:smc_iteration, 0)
    m <= Setting(:use_chand_recursion, true)
    m <= Setting(:tempered_update_prior_weight, 0.0)
    return m
end

@testset "Bridge reusing the new-data system for the old data" begin
    m_old = rs_model("old")
    smc2(m_old, rs_old_data; verbose = :none, run_csminwel = false, smc_seed = 21)
    old_cloud = load(rawpath(m_old, "estimate", "smc_cloud.jld2"), "cloud")

    @test get_setting(rs_model("x"), :smc_reuse_system_new_params) == false
    function bridge(name, reuse; kw...)
        m = rs_model(name)
        m <= Setting(:smc_reuse_system_new_params, reuse)
        t = @elapsed smc2(m, rs_data; verbose = :none, run_csminwel = false, smc_seed = 22,
                          old_data = rs_old_data, old_cloud = deepcopy(old_cloud), kw...)
        return load(rawpath(m, "estimate", "smc_cloud.jld2")), t, m
    end
    bridge("warmup", true)                                   # compile both paths before timing
    bridge("warmup2", false)
    off, t_off, m_off = bridge("off", false)
    on,  t_on,  _     = bridge("on", true)
    @info "Bridge wall time: reuse off $(round(t_off; digits = 2)) s, reuse on $(round(t_on; digits = 2)) s"

    striptime(h) = Dict(k => v for (k, v) in h if k != "time")
    @testset "The whole final cloud is bitwise the same" begin
        c_on, c_off = on["cloud"], off["cloud"]
        @test isequal(c_on.particles, c_off.particles)        # includes loglh and old_loglh
        @test isequal(SMC.get_old_loglh(c_on), SMC.get_old_loglh(c_off))
        @test isequal(SMC.get_loglh(c_on), SMC.get_loglh(c_off))
        for f in (:tempering_schedule, :ESS, :stage_index, :n_Φ, :resamples, :c, :accept)
            @test isequal(getfield(c_on, f), getfield(c_off, f))
        end
        @test isequal(on["w"], off["w"]) && isequal(on["W"], off["W"])
        @test isequal(striptime(on["stage_health"]), striptime(off["stage_health"]))
        @test isequal(on["log_mdd_estimates"], off["log_mdd_estimates"])
    end

    @testset "The old-data loglh is that of a separate solve" begin
        # Every final particle's old_loglh recomputed on a fresh model with the old data
        P  = SMC.get_vals(on["cloud"])
        mf = rs_model("fresh")
        fresh = map(1:size(P, 2)) do i
            DSGE.update!(mf, P[:, i])
            likelihood(mf, rs_old_data; sampler = false, catch_errors = true, verbose = :none,
                       use_chand_recursion = true)
        end
        @test isequal(SMC.get_old_loglh(on["cloud"]), fresh)
        @test !isequal(SMC.get_old_loglh(on["cloud"]), SMC.get_loglh(on["cloud"]))   # old ≠ new data
    end

    @testset "No reuse when old_model is another object" begin
        sep_off, _, _ = bridge("sep_off", false; old_model = rs_model("sep_old_off"))
        sep_on,  _, _ = bridge("sep_on", true;  old_model = rs_model("sep_old_on"))
        @test isequal(sep_on["cloud"].particles, sep_off["cloud"].particles)
        @test isequal(sep_on["cloud"].particles, off["cloud"].particles)   # an equal old model: same run
    end
end

rm(rs_dir; recursive = true, force = true)
