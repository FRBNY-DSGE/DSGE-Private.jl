using DSGE, ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test, Dates, OrderedCollections

# Vintage bridge through DSGE (smc2 / estimate): same model object (old_model === m, the
# default), same priors, same number of periods; old_data = a "revised" vintage, data = a
# "real-time" vintage that differs in some observations. With mutation blocks + blockwise c.
#   1. every particle's old_loglh equals a fresh likelihood on old_data, and loglh a fresh one
#      on data (so no cached system or memo from one data set leaks into the other);
#   2. a resume from a mid-bridge intermediate save is bitwise equal to the uninterrupted run;
#   3. estimate() passes the bridge arguments through (bitwise equal to smc2);
#   4. identity bridge (old_data == data as a separate matrix): one stage, constant weights;
#   5. the same with a regime-switching AnSchorfheide (old_model === m, regime-switching update).

path = dirname(@__FILE__)
vb_dir = mktempdir()
vb_rev = h5read(joinpath(path, "reference/smc.h5"), "data")          # "revised" vintage
vb_rt  = let r = Xoshiro(5), d = copy(vb_rev)                        # "real-time" vintage
    T = size(d, 2); d[:, (T ÷ 2):T] .+= 0.3 .* randn(r, size(d, 1), T - T ÷ 2 + 1); d
end

function vb_model(name; n_particles = 300, seed = 1, blocks = true)
    m = AnSchorfheide()
    m <= Setting(:saveroot, joinpath(vb_dir, name))
    for (k, v) in [(:n_particles, n_particles), (:n_Φ, 100), (:λ, 2.0), (:n_smc_blocks, 1),
                   (:use_parallel_workers, false), (:step_size_smc, 0.5), (:n_mh_steps_smc, 2),
                   (:resampler_smc, :systematic), (:target_accept, 0.25), (:mixture_proportion, 0.9),
                   (:use_fixed_schedule, false), (:adaptive_tempering_target_smc, 0.95),
                   (:resampling_threshold, 0.5), (:smc_iteration, 0), (:use_chand_recursion, true),
                   (:smc_bridge_sampling, false), (:smc_seed, seed), (:tempered_update_prior_weight, 0.0)]
        m <= Setting(k, v)
    end
    if blocks
        m <= Setting(:smc_param_blocks, [[:ρ_z], [:σ_R, :σ_g]])
        m <= Setting(:smc_blockwise_c, true)
    end
    return m
end

# fresh likelihoods of every particle of `cloud` on `data`, on a deep copy of the model
function vb_fresh(m, cloud, data; regime_switching = false)
    P = SMC.get_vals(cloud)
    pv = deepcopy(m.parameters)
    mf = deepcopy(m)
    map(1:size(P, 2)) do i
        ModelConstructors.update!(pv, P[:, i])
        DSGE.update!(mf, pv; regime_switching = regime_switching)
        mf <= Setting(:preprocessed_transitions, Dict())
        likelihood(mf, data; sampler = false, catch_errors = true, verbose = :none,
                   use_chand_recursion = get_setting(m, :use_chand_recursion))
    end
end

@testset "Vintage bridge through smc2: old_model === m, blocks + blockwise c" begin
    @test size(vb_rt) == size(vb_rev) && vb_rt != vb_rev
    m = vb_model("ab")
    smc2(m, vb_rev; verbose = :none, run_csminwel = false, filestring_addl = ["old"])
    old = load(rawpath(m, "estimate", "smc_cloud.jld2", ["old"]))
    old_cloud = old["cloud"]
    lmdd_old = old["log_mdd_estimates"]["smc"]
    println("old cloud: ESS[end] = $(old_cloud.ESS[end]), weights ESS = ",
            sum(SMC.get_weights(old_cloud))^2 / sum(abs2, SMC.get_weights(old_cloud)))
    # the stored old loglh is the likelihood on the revised data
    @test maximum(abs.(SMC.get_loglh(old_cloud) .- vb_fresh(m, old_cloud, vb_rev))) < 1e-8

    smc2(m, vb_rt; verbose = :none, run_csminwel = false, filestring_addl = ["bridge"],
         old_data = vb_rev, old_cloud = deepcopy(old_cloud), log_prob_old_data = lmdd_old,
         save_intermediate = true, intermediate_stage_increment = 2)
    final = rawpath(m, "estimate", "smc_cloud.jld2", ["bridge"])
    full = load(final)
    cloud = full["cloud"]
    n = cloud.stage_index
    println("bridge: $(n - 1) stages, schedule ", round.(cloud.tempering_schedule, digits = 4))
    @test n > 2
    @test full["stage_health"]["block_labels"] == ["ρ_z", "block2", "rest"]

    # 1. stored columns = fresh evaluations on the right data
    fresh_old = vb_fresh(m, cloud, vb_rev)
    fresh_new = vb_fresh(m, cloud, vb_rt)
    @test maximum(abs.(SMC.get_old_loglh(cloud) .- fresh_old)) < 1e-8
    @test maximum(abs.(SMC.get_loglh(cloud) .- fresh_new)) < 1e-8
    @test maximum(abs.(fresh_old .- fresh_new)) > 1.0      # the two vintages really differ
    @test isfinite(full["log_mdd_estimates"]["smc_level"])

    # 2. resume mid-bridge
    k = 2 * max(1, div(n - 1, 4))
    @test isfile(rawpath(m, "estimate", "smc_cloud", ["bridge"]) * "_stage=$(k).jld2")
    cp(final, final * ".full"; force = true)
    smc2(m, vb_rt; verbose = :none, run_csminwel = false, filestring_addl = ["bridge"],
         old_data = vb_rev, old_cloud = deepcopy(old_cloud), log_prob_old_data = lmdd_old,
         continue_intermediate = true, intermediate_stage_start = k)
    res = load(final)
    println("resumed from stage $k of $n")
    @test res["cloud"].particles == full["cloud"].particles
    @test res["w"] == full["w"] && res["W"] == full["W"]
    @test res["cloud"].tempering_schedule == full["cloud"].tempering_schedule
    @test res["stage_health"]["c_blocks"] == full["stage_health"]["c_blocks"]
    @test res["log_mdd_estimates"]["smc"] == full["log_mdd_estimates"]["smc"]

    # 3. estimate() gives the same bridge
    m2 = vb_model("ab")
    m2 <= Setting(:sampling_method, :SMC)
    DSGE.estimate(m2, vb_rt; verbose = :none, run_csminwel = false, filestring_addl = ["est"],
             old_data = vb_rev, old_cloud = deepcopy(old_cloud), log_prob_old_data = lmdd_old)
    est = load(rawpath(m2, "estimate", "smc_cloud.jld2", ["est"]))
    @test est["cloud"].particles == full["cloud"].particles

    # 4. identity bridge, without log_prob_old_data (μ = 0 needs none): one stage, constant weights
    same = copy(vb_rev)
    smc2(m, vb_rev; verbose = :none, run_csminwel = false, filestring_addl = ["ident"],
         old_data = same, old_cloud = deepcopy(old_cloud))
    id = load(rawpath(m, "estimate", "smc_cloud.jld2", ["ident"]))
    @test id["cloud"].tempering_schedule == [0.0, 1.0]
    @test maximum(abs.(log.(id["w"][:, 2]))) < 1e-8
    @test SMC.get_old_loglh(id["cloud"]) == SMC.get_loglh(id["cloud"])
    @test isnan(id["log_mdd_estimates"]["smc_level"])
    @test abs(id["log_mdd_estimates"]["smc"]) < 1e-8

    # 4b. the same with log_prob_old_data = the old MDD: exp(Δϕ log p(Ỹ)) underflows once
    # Δϕ |log p(Ỹ)| > 745, so the run takes an extra stage at the underflow boundary and its MDD
    # is off (first run, job 343018: schedule [0, 0.6404, 1], MDD level error 0.288). Known bug.
    smc2(m, vb_rev; verbose = :none, run_csminwel = false, filestring_addl = ["ident_lpod"],
         old_data = same, old_cloud = deepcopy(old_cloud), log_prob_old_data = lmdd_old)
    id2 = load(rawpath(m, "estimate", "smc_cloud.jld2", ["ident_lpod"]))
    println("identity with log_prob_old_data = $lmdd_old: schedule ", id2["cloud"].tempering_schedule,
            ", MDD level error ", id2["log_mdd_estimates"]["smc_level"] - lmdd_old)
    @test_broken id2["cloud"].tempering_schedule == [0.0, 1.0]
    @test_broken abs(id2["log_mdd_estimates"]["smc_level"] - lmdd_old) < 1e-8
end

@testset "Vintage bridge, regime-switching AnSchorfheide, old_model === m" begin
    data = h5read(joinpath(path, "..", "..", "reference", "sim_data_regswitch_anschorfheide.h5"), "data")
    rt = let r = Xoshiro(6), d = copy(data)
        T = size(d, 2); d[:, (T - 20):T] .+= 0.3 .* randn(r, size(d, 1), 21); d
    end
    m = vb_model("rs"; n_particles = 150, blocks = false)
    m <= Setting(:smc_param_blocks, [[:ρ_z_reg2]])
    m <= Setting(:smc_blockwise_c, true)
    m <= Setting(:regime_switching, true, true, "rs2", "")
    m <= Setting(:n_regimes, 3)
    m <= Setting(:regime_dates, Dict(1 => date_presample_start(m),
                                     2 => DSGE.iterate_quarters(Date(1960, 3, 31), 50),
                                     3 => DSGE.iterate_quarters(Date(1960, 3, 31), 100)))
    setup_regime_switching_inds!(m)
    m <= Setting(:model2para_regimes, Dict{Int, Int}(1 => 1, 2 => 1, 3 => 2))
    for i in 1:length(m.parameters), k in 1:2
        ModelConstructors.set_regime_val!(m.parameters[i], k, m.parameters[i].value)
    end
    DSGE.setup_param_regimes!(m, repeat([1 1 2], length(m.parameters)))
    m <= Setting(:use_chand_recursion, false)

    smc2(m, data; regime_switching = true, verbose = :none, run_csminwel = false, filestring_addl = ["old"])
    old = load(rawpath(m, "estimate", "smc_cloud.jld2", ["old"]))
    smc2(m, rt; regime_switching = true, verbose = :none, run_csminwel = false, filestring_addl = ["bridge"],
         old_data = data, old_cloud = deepcopy(old["cloud"]),
         log_prob_old_data = old["log_mdd_estimates"]["smc"])
    out = load(rawpath(m, "estimate", "smc_cloud.jld2", ["bridge"]))
    cloud = out["cloud"]
    @test cloud.stage_index > 2
    fo = vb_fresh(m, cloud, data; regime_switching = true)
    fn = vb_fresh(m, cloud, rt; regime_switching = true)
    @test maximum(abs.(SMC.get_old_loglh(cloud) .- fo)) < 1e-8
    @test maximum(abs.(SMC.get_loglh(cloud) .- fn)) < 1e-8
    @test maximum(abs.(fo .- fn)) > 1.0
end

rm(vb_dir; recursive = true, force = true)
