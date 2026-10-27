# One-time SMC validation on An–Schorfheide (SMC reliability plan, step 3 part C, items 5 and 6).
# Each task is one job; run the per-seed / per-chunk tasks as a job array and then the summaries.
#
#   julia smc_reliability_an_schorfheide.jl <outdir> nse <seed>             item 5: one grouped run
#   julia smc_reliability_an_schorfheide.jl <outdir> nse_summary            item 5: cross-run sd vs NSE
#   julia smc_reliability_an_schorfheide.jl <outdir> sbc <first> <last>     item 5: SBC replications
#   julia smc_reliability_an_schorfheide.jl <outdir> sbc_summary            item 5: rank uniformity
#   julia smc_reliability_an_schorfheide.jl <outdir> bridge <seed>          item 6: old run, full run,
#                                                                           and the three bridges
#
# Settings follow production ss104 (refactor/base_m1002_ss104.jl): adaptive tempering (0.95),
# 3 MH steps in 1 block, α = 0.975, resampling threshold 0.5, N = 3000, here in J = 10 groups.
# A production-size run takes about 50 minutes on one core; the environment variables
# SMC_REL_N, SMC_REL_J and SMC_REL_SBC_N (particles per SBC replication, default 1000) set
# smaller sizes.
# The output records the SMC/DSGE commits (run_info), so the same tasks rerun after the step-3
# fixes give the before/after comparison.

using DSGE, ModelConstructors, SMC, HDF5, JLD2, FileIO, Random, Printf, Statistics

const OUTDIR = ARGS[1]
const TASK   = ARGS[2]
mkpath(OUTDIR)

const DATA   = h5read(joinpath(pkgdir(DSGE), "test", "estimate", "smc", "reference", "smc.h5"), "data")
const T_OLD  = size(DATA, 2) - 20     # item 6: the bridge adds the last 20 quarters
const N      = parse(Int, get(ENV, "SMC_REL_N", "3000"))
const J      = parse(Int, get(ENV, "SMC_REL_J", "10"))
const N_SBC  = parse(Int, get(ENV, "SMC_REL_SBC_N", "1000"))

m = AnSchorfheide()
m <= Setting(:use_chand_recursion, true)

loglik = function (parameters, data)
    DSGE.update!(m, parameters)
    likelihood(m, data; sampler = false, catch_errors = true)
end

settings = (n_parts = N, n_smc_groups = J, n_blocks = 1, n_mh_steps = 3, α = 0.975, c = 0.5,
            target = 0.25, use_fixed_schedule = false, tempering_target = 0.95, n_Φ = 300, λ = 2.1,
            threshold_ratio = 0.5, resampling_method = :multinomial, bridge_sampling = true,
            verbose = :low)

# Every run has an explicit seed, so a rerun of a task reuses the groups already finished
run_smc(name, data; kwargs...) =
    SMC.smc(loglik, m.parameters, data; settings..., savepath = joinpath(OUTDIR, name * ".jld2"),
            particle_store_path = joinpath(OUTDIR, name * ".h5"), continue_intermediate = true,
            kwargs...)

if TASK == "nse"
    seed = parse(Int, ARGS[3])
    run_smc(@sprintf("nse_seed=%03d", seed), DATA; smc_seed = seed)

elseif TASK == "nse_summary"
    files = Base.filter(f -> occursin(r"^nse_seed=\d+\.jld2$", f), readdir(OUTDIR))
    v = smc_nse_validation(joinpath.(OUTDIR, files))
    jldopen(joinpath(OUTDIR, "nse_summary.jld2"), "w") do file
        for (key, value) in v
            write(file, key, value)
        end
    end
    open(joinpath(OUTDIR, "nse_summary.csv"), "w") do io
        println(io, "estimate,grand_mean,cross_sd,rms_nse,sd_ratio,coverage")
        for k in eachindex(v["labels"])
            println(io, join((v["labels"][k], v["grand_mean"][k], v["cross_sd"][k], v["rms_nse"][k],
                              v["sd_ratio"][k], v["coverage"][k]), ","))
        end
    end
    @printf "%d runs, J = %d: median sd ratio %.3f, mean coverage %.3f\n" v["n_runs"] v["n_smc_groups"] v["median_sd_ratio"] v["mean_coverage"]

elseif TASK == "sbc"
    # Smaller N per replication: SBC tests the sampler's calibration, not its precision
    reps = parse(Int, ARGS[3]):parse(Int, ARGS[4])
    ranks, names = smc_sbc(m, 100; reps = reps, n_draws = 99, seed = 1793, n_parts = N_SBC,
                           n_blocks = 1, n_mh_steps = 3, α = 0.975, c = 0.5, target = 0.25,
                           use_fixed_schedule = false, tempering_target = 0.95, threshold_ratio = 0.5,
                           resampling_method = :multinomial)
    jldsave(joinpath(OUTDIR, "sbc_$(first(reps))_$(last(reps)).jld2"); ranks, names, reps = collect(reps))

elseif TASK == "sbc_summary"
    files = Base.filter(f -> occursin(r"^sbc_\d+_\d+\.jld2$", f), readdir(OUTDIR))
    ranks = reduce(vcat, [load(joinpath(OUTDIR, f), "ranks") for f in files])
    names = load(joinpath(OUTDIR, files[1]), "names")
    p = smc_sbc_uniformity(ranks, 99)
    jldsave(joinpath(OUTDIR, "sbc_summary.jld2"); ranks, names, pvalues = p)
    println("$(size(ranks, 1)) replications; χ² uniformity p-values per parameter:")
    for (name, pk) in zip(names, p)
        @printf "  %-8s %.3f\n" name pk
    end

elseif TASK == "bridge"
    seed = parse(Int, ARGS[3])
    old_data = DATA[:, 1:T_OLD]
    tag(name) = @sprintf("%s_seed=%03d", name, seed)
    seeds = SMC.stream_seed.(UInt64(seed), [:old, :full, :mu0_same, :mu0_resized, :mu025])

    run_smc(tag("old"), old_data; smc_seed = seeds[1])
    old_file = joinpath(OUTDIR, tag("old") * ".jld2")
    old_groups = [load(replace(old_file, ".jld2" => "_group=$g.jld2"), "cloud") for g in 1:J]
    old_log_mdd = load(old_file, "log_mdd_estimates")["smc_level"]
    old_log_mdd_nse = smc_group_estimates(old_file)[3][end]

    run_smc(tag("full"), DATA; smc_seed = seeds[2])
    bridge = (old_data = old_data, old_loglikelihood = loglik)
    # μ = 0, same N: each group bridges from the matching old group (branch 1)
    run_smc(tag("mu0_same"), DATA; bridge..., old_cloud = old_groups, smc_seed = seeds[3])
    # μ = 0, different N: 1.5× the old group size (branch 2: F2, F3, F7)
    run_smc(tag("mu0_resized"), DATA; bridge..., old_cloud = old_groups, smc_seed = seeds[4],
            n_parts = 3N ÷ 2)
    # μ = 0.25 (F1, F5)
    run_smc(tag("mu025"), DATA; bridge..., old_cloud = old_groups, smc_seed = seeds[5],
            tempered_update_prior_weight = 0.25, log_prob_old_data = old_log_mdd)

    full = joinpath(OUTDIR, tag("full") * ".jld2")
    results = Dict{String, Any}("old_log_mdd" => old_log_mdd,
                                "run_info" => load(full, "run_info"))
    # Without log_prob_old_data a μ = 0 bridge reports the increment log p(Y) - log p(Ỹ) (F4), so
    # the old log MDD and its error add in. With it (μ = 0.25), an error δ in log p(Ỹ) moves the
    # result by log(μ + (1 - μ)e^(-δ)) ≈ -(1 - μ)δ: no offset, but (1 - μ) times the old NSE.
    for (variant, offset, offset_nse) in (("mu0_same", old_log_mdd, old_log_mdd_nse),
                                          ("mu0_resized", old_log_mdd, old_log_mdd_nse),
                                          ("mu025", 0.0, 0.75 * old_log_mdd_nse))
        c = smc_compare_runs(joinpath(OUTDIR, tag(variant) * ".jld2"), full; log_mdd_offset_a = offset,
                             log_mdd_offset_nse_a = offset_nse)
        results[variant] = c
        @printf "%-12s |z|>2: %5.1f%%  KS p = %.3g  log MDD %.3f vs full %.3f (z = %.2f)  time %.0fs vs %.0fs\n" variant 100c["share_above_2"] c["ks_pvalue"] c["log_mdd_a"] c["log_mdd_b"] c["log_mdd_z"] c["time_a"] c["time_b"]
    end
    jldsave(joinpath(OUTDIR, tag("bridge_summary") * ".jld2"); results)

else
    error("unknown task $TASK")
end
