# SMC reliability checks that need a model (SMC reliability plan, step 3 part C). The
# model-free pieces (per-stage health, lineage NSEs, the log MDD estimators, particle groups,
# the cross-run comparisons) live in SMC.jl: src/diagnostics.jl and src/groups.jl.
#
#   item 2  plot_smc_stage_health: the one-page plot of a run's per-stage health
#   item 4  smc_mode_check: optimizer from the best particles, Hessian, Laplace log MDD,
#           multimodality
#   item 5  smc_nse_validation (R independent grouped runs) and smc_sbc (simulation-based
#           calibration)
#   item 6  smc_compare_runs: bridge vs full tempering

"""
```
plot_smc_stage_health(file; outfile = <file>_stage_health.pdf)
```
One-page plot of the per-stage health record saved in an SMC output `file` (item 2): the
tempering schedule, ESS and conditional ESS, particle diversity, acceptance and scale, particle
movement, and -Inf proposals, with the flag thresholds as dashed lines and the run's flags
in the title. Returns the plot.
"""
function plot_smc_stage_health(file::AbstractString;
                               outfile::AbstractString = replace(file, r"\.jld2$" => "") * "_stage_health.pdf")
    h, flags = load(file, "stage_health"), load(file, "health_flags")
    t = SMC.DEFAULT_HEALTH_THRESHOLDS
    x = h["stage"]
    nanmax(v) = all(isnan, v) ? NaN : maximum(Base.filter(!isnan, v))
    nanmedian(v) = all(isnan, v) ? NaN : median(Base.filter(!isnan, v))

    p_phi = plot(x, h["phi"], label = "φ", xlabel = "stage", legend = :topleft)
    plot!(twinx(p_phi), x, h["dphi"], label = "Δφ", color = :red, yscale = :log10, legend = :bottomright)
    p_ess = plot(x, [h["ess"] h["cess"] h["max_weight"]], label = ["ESS/N" "CESS/N" "max weight"],
                 xlabel = "stage", ylims = (0, 1.05))
    hline!(p_ess, [t.min_ess], linestyle = :dash, color = :black, label = "")
    p_div = plot(x, [h["distinct_selected"] h["distinct_mutated"]],
                 label = ["distinct after selection" "distinct after mutation"], xlabel = "stage",
                 ylims = (0, 1.05))
    hline!(p_div, [t.min_distinct], linestyle = :dash, color = :black, label = "")
    p_acc = plot(x, h["accept"], label = "acceptance", xlabel = "stage")
    hline!(p_acc, collect(t.accept_range), linestyle = :dash, color = :black, label = "")
    plot!(twinx(p_acc), x, h["c"], label = "c", color = :red, legend = :bottomright)
    p_move = plot(x, [nanmax(h["corr"][:, s]) for s in axes(h["corr"], 2)],
                  label = "max pre/post correlation", xlabel = "stage", ylims = (-0.05, 1.05))
    plot!(p_move, x, [nanmedian(h["esjd"][:, s]) for s in axes(h["esjd"], 2)],
          label = "median relative jump²")
    hline!(p_move, [t.max_late_corr], linestyle = :dash, color = :black, label = "")
    p_inf = plot(x, h["inf_proposals"] ./ max.(h["n_proposals"], 1), label = "-Inf proposals",
                 xlabel = "stage")

    title = isempty(flags) ? "no health flags" : join(first(flags, 3), "\n") *
        (length(flags) > 3 ? "\n(+ $(length(flags) - 3) more)" : "")
    p = plot(p_phi, p_ess, p_div, p_acc, p_move, p_inf, layout = (3, 2), size = (1100, 1000),
             plot_title = title, plot_titlefontsize = 8)
    savefig(p, outfile)
    return p
end

"""
```
smc_mode_check(m, data, cloud; n_starts = 20, max_hessians = 3, distinct_sd = 0.5, k_max = 4)
```
Item 4: compare the SMC cloud with the posterior mode.
- Runs `optimize!` (csminwel) from the `n_starts` best distinct particles. Endpoints more than
  `distinct_sd` posterior sds apart in some parameter count as distinct modes.
- Computes the Hessian at the best mode, and at up to `max_hessians` modes in total, and the
  Laplace log MDD log p(Y|θ̂) + log p(θ̂) + (d/2) log 2π - ½ log|H| (Tierney and Kadane 1986).
- Per free parameter: (SMC mean - mode) in SMC posterior sds, and SMC sd / inverse-Hessian sd.
- Multimodality: each particle is assigned to its nearest mode (in the cloud's Mahalanobis
  metric) to get the cloud's share near each mode, compared with the modes' Laplace masses;
  and a Gaussian mixture with BIC on the cloud (`SMC.gaussian_mixture_bic`).

Returns a `Dict`. Its `"flags"` fire when a mode has a higher posterior than every particle
but under 1% of the cloud nearby (a fail: the cloud misses a mode), when the cloud's share
near a mode differs from its Laplace share by more than 0.2, or when the mixture prefers more
than one component. Large mean–mode or sd gaps are reported, not flagged. Each optimization
costs thousands of likelihood evaluations and each Hessian about d² more, so this is for
one-off checks.
"""
function smc_mode_check(m::AbstractDSGEModel, data::AbstractArray, cloud::SMC.Cloud;
                        n_starts::Int = 20, max_hessians::Int = 3, distinct_sd::Real = 0.5,
                        k_max::Int = 4, iterations::Int = 1000, rng::AbstractRNG = Xoshiro(0),
                        verbose::Symbol = :low)
    regime_switching = haskey(get_settings(m), :regime_switching) && get_setting(m, :regime_switching)
    free = ModelConstructors.get_free_para_inds(get_parameters(m); regime_switching = regime_switching)
    n_para = SMC.ind_para_end(size(cloud.particles, 2))
    θ = cloud.particles[:, 1:n_para]
    W = SMC.get_weights(cloud) ./ sum(SMC.get_weights(cloud))
    logpost = SMC.get_logpost(cloud)
    μ, Σ = SMC.weighted_mean_cov(θ[:, free], W)
    sd = sqrt.(diag(Σ))
    names = [p.key for p in get_parameters(m)]
    names = length(names) == n_para ? names[free] : Symbol.("θ" .* string.(free))

    # Optimize from the best distinct particles
    starts = Int[]
    for k in sortperm(logpost; rev = true)
        any(s -> θ[s, :] == θ[k, :], starts) || push!(starts, k)
        length(starts) == n_starts && break
    end
    endpoints, end_logpost = Vector{Vector{Float64}}(), Float64[]
    for (n, k) in enumerate(starts)
        println(verbose, :low, "smc_mode_check: optimization $n of $(length(starts))")
        DSGE.update!(m, θ[k, :])
        out, _ = optimize!(m, data; iterations = iterations)
        push!(endpoints, copy(out.minimizer))
        push!(end_logpost, posterior!(m, out.minimizer, data))
    end

    # Distinct modes, best first
    modes, mode_logpost = Vector{Vector{Float64}}(), Float64[]
    for k in sortperm(end_logpost; rev = true)
        isfinite(end_logpost[k]) || continue
        any(x -> maximum(abs.(x[free] .- endpoints[k][free]) ./ sd) <= distinct_sd, modes) && continue
        push!(modes, endpoints[k])
        push!(mode_logpost, end_logpost[k])
    end
    isempty(modes) && error("smc_mode_check: no optimization ended at a finite posterior")

    # Hessians and Laplace log MDDs (the Hessian is of minus the log posterior)
    d = length(free)
    laplace = fill(NaN, length(modes))
    hessian_free = Matrix{Float64}[]
    for k in 1:min(max_hessians, length(modes))
        println(verbose, :low, "smc_mode_check: Hessian at mode $k")
        H, _ = hessian!(m, modes[k], data)
        Hf = Symmetric(H[free, free])
        push!(hessian_free, Matrix(Hf))
        laplace[k] = isposdef(Hf) ? mode_logpost[k] + d / 2 * log(2π) - logdet(cholesky(Hf)) / 2 : NaN
    end
    best = modes[1]
    hess_sd = isposdef(Symmetric(hessian_free[1])) ? sqrt.(diag(inv(Symmetric(hessian_free[1])))) :
        fill(NaN, d)

    # The cloud's weight nearest each mode, against the modes' Laplace masses
    Σ_inv = inv(Symmetric(SMC.floored_cov(Σ)))
    nearest = [argmin([dot(θ[i, free] .- x[free], Σ_inv * (θ[i, free] .- x[free])) for x in modes])
               for i in axes(θ, 1)]
    cloud_share = [sum(W[nearest .== k]) for k in eachindex(modes)]
    have_laplace = findall(isfinite, laplace)
    laplace_share = fill(NaN, length(modes))
    if length(have_laplace) > 1
        laplace_share[have_laplace] = exp.(laplace[have_laplace] .- SMC.logsumexp(laplace[have_laplace]))
    end
    mixture = SMC.gaussian_mixture_bic(θ[:, free], W; k_max = k_max, rng = rng)

    flags = String[]
    for k in eachindex(modes)
        if mode_logpost[k] > maximum(logpost) && cloud_share[k] < 0.01
            push!(flags, "mode $k (log posterior $(round(mode_logpost[k], digits = 2)), above every " *
                         "particle) has $(round(100 * cloud_share[k], digits = 2))% of the cloud near it: " *
                         "the cloud misses a mode")
        end
        if isfinite(laplace_share[k]) && abs(cloud_share[k] - laplace_share[k]) > 0.2
            push!(flags, "mode $k: cloud share $(round(cloud_share[k], digits = 2)) vs Laplace share " *
                         "$(round(laplace_share[k], digits = 2))")
        end
    end
    mixture.best_k > 1 &&
        push!(flags, "a Gaussian mixture with $(mixture.best_k) components fits the cloud best (BIC)")

    return Dict{String, Any}("parameters" => names, "modes" => modes, "mode_logpost" => mode_logpost,
                             "laplace_log_mdd" => laplace, "hessian_free" => hessian_free,
                             "mean_minus_mode_sd" => (μ .- best[free]) ./ sd,
                             "sd_ratio" => sd ./ hess_sd, "cloud_share" => cloud_share,
                             "laplace_share" => laplace_share, "mixture_bic" => mixture.bic,
                             "mixture_best_k" => mixture.best_k, "max_particle_logpost" => maximum(logpost),
                             "flags" => flags)
end

"""
```
smc_group_estimates(file)
```
The pooled estimates and NSEs of a grouped SMC run (`n_smc_groups > 1`) saved in `file`, as
`(labels, estimates, nses)`: per free parameter the mean and the 5/50/95% quantiles, then the
log MDD (the level when known, otherwise the incremental-weight product).
"""
function smc_group_estimates(file::AbstractString)
    s = load(file, "smc_groups")
    labels, est, nse = String[], Float64[], Float64[]
    for key in ("mean", "q05", "q50", "q95")
        nse_key = key == "mean" ? "mean_nse" : key * "_nse"
        append!(labels, string.(s["parameters"]) .* "_" .* key)
        append!(est, s[key])
        append!(nse, s[nse_key])
    end
    mdd = s["log_mdd_estimates"]
    key = haskey(mdd, "smc_level") ? "smc_level" : "smc"
    push!(labels, "log_mdd")
    push!(est, mdd[key])
    push!(nse, mdd[key * "_nse"])
    return labels, est, nse
end

"""
```
smc_nse_validation(files; level = 0.95)
```
Item 5: validate the in-run NSEs of R independent grouped runs (their pooled output files) with
`SMC.nse_validation`. Returns a `Dict` with, per estimate, the cross-run sd / RMS in-run NSE
(≈ 1) and the interval coverage (≈ `level`), plus the summary over estimates.
"""
function smc_nse_validation(files::AbstractVector{<:AbstractString}; level::Real = 0.95)
    runs = smc_group_estimates.(files)
    labels = runs[1][1]
    all(r -> r[1] == labels, runs) || error("smc_nse_validation: the runs estimate different quantities")
    J = load(files[1], "n_smc_groups")
    v = SMC.nse_validation(reduce(vcat, [r[2]' for r in runs]), reduce(vcat, [r[3]' for r in runs]),
                           J; level = level)
    return Dict{String, Any}("labels" => labels, "sd_ratio" => v.sd_ratio, "coverage" => v.coverage,
                             "cross_sd" => v.cross_sd, "rms_nse" => v.rms_nse, "grand_mean" => v.grand_mean,
                             "median_sd_ratio" => median(v.sd_ratio), "mean_coverage" => mean(v.coverage),
                             "n_runs" => length(files), "n_smc_groups" => J)
end

"""
```
smc_compare_runs(file_a, file_b; log_mdd_offset_a = 0.0)
```
Item 6: compare two grouped SMC runs of the same posterior, e.g. a bridge (`file_a`) and full
tempering (`file_b`), with `SMC.compare_estimates` on the means and quantiles: z per estimate,
the share with |z| > 2 (≈ 5% if they agree) and a KS test of the z's against N(0, 1). The log
MDDs are compared separately, after adding `log_mdd_offset_a` to the first (e.g. the old run's
log MDD, for a μ = 0 bridge that reports the increment). Also reports the run times.
"""
function smc_compare_runs(file_a::AbstractString, file_b::AbstractString; log_mdd_offset_a::Real = 0.0)
    la, a, nse_a = smc_group_estimates(file_a)
    lb, b, nse_b = smc_group_estimates(file_b)
    la == lb || error("smc_compare_runs: the runs estimate different quantities")
    para = 1:length(la)-1
    c = SMC.compare_estimates(a[para], nse_a[para], b[para], nse_b[para])
    mdd_z = (a[end] + log_mdd_offset_a - b[end]) / sqrt(nse_a[end]^2 + nse_b[end]^2)
    time(file) = load(file, "cloud").total_sampling_time
    return Dict{String, Any}("labels" => la[para], "z" => c.z, "share_above_2" => c.share_above_2,
                             "ks_statistic" => c.ks_statistic, "ks_pvalue" => c.ks_pvalue,
                             "log_mdd_a" => a[end] + log_mdd_offset_a, "log_mdd_b" => b[end],
                             "log_mdd_z" => mdd_z, "time_a" => time(file_a), "time_b" => time(file_b))
end

"""
```
smc_sbc(m, n_periods; n_reps = 200, n_draws = 99, seed = 0, reps = 1:n_reps, smc_kwargs...)
```
Item 5, sampler validation: simulation-based calibration (Cook, Gelman and Rubin 2006; Talts et
al. 2018). For each replication r in `reps`: draw θ from the prior (redrawing until the model
solves), simulate `n_periods` of observables, run `SMC.smc` on them with `m`'s SMC settings
(overridden by `smc_kwargs`), and record for each free parameter the rank of the true value
among `n_draws` posterior draws resampled from the cloud. The data are simulated from the model's
state space with its measurement error (`EE`). Replication r uses seeds derived
from (`seed`, r) only, so `reps` can be split across jobs and the ranks concatenated.

Returns the ranks (length(reps) × free parameters) and the parameter names. Under a correct
sampler each column is uniform on 0:n_draws; see `smc_sbc_uniformity`. The draws are resampled
particles, not independent draws, so small departures can come from their dependence.
"""
function smc_sbc(m::AbstractDSGEModel, n_periods::Int; n_reps::Int = 200, n_draws::Int = 99,
                 seed::Integer = 0, reps = 1:n_reps, burnin::Int = 100, smc_kwargs...)
    free = ModelConstructors.get_free_para_inds(get_parameters(m))
    loglik = function (parameters, data)
        update!(m, parameters)
        likelihood(m, data; sampler = false, catch_errors = true)
    end
    ranks = zeros(Int, length(reps), length(free))
    for (row, r) in enumerate(reps)
        Random.seed!(SMC.stream_seed(UInt64(seed), :sbc_truth, 0, r))
        θ_true, data = nothing, nothing
        while data === nothing
            θ_true = vec(rand(get_parameters(m), 1))
            try
                DSGE.update!(m, θ_true)
                data = simulate_from_model(m, burnin, n_periods)
                isfinite(loglik(get_parameters(m), data)) || (data = nothing)
            catch err
                (err isa GensysError || err isa ParamBoundsError || err isa DomainError) || rethrow(err)
                data = nothing
            end
        end
        cloud = SMC.smc(loglik, get_parameters(m), data; verbose = :none, testing = true,
                        smc_seed = SMC.stream_seed(UInt64(seed), :sbc_smc, 0, r),
                        sbc_smc_settings(m)..., smc_kwargs...)
        rng = Xoshiro(SMC.stream_seed(UInt64(seed), :sbc_draws, 0, r))
        draws = sample(rng, 1:length(cloud), Weights(SMC.get_weights(cloud)), n_draws)
        for (col, k) in enumerate(free)
            ranks[row, col] = count(<(θ_true[k]), cloud.particles[draws, k])
        end
    end
    return ranks, [p.key for p in get_parameters(m)][free]
end

# Observables simulated from the state space the likelihood uses, measurement error (EE)
# included (simulate_observables adds a fixed 1e-3 noise instead)
function simulate_from_model(m::AbstractDSGEModel, burnin::Int, n_periods::Int)
    system = compute_system(m)
    s, _ = simulate_states(system[:TTT], system[:RRR], system[:CCC], system[:QQ];
                           burnin = burnin, n_periods = n_periods)
    EE = system[:EE]
    n_obs = length(system[:DD])
    noise = iszero(EE) ? zeros(n_obs, n_periods) : cholesky(Symmetric(EE)).L * randn(n_obs, n_periods)
    return system[:DD] .+ system[:ZZ] * s[:, burnin+1:burnin+n_periods] .+ noise
end

# m's SMC settings as SMC.smc keywords
sbc_smc_settings(m::AbstractDSGEModel) =
    (n_parts = get_setting(m, :n_particles), n_blocks = get_setting(m, :n_smc_blocks),
     n_mh_steps = get_setting(m, :n_mh_steps_smc), λ = get_setting(m, :λ), n_Φ = get_setting(m, :n_Φ),
     resampling_method = get_setting(m, :resampler_smc), threshold_ratio = get_setting(m, :resampling_threshold),
     c = get_setting(m, :step_size_smc), α = get_setting(m, :mixture_proportion),
     target = get_setting(m, :target_accept),
     tempering_target = get_setting(m, :adaptive_tempering_target_smc),
     use_fixed_schedule = get_setting(m, :use_fixed_schedule))

"""
```
smc_sbc_uniformity(ranks, n_draws; n_bins = 10)
```
χ² test of uniformity of each column of SBC ranks (on 0:n_draws) over `n_bins` bins of rank
values. Returns the p-values, one per parameter.
"""
function smc_sbc_uniformity(ranks::AbstractMatrix{<:Integer}, n_draws::Int; n_bins::Int = 10)
    R = size(ranks, 1)
    bin(x) = 1 + floor(Int, x * n_bins / (n_draws + 1))
    expected = [count(x -> bin(x) == b, 0:n_draws) for b in 1:n_bins] .* (R / (n_draws + 1))
    map(axes(ranks, 2)) do k
        counts = [count(==(b), bin.(ranks[:, k])) for b in 1:n_bins]
        ccdf(Chisq(n_bins - 1), sum((counts .- expected) .^ 2 ./ expected))
    end
end
