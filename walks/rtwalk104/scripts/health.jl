# RTWALK104 per-step health check and gate. Included by step.jl on the master and the workers (after setup.jl).
#
# What it does (modelled on EST-2019Q4-BLK-MH-CHECK.md, work/est2019blk-mh/{kern,mh,analyze}.jl and its port to the 2019Q4
# real-time cloud, work/rtvint104-health/; the MH kernel and the statistics are copied from there):
#   1. stage health of the bridge: φ schedule, ESS path, resamples, acceptance and c per block (ρ_σ_w vs rest), max
#      correlation, distinct particles (SMC's stage_health);
#   2. posterior movement vs the previous origin (the seed cloud): Δmean in seed-sd units, sd ratios, ρ_σ_w quantiles;
#   3. a short blocked MH drift check at φ = 1 on the new data: the final cloud is systematically resampled to N equal
#      weights, then H_PASSES passes of SMC.mutation (the run's kernel, α = 0.975, B1 = {ρ_σ_w}, B2 = rest, proposal =
#      weighted mean/cov of the final cloud, c from the last bridge stage, tuned in H_NTUNE passes then frozen);
#      reports Δlogpost, Δloglh (cluster-robust paired SEs), RMS drift (start sd), n|z|>3, ESJD;
#   4. optional H_POLISH further passes (same kernel); the cloud after all passes is the next step's seed;
#   5. the gate (thresholds below, overridable by env): writes HEALTH_PASS or HEALTH_FAIL (with the reasons) to the
#      step directory, plus health.md, health_stage.csv, health_passes.csv, health_params.csv, cloud_health.jld2, and
#      replaces this step's row in the cumulative summary (rtwalk104_summary.csv next to RTWALK104.md).
#      Since RTWALK104-REVIEW (§5): the MH statistics are gated on the window of the LAST H_PASSES passes (= passes
#      1..H_PASSES without polishing; with H_POLISH = k, passes k+1..k+H_PASSES, ending at the next seed), with the
#      extrapolated gap drift / (1 − ρ̄^H_PASSES) (RMS ≤ 0.10, max ≤ 0.25 sd) instead of the raw RMS drift. Order of
#      writes: old marker removed first; cloud_health.jld2 via .tmp + mv; the new marker last (via .tmp + mv).
using StatsBase, Printf, Random, LinearAlgebra

# ---------------------------------------------------------------- gate thresholds (env H_<NAME> overrides)
const GATE = Dict{Symbol, Float64}(
    :ess_min        => 0.30,   # final-cloud ESS / N (weights after the last stage)
    :acc_rest_lo    => 0.10, :acc_rest_hi => 0.50,      # mean acceptance of the rest block, last 5 bridge stages
    :acc_rsw_lo     => 0.05, :acc_rsw_hi  => 0.90,      # same, ρ_σ_w block
    :distinct_min   => 0.50,   # distinct mutated particles / N, last bridge stage
    :maxcorr_max    => 0.995,  # max pre/post-mutation correlation, last bridge stage (warn above 0.95)
    :mh_acc_lo      => 0.10, :mh_acc_hi => 0.50,        # drift check, B2 acceptance after tuning
    :dlogpost_max   => 1.0,    # |Δ mean logpost| after the drift-check passes (2019Q4 seed: -0.002 after 10 passes)
    :dlogpost_z_max => 5.0,    # ... and |Δ| / paired SE
    # extrapolated distance to equilibrium (RTWALK104-REVIEW §5.2; replaces the old RMS-drift ≤ 0.05 test, which passed
    # gaps up to ≈ 0.65 sd): gap = drift / (1 − ρ̄^npass), ρ̄ = median per-pass autocorrelation of the B2 parameters,
    # drift in start-sd units over the gated window of npass passes (see run_health)
    :gap_rms_max    => 0.10,   # RMS over B2 (2019Q4 seed: ≈ 0.07)
    :gap_max_max    => 0.25,   # worst B2 parameter (2019Q4 seed: ≈ 0.18, ρ_λ_w)
    :nz3_share_max  => 0.33,   # share of B2 parameters with |z| > 3 (2019Q4 seed: 9 of 62 after 10 passes)
    # RTWALK104-STEP1FIX: the extrapolated gap is only informative if its NOISE FLOOR is well below the gate. Noise floor
    # = RMS over B2 of se(drift_j) / relax_j, ≈ sqrt(2 / (N_eff · relax)). Failed step 1: 0.097 (ρ̄ 0.9986), i.e. a
    # perfect cloud would have failed the 0.10 gate about half the time. Gate it at 0.6 × gap_rms_max.
    :gap_noise_max  => 0.06,
    # RTWALK104-STEP1FIX-VERIFY (Marco, 5 Nov 2026): the "covid" group (CAND38 slots freed by the walk; weakly identified,
    # heavy-tailed, moved in LEVELS) gets its own check. The gap / noise-floor / maxcorr / |z|>3 tests above are taken on
    # the ρ_σ_w and rest blocks only (unchanged strictness); for covid they are warnings. Covid fails the step only if it
    # is CLEARLY off: acceptance outside [covid_acc_lo, covid_acc_hi] (the kernel must move at all), more than
    # covid_z4_share_max of its parameters with |z| > 4, or a parameter with |z| > covid_net_zmin AND a net-of-noise
    # extrapolated gap noise_j·sqrt(z_j² − 1) > covid_net_gap_max sd. Failed step 1: no covid |z| > 3 (RMS z 1.20).
    :covid_acc_lo   => 0.05, :covid_acc_hi => 0.90,
    :covid_z4_share_max => 0.25,
    :covid_net_zmin => 3.0,
    :covid_net_gap_max  => 0.50)
# block groups of the drift check = of the bridge (RTWALK104-STEP1FIX: [ρ_σ_w, covid, rest], setup.jl walk_block_labels);
# saved with c_blocks. B1 = {ρ_σ_w}, B2 = every other parameter (all non-ρ_σ_w groups) for the drift statistics.
health_block_labels() = walk_block_labels(m)
for k in collect(keys(GATE))
    haskey(ENV, "H_" * uppercase(string(k))) && (GATE[k] = parse(Float64, ENV["H_" * uppercase(string(k))]))
end

# ---------------------------------------------------------------- the MH kernel (= work/est2019blk-mh/kern.jl)
const HEALTH_DATA = Ref{Matrix{Float64}}()       # new data, set on every process by step.jl
function LIKP(parameters::ParameterVector, data::Matrix{Float64})::Float64   # = DSGE.smc2's _my_likelihood_dsge
    DSGE.update!(m, parameters, regime_switching = REGSW)
    m <= Setting(:preprocessed_transitions, Dict())
    return likelihood(m, data; sampler = false, catch_errors = true, add_zlb_duration = (false, 1),
                      use_chand_recursion = USECHAND, verbose = :none)
end
# RTWALK104-STEP1FIX: a plain Gaussian random walk on log θ for one block (the "covid" σ's, whose level posteriors are
# heavy-tailed but close to Gaussian in logs: excess kurtosis ≤ 1.6, Mahalanobis var ratio 1.7 vs 48 in levels).
# Proposal x' = x + c·L·ε, x = log θ[cols], L = chol of the cloud's weighted cov of log θ[cols]. Target in x:
# π(e^x)·Π e^x, so the log acceptance ratio is Δ(loglh + logprior) + Σ(x' − x). π-invariant at φ = 1; the bridge itself
# keeps SMC's model-space kernel (SMC has no transformed proposal).
function mh_log_step!(p::Vector{Float64}, cols::Vector{Int}, L::Matrix{Float64}, c::Float64)
    N = length(p); para = p[1:N-5]; like = p[N-4]; lpr = p[N-3]
    x = log.(para[cols]); xn = x .+ c .* (L * randn(length(cols)))
    pn = copy(para); pn[cols] = exp.(xn)
    like_n = -Inf; lpr_n = -Inf
    try
        ModelConstructors.update!(m.parameters, pn)
        lpr_n = ModelConstructors.prior(m.parameters)
        like_n = LIKP(m.parameters, HEALTH_DATA[])
        ModelConstructors.toggle_regime!(m.parameters, 1)
    catch err
        ModelConstructors.toggle_regime!(m.parameters, 1)
        (err isa ModelConstructors.ParamBoundsError || err isa LinearAlgebra.LAPACKException || err isa PosDefException ||
         err isa SingularException || err isa DomainError) || rethrow(err)
        like_n = lpr_n = -Inf
    end
    fail = isfinite(like_n) ? 0.0 : 1.0
    η = (like_n + lpr_n) - (like + lpr) + sum(xn .- x)
    if isfinite(like_n) && isfinite(lpr_n) && log(rand()) < η
        p[1:N-5] = pn; p[N-4] = like_n; p[N-3] = lpr_n; p[N-1] = 1.0
        return p, 1.0, fail
    end
    p[N-1] = 0.0
    return p, 0.0, fail
end
# kinds[b] = :mix (SMC.mutation, as before) or :log (mh_log_step! with Ls[b]); the default reproduces the old kernel
function one_particle(p::Vector{Float64}, k::Int, step::Int, μ, Σ, nfree, bf, ba, cs, seed::UInt64,
                      kinds = fill(:mix, length(bf)), Ls = fill(zeros(0, 0), length(bf)))
    Random.seed!(SMC.stream_seed(seed, :mhblk, step, k))
    nb = length(bf); acc = zeros(nb); fail = zeros(nb); dlp = zeros(nb); L = length(p)
    for b in 1:nb
        lp0 = p[L-4] + p[L-3]     # STEP1FIX-VERIFY: Δ(loglh + logprior) made by block b's step (Δlogpost attribution)
        if kinds[b] == :mix
            nf = Ref(0)
            p = SMC.mutation(LIKP, m.parameters, HEALTH_DATA[], p, μ, Σ, nfree, [bf[b]], [ba[b]], 1.0;
                             c = cs[b], α = 0.975, n_mh_steps = 1, regime_switching = REGSW, toggle = true, n_failed = nf)
            acc[b] = p[end-1] > 0 ? 1.0 : 0.0; fail[b] = nf[]
        else
            p, acc[b], fail[b] = mh_log_step!(p, ba[b], Ls[b], cs[b])
        end
        dlp[b] = (p[L-4] + p[L-3]) - lp0
    end
    return vcat(p, acc, fail, dlp)
end

if myid() == 1
using CSV, DataFrames

function proposal_from(P::Matrix{Float64}, FREE)
    θbar = SMC.weighted_mean(P); R = SMC.weighted_cov(P)
    Rf = (R[FREE, FREE] + R[FREE, FREE]') / 2.
    floored = false
    if !isposdef(Rf)
        F = eigen(Symmetric(Rf)); λf = max(maximum(F.values) * 1e-10, eps(Float64))
        Rf = F.vectors * Diagonal(max.(F.values, λf)) * F.vectors'; Rf = (Rf + Rf') / 2.; floored = true
    end
    return θbar[FREE], Rf, floored
end
wmean(X, w) = vec(sum(w .* X, dims = 1)) ./ sum(w)
wsd(X, w) = (μ = wmean(X, w); sqrt.(vec(sum(w .* (X .- μ') .^ 2, dims = 1)) ./ sum(w)))
wquant(x, w, qs) = (o = sortperm(x); cw = cumsum(w[o]) ./ sum(w); [x[o][min(searchsortedfirst(cw, q), length(x))] for q in qs])

"""
run_health(; ...) -> (pass::Bool, summary_row::NamedTuple)
  cloud      final bridge cloud (SMC.Cloud) on the new data;  seed: the step's seed cloud (previous origin, θ̃ redrawn)
  stage_health: the Dict SMC saved;  thetatilde: slots drawn from the prior at this step (excluded from movement stats)
"""
function run_health(; cloud, seed, stage_health, thetatilde, stepdir, info::Dict, cs0::Vector{Float64},
                    npass::Int = 10, ntune::Int = 2, nchunk::Int = 5, polish::Int = 0, hseed::UInt64 = UInt64(20261106),
                    bridge_labels::Union{Nothing, Vector{String}} = nothing)   # STEP1FIX: labels of the bridge's block groups (rows of stage_health c_blocks)
    mkpath(stepdir)
    # remove the old verdict FIRST (RTWALK104-REVIEW §5.4): a run killed before its own verdict leaves no marker, never
    # an old HEALTH_PASS beside a new, unchecked cloud_health.jld2
    for f in ("HEALTH_PASS", "HEALTH_FAIL"); isfile(joinpath(stepdir, f)) && rm(joinpath(stepdir, f)); end
    P0 = cloud.particles; NP, N = size(P0)
    ModelConstructors.toggle_regime!(m.parameters, 1)
    FREE = ModelConstructors.get_free_para_inds(m.parameters; regime_switching = REGSW, toggle = true); NFREE = length(FREE)
    nm(j) = string(SLOTSYMS[FREE[j]])
    fr = findfirst(==(slot_index(:ρ_σ_w, 1)), FREE); B1 = [fr]; B2 = setdiff(1:NFREE, B1)
    # MH block groups (RTWALK104-STEP1FIX): the walk's fixed groups (ρ_σ_w first), then the rest; B2 = all but ρ_σ_w
    HL = health_block_labels(); NB = length(HL)
    BF = [sort([findfirst(==(slot_index_sym(s)), FREE) for s in syms]) for (_, syms) in walk_block_groups(m)]
    @assert BF[1] == B1 && HL[1] == "ρ_σ_w"
    push!(BF, setdiff(1:NFREE, vcat(BF...))); @assert sort(vcat(BF...)) == collect(1:NFREE) && all(!isempty, BF)
    BA = SMC.generate_all_blocks(BF, FREE)
    grp = zeros(Int, NFREE); for (g, b) in enumerate(BF); grp[b] .= g; end
    # STEP1FIX-VERIFY: BC = the covid group (own check, see GATE), BG = the gated B2 parameters (rest block; = B2 without covid)
    GC = findfirst(==("covid"), HL); BC = GC === nothing ? Int[] : BF[GC]; BG = setdiff(B2, BC); @assert !isempty(BG)
    @assert length(cs0) == NB "cs0 $cs0 has $(length(cs0)) scales for the $NB groups $HL"
    NMH = get_setting(m, :n_mh_steps_smc)
    reasons = String[]; warns = String[]
    chk(ok, msg) = (ok || push!(reasons, msg); ok)

    # ---------------- 1. stage health
    h = stage_health; S = length(h["stage"])
    sh = DataFrame(stage = h["stage"], phi = h["phi"], accept = h["accept"], c = h["c"])
    for k in sort(collect(keys(h)))
        k in ("stage", "phi", "accept", "c", "time") && continue
        v = h[k]
        if v isa AbstractVector && length(v) == S
            sh[!, k] = v
        elseif v isa AbstractMatrix && size(v, 2) == S
            for i in 1:size(v, 1); sh[!, "$(k)_$i"] = v[i, :]; end
        end
    end
    CSV.write(joinpath(stepdir, "health_stage.csv"), sh)
    w = P0[:, end]; w = w ./ sum(w); essN = 1 / sum(w .^ 2) / NP
    ll = P0[:, N-4]; lp = P0[:, N-3]
    nonfin_mass = sum(w[.!(isfinite.(ll) .& isfinite.(lp))])
    last5 = max(1, S - 4):S
    # the BRIDGE's groups (rows of accept_blocks / c_blocks; = HL unless HEALTH_REDO on a bridge run with other groups)
    BL = bridge_labels === nothing ? HL : bridge_labels
    haskey(h, "accept_blocks") && @assert size(h["accept_blocks"], 1) == length(BL) "stage health has $(size(h["accept_blocks"], 1)) block groups, labels $BL"
    brow(l) = findfirst(==(l), BL)
    acc_rsw = haskey(h, "accept_blocks") ? mean(h["accept_blocks"][brow("ρ_σ_w"), last5]) : NaN
    acc_rest = haskey(h, "accept_blocks") ? mean(h["accept_blocks"][brow("rest"), last5]) : mean(h["accept"][last5])
    acc_other = haskey(h, "accept_blocks") ? [(l, mean(h["accept_blocks"][brow(l), last5])) for l in BL if !(l in ("ρ_σ_w", "rest"))] : Tuple{String, Float64}[]
    getlast(k) = haskey(h, k) ? h[k][end] : NaN
    cblk(l) = haskey(h, "c_blocks") ? h["c_blocks"][brow(l), end] : h["c"][end]
    lastfin(k) = (haskey(h, k) && (v = Base.filter(x -> x isa Real && isfinite(x), h[k]); !isempty(v)) ? v[end] : NaN)   # last finite value
    distinct = lastfin("distinct_mutated")
    # rows of h["corr"] = free parameters. STEP1FIX-VERIFY: the gated max correlation excludes the covid group (BC);
    # the covid group's own last-stage max is reported (maxcorr_covid) and only warned about
    maxcorr, maxcorr_para, maxcorr_covid = if haskey(h, "corr") && h["corr"] isa AbstractMatrix && any(isfinite, h["corr"][:, end])
        cc = h["corr"][:, end]; cc = map(x -> isfinite(x) ? x : -Inf, cc)
        @assert length(cc) == NFREE "stage-health corr has $(length(cc)) rows for $NFREE free parameters"
        ng = setdiff(1:NFREE, BC); k = ng[argmax(cc[ng])]
        (cc[k], nm(k), isempty(BC) ? NaN : maximum(cc[BC]))
    else
        (lastfin("max_corr"), "?", NaN)
    end
    !isnan(maxcorr_covid) && maxcorr_covid > 0.95 && push!(warns, @sprintf("covid group: bridge last-stage max correlation %.4f (not gated)", maxcorr_covid))
    nres = haskey(h, "resampled") ? Int(sum(h["resampled"])) : -1
    chk(cloud.tempering_schedule[end] == 1.0, "φ_end = $(cloud.tempering_schedule[end]) < 1")
    chk(!any(isnan, P0), "NaN in the final cloud")
    chk(nonfin_mass == 0, "weight $(nonfin_mass) on particles with non-finite loglh/logprior")
    chk(essN >= GATE[:ess_min], @sprintf("final ESS/N %.3f < %.2f", essN, GATE[:ess_min]))
    chk(GATE[:acc_rest_lo] <= acc_rest <= GATE[:acc_rest_hi], @sprintf("rest-block acceptance (last 5 stages) %.3f outside [%.2f, %.2f]", acc_rest, GATE[:acc_rest_lo], GATE[:acc_rest_hi]))
    isnan(acc_rsw) || chk(GATE[:acc_rsw_lo] <= acc_rsw <= GATE[:acc_rsw_hi], @sprintf("ρ_σ_w acceptance (last 5 stages) %.3f outside [%.2f, %.2f]", acc_rsw, GATE[:acc_rsw_lo], GATE[:acc_rsw_hi]))
    for (l, a) in acc_other     # STEP1FIX: every further group; covid in its own range (STEP1FIX-VERIFY)
        lo, hi = l == "covid" ? (GATE[:covid_acc_lo], GATE[:covid_acc_hi]) : (GATE[:acc_rest_lo], GATE[:acc_rest_hi])
        chk(lo <= a <= hi, @sprintf("%s-block acceptance (last 5 stages) %.3f outside [%.2f, %.2f]", l, a, lo, hi))
    end
    isnan(distinct) || chk(distinct >= GATE[:distinct_min], @sprintf("distinct mutated %.3f < %.2f", distinct, GATE[:distinct_min]))
    # max correlation (STEP1FIX): without polishing it is the bridge's last stage (as before). With H_POLISH > 0 the
    # next seed is the POLISHED cloud, so the like-for-like test is on the kernel that produced it: max over B2 of
    # ρ_j^NMH (ρ_j = median per-pass correlation over the gated window; NMH = the bridge's MH steps per stage = 3), gated
    # below after the MH passes; the bridge's last-stage value is then reported as a warning only.
    polish == 0 && !isnan(maxcorr) && chk(maxcorr <= GATE[:maxcorr_max], @sprintf("max correlation %.4f > %.3f", maxcorr, GATE[:maxcorr_max]))
    !isnan(maxcorr) && maxcorr > 0.95 && push!(warns, @sprintf("%smax correlation %.4f (%s) > 0.95", polish > 0 ? "bridge last-stage " : "", maxcorr, maxcorr_para))

    # ---------------- 2. movement vs the previous origin (seed cloud, weighted) on parameters free at both origins
    Ps = seed.particles; ws = Ps[:, end] ./ sum(Ps[:, end])
    tt = Set(slot_index(k, r) for (k, r) in thetatilde)
    keepj = [j for j in 1:NFREE if !(FREE[j] in tt)]
    ms = wmean(Ps[:, FREE], ws); ss = wsd(Ps[:, FREE], ws); mn = wmean(P0[:, FREE], w); sn = wsd(P0[:, FREE], w)
    dmean = (mn .- ms) ./ ss; sdr = sn ./ ss
    jr = fr
    qs = [0.01, 0.05, 0.5, 0.95]
    q_prev = wquant(Ps[:, FREE[jr]], ws, qs); q_new = wquant(P0[:, FREE[jr]], w, qs)

    # ---------------- 3./4. MH drift check (+ polishing) at φ = 1 on the new data
    μf, Σf, fl = proposal_from(P0, FREE)
    # STEP1FIX: kernel kind per group. H_COVID_LOG=1 (opt-in; default 0 = levels, STEP1FIX-VERIFY: Marco 5 Nov): the "covid" group moves by a random walk on log θ
    # (mh_log_step!), starting at scale H_C_LOG0 (default 2.0/√k, k = group size; tuned in the H_NTUNE passes); its
    # saved c_blocks entry stays the BRIDGE's model-space scale (cs0), since the next bridge uses SMC's kernel.
    # Requires every covid slot > 0 in the whole cloud; otherwise (or H_COVID_LOG=0) the group uses SMC.mutation.
    kinds = fill(:mix, NB); Ls = fill(zeros(0, 0), NB)
    gcov = findfirst(==("covid"), HL)
    if gcov !== nothing && get(ENV, "H_COVID_LOG", "0") == "1"
        cols = FREE[BF[gcov]]
        if all(>(0), P0[:, cols])
            LXc = log.(P0[:, cols]); lmu = vec(sum(w .* LXc, dims = 1)); LDc = LXc .- lmu'
            Ls[gcov] = Matrix(cholesky(Symmetric((LDc .* w)' * LDc)).L); kinds[gcov] = :log
        else
            push!(warns, "covid group has non-positive values: kept SMC's model-space kernel for it (H_COVID_LOG ignored)")
        end
    end
    println("  health kernels: ", join(["$(HL[g]) ($(length(BF[g])), $(kinds[g]))" for g in 1:NB], ", "))
    idx = SMC.resample(w; n_parts = NP, method = :systematic, rng = Xoshiro(SMC.stream_seed(hseed, :resample)))
    Pstart = P0[idx, :]; Pstart[:, end] .= 1.0
    X0 = Pstart[:, FREE]; m0 = vec(mean(X0, dims = 1)); s0 = vec(std(X0, dims = 1)); s0[s0 .== 0] .= 1.0
    ll0 = Pstart[:, N-4]; lpost0 = ll0 .+ Pstart[:, N-3]
    function cse(D::AbstractMatrix)          # cluster-robust (by resampling parent) SE of the column means
        n = size(D, 1); Dc = D .- mean(D, dims = 1); Sm = zeros(NP, size(D, 2))
        for i in 1:n; Sm[idx[i], :] .+= view(Dc, i, :); end
        return vec(sqrt.(sum(Sm .^ 2, dims = 1))) ./ n
    end
    cse(v::AbstractVector) = cse(reshape(v, :, 1))[1]
    function mhrows(P, rows, pass, cs)
        μl = μf; Σl = Σf; bf = BF; ba = BA; nf = NFREE; sd = hseed; csl = copy(cs); kl = kinds; Ll = Ls
        res = pmap(r -> one_particle(r[1], r[2], pass, μl, Σl, nf, bf, ba, csl, sd, kl, Ll), [(P[k, :], k) for k in rows]; batch_size = 10)
        M = reduce(hcat, res)'
        return Matrix(M[:, 1:N]), Matrix(M[:, N+1:N+NB]), Matrix(M[:, N+NB+1:N+2NB]), Matrix(M[:, N+2NB+1:N+3NB])
    end
    fmt(v) = join([@sprintf("%.4f", x) for x in v], "/")
    cs = copy(cs0); P = copy(Pstart); PLOG = []; PROWS = []; CORRS = Vector{Vector{Float64}}(); last_acc = fill(NaN, NB)
    DLPS = Vector{Vector{Float64}}()   # STEP1FIX-VERIFY: DLPS[pass][g] = mean Δlogpost made by group g's steps in that pass
    for g in 1:NB; kinds[g] == :log && (cs[g] = haskey(ENV, "H_C_LOG0") ? parse(Float64, ENV["H_C_LOG0"]) : 2.0 / sqrt(length(BF[g]))); end
    # the gated window = the LAST npass passes (RTWALK104-REVIEW §5.3): its start is the cloud after pass `polish`
    # (= Pstart when polish = 0, i.e. the original pass-npass statistics); its end is the cloud that becomes the seed
    Pwin = polish == 0 ? Pstart : nothing
    tH = time()
    for pass in 1:(npass + polish)
        tq = time(); cs_used = copy(cs)
        if pass <= ntune
            Pn = similar(P); A = zeros(NP, NB); Fl = zeros(NP, NB); DL = zeros(NP, NB)
            for ch in Iterators.partition(1:NP, cld(NP, nchunk))
                Pc, Ac, Fc, Dc = mhrows(P, collect(ch), pass, cs)
                Pn[ch, :] = Pc; A[ch, :] = Ac; Fl[ch, :] = Fc; DL[ch, :] = Dc
                cs = cs .* clamp.(exp.(2.5 .* (vec(mean(Ac, dims = 1)) .- 0.25)), 0.5, 2.0)
            end
        else
            Pn, A, Fl, DL = mhrows(P, collect(1:NP), pass, cs)
        end
        X = Pn[:, FREE]; mu = vec(mean(X, dims = 1))
        corr, esjd = SMC.particle_movement(P[:, FREE], X, ones(NP)); push!(CORRS, corr); push!(DLPS, vec(mean(DL, dims = 1)))
        sep = cse(X .- X0); z = [sep[j] > 0 ? (mu[j] - m0[j]) / sep[j] : 0.0 for j in 1:NFREE]
        drift = (mu .- m0) ./ s0
        lln = Pn[:, N-4]; lpn = lln .+ Pn[:, N-3]
        acc = vec(mean(A, dims = 1)); fail = vec(mean(Fl, dims = 1)); last_acc = acc
        # c_rsw / acc_rsw = group 1 (ρ_σ_w), c_rest / acc_rest = the LAST group (rest); all groups in the *_blocks strings
        push!(PLOG, (pass = pass, polish = pass > npass, t_s = time() - tq, c_rsw = cs_used[1], c_rest = cs_used[end],
                     acc_rsw = acc[1], acc_rest = acc[end], inf_rsw = fail[1], inf_rest = fail[end],
                     dloglh = mean(lln .- ll0), se_dloglh = cse(lln .- ll0), dlogpost = mean(lpn .- lpost0), se_dlogpost = cse(lpn .- lpost0),
                     rms_drift_B2 = sqrt(mean(drift[B2] .^ 2)), max_drift_B2 = maximum(abs.(drift[B2])),
                     n_z3_B2 = sum(abs.(z[B2]) .> 3), esjd_B2_med = median(esjd[B2]), corr_B2_med = median(corr[B2]),
                     drift_rsw = drift[jr], z_rsw = z[jr], nonfinite = count(!isfinite, lln),
                     groups = join(HL, "/"), c_blocks = fmt(cs_used), acc_blocks = fmt(acc), acc_min_B2 = minimum(acc[2:end]), acc_max_B2 = maximum(acc[2:end]),
                     esjd_med_blocks = fmt([median(esjd[b]) for b in BF]), corr_med_blocks = fmt([median(corr[b]) for b in BF])))
        r = PLOG[end]
        println(@sprintf("  health pass %2d%s  c %s  acc %s  Δlogpost %+.3f (se %.3f)  Δloglh %+.3f  RMS drift B2 %.4f  #|z|>3 %d  ESJD med %.4g (by group %s)  %.1f s",
                         pass, r.polish ? " (polish)" : "", r.c_blocks, r.acc_blocks, r.dlogpost, r.se_dlogpost, r.dloglh,
                         r.rms_drift_B2, r.n_z3_B2, r.esjd_B2_med, r.esjd_med_blocks, r.t_s)); flush(stdout)
        if pass == npass
            for j in 1:NFREE
                push!(PROWS, (para = nm(j), block = j == jr ? "B1" : "B2", new_at_step = FREE[j] in tt,
                              mean_prev = ms[j], sd_prev = ss[j], mean_new = mn[j], sd_new = sn[j], dmean_prev_sd = dmean[j], sdratio_prev = sdr[j],
                              mh_drift_sd = drift[j], mh_z = z[j], mh_esjd = esjd[j], mh_corr = corr[j], group = HL[grp[j]]))
            end
        end
        P = Pn
        pass == polish && (Pwin = copy(P))
    end
    tH = time() - tH
    CSV.write(joinpath(stepdir, "health_passes.csv"), DataFrame(PLOG))
    CSV.write(joinpath(stepdir, "health_params.csv"), DataFrame(PROWS))
    # ---- gate on the window of the last npass passes (= passes 1..npass when polish = 0; bitwise the PLOG[npass] stats)
    rc = PLOG[end]                                    # acceptance, non-finite, ESJD of the last pass
    gw = let Xa = Pwin[:, FREE], Xe = P[:, FREE]
        ma = vec(mean(Xa, dims = 1)); sa = vec(std(Xa, dims = 1)); sa[sa .== 0] .= 1.0; me = vec(mean(Xe, dims = 1))
        sep = cse(Xe .- Xa); z = [sep[j] > 0 ? (me[j] - ma[j]) / sep[j] : 0.0 for j in 1:NFREE]
        drift = (me .- ma) ./ sa
        dlp = (P[:, N-4] .+ P[:, N-3]) .- (Pwin[:, N-4] .+ Pwin[:, N-3])
        ρ̄ = median([r.corr_B2_med for r in PLOG[polish+1:end]])         # per-pass autocorrelation of the B2 parameters (reported)
        relax = 1 - abs(ρ̄)^npass                                          # share of an equilibrium gap the npass passes remove
        # RTWALK104-STEP1FIX: with several block groups the parameters relax at different rates, so the gap is formed per
        # parameter: gap_j = |drift_j| / (1 − |ρ_j|^npass), ρ_j = median over the window of parameter j's per-pass
        # correlation (identical to the ρ̄ formula when all ρ_j are equal), and its noise floor se_j / sa_j / relax_j
        Cw = reduce(hcat, CORRS[polish+1:end]); ρj = vec(median(Cw, dims = 2)); relaxj = 1 .- abs.(ρj) .^ npass
        okr = isfinite.(relaxj) .& (relaxj .> 0)
        gapj = [okr[j] ? abs(drift[j]) / relaxj[j] : Inf for j in 1:NFREE]
        noisej = [okr[j] ? sep[j] / sa[j] / relaxj[j] : Inf for j in 1:NFREE]
        # STEP1FIX-VERIFY: the gated statistics are over BG (B2 without the covid group); covid gets its own (cv_*)
        rms = sqrt(mean(drift[BG] .^ 2)); mx = maximum(abs.(drift[BG]))
        kmax = BG[argmax(gapj[BG])]
        # covid: z_j = gap_j / noise_j, net-of-noise gap_j = noise_j · sqrt(max(0, z_j² − 1)); NaN-safe (NaN counts as off)
        netj = [okr[j] && isfinite(z[j]) ? noisej[j] * sqrt(max(0.0, z[j]^2 - 1)) : Inf for j in 1:NFREE]
        cvoff = [j for j in BC if !(abs(z[j]) <= GATE[:covid_net_zmin]) && !(netj[j] <= GATE[:covid_net_gap_max])]
        cvmax = isempty(BC) ? 0 : BC[argmax(netj[BC])]
        # Δlogpost of the window by block group: Σ over the window passes of each group's mean Δ(loglh + logprior)
        dlpg = vec(sum(reduce(hcat, DLPS[polish+1:end]), dims = 2))
        (dlogpost = mean(dlp), se_dlogpost = cse(dlp), rms_drift_B2 = rms, max_drift_B2 = mx, n_z3_B2 = sum(abs.(z[BG]) .> 3),
         rho_bar = ρ̄, relax = relax, gap_rms = sqrt(mean(gapj[BG] .^ 2)), gap_max = gapj[kmax], gap_max_para = nm(kmax),
         gap_noise = sqrt(mean(noisej[BG] .^ 2)), relax_min = minimum(relaxj[BG]),
         gap_signal = sqrt(max(0.0, mean(gapj[BG] .^ 2) - mean(noisej[BG] .^ 2))),   # RMS gap net of its noise floor (reported, not gated)
         maxcorr_kernel = maximum(abs.(ρj[BG]) .^ NMH), maxcorr_kernel_para = nm(BG[argmax(abs.(ρj[BG]))]),
         gap_rms_median_rho = isfinite(relax) && relax > 0 ? rms / relax : Inf,
         dlogpost_by_group = dlpg,
         cv_n = length(BC), cv_n_z4 = count(j -> !(abs(z[j]) <= 4), BC), cv_off = cvoff,
         cv_net_max = isempty(BC) ? 0.0 : netj[cvmax], cv_net_max_para = isempty(BC) ? "" : nm(cvmax),
         cv_gap_rms = isempty(BC) ? 0.0 : sqrt(mean(gapj[BC] .^ 2)), cv_gap_max = isempty(BC) ? 0.0 : maximum(gapj[BC]),
         cv_noise = isempty(BC) ? 0.0 : sqrt(mean(noisej[BC] .^ 2)), cv_maxcorr_kernel = isempty(BC) ? NaN : maximum(abs.(ρj[BC]) .^ NMH),
         cv_rows = [(para = nm(j), rho = ρj[j], drift_sd = drift[j], z = z[j], gap = gapj[j], noise = noisej[j], net_gap = netj[j]) for j in BC])
    end
    win = polish == 0 ? "passes 1-$npass" : "passes $(polish+1)-$(npass+polish), after $polish polishing passes"
    for g in 2:NB    # STEP1FIX: every non-ρ_σ_w group; covid in its own (wider) range (STEP1FIX-VERIFY)
        lo, hi = g == GC ? (GATE[:covid_acc_lo], GATE[:covid_acc_hi]) : (GATE[:mh_acc_lo], GATE[:mh_acc_hi])
        chk(lo <= last_acc[g] <= hi, @sprintf("drift-check %s acceptance %.3f outside [%.2f, %.2f]", HL[g], last_acc[g], lo, hi))
    end
    polish > 0 && chk(gw.maxcorr_kernel <= GATE[:maxcorr_max], @sprintf("max %d-pass correlation of the polishing kernel %.4f (%s) > %.3f over %s", NMH, gw.maxcorr_kernel, gw.maxcorr_kernel_para, GATE[:maxcorr_max], win))
    dlp_by = join([@sprintf("%s %+.3f", HL[g], gw.dlogpost_by_group[g]) for g in 1:NB], ", ")
    chk(abs(gw.dlogpost) <= GATE[:dlogpost_max], @sprintf("|Δlogpost| %.3f > %.2f over %s (made by block: %s)", abs(gw.dlogpost), GATE[:dlogpost_max], win, dlp_by))
    chk(abs(gw.dlogpost) <= GATE[:dlogpost_z_max] * max(gw.se_dlogpost, 1e-12), @sprintf("|Δlogpost| / SE = %.1f > %.1f over %s (made by block: %s)", abs(gw.dlogpost) / gw.se_dlogpost, GATE[:dlogpost_z_max], win, dlp_by))
    # covid group (STEP1FIX-VERIFY): hard only if CLEARLY off; slow-mixing statistics are warnings
    if !isempty(BC)
        chk(gw.cv_n_z4 <= GATE[:covid_z4_share_max] * gw.cv_n, @sprintf("covid group clearly off: %d of %d parameters with |z| > 4 over %s", gw.cv_n_z4, gw.cv_n, win))
        chk(isempty(gw.cv_off), @sprintf("covid group clearly off: %s with |z| > %.1f and net-of-noise extrapolated gap > %.2f sd over %s", join(nm.(gw.cv_off), ", "), GATE[:covid_net_zmin], GATE[:covid_net_gap_max], win))
        gw.cv_noise > GATE[:gap_noise_max] && push!(warns, @sprintf("covid group (not gated): noise floor %.3f sd > %.2f (mixes too slowly in levels to certify its gap)", gw.cv_noise, GATE[:gap_noise_max]))
        gw.cv_gap_rms > GATE[:gap_rms_max] && push!(warns, @sprintf("covid group (not gated): extrapolated RMS gap %.3f sd > %.2f", gw.cv_gap_rms, GATE[:gap_rms_max]))
        gw.cv_gap_max > GATE[:gap_max_max] && push!(warns, @sprintf("covid group (not gated): extrapolated max gap %.3f sd > %.2f", gw.cv_gap_max, GATE[:gap_max_max]))
        polish > 0 && gw.cv_maxcorr_kernel > GATE[:maxcorr_max] && push!(warns, @sprintf("covid group (not gated): max %d-pass kernel correlation %.4f > %.3f", NMH, gw.cv_maxcorr_kernel, GATE[:maxcorr_max]))
        println(@sprintf("  covid group (%d): #|z|>4 %d, max net gap %.3f sd (%s), gap RMS %.3f / max %.3f sd, noise floor %.3f, kernel max %d-pass corr %.4f, acc %.3f; window Δlogpost by block: %s",
                         gw.cv_n, gw.cv_n_z4, gw.cv_net_max, gw.cv_net_max_para, gw.cv_gap_rms, gw.cv_gap_max, gw.cv_noise, NMH, gw.cv_maxcorr_kernel, last_acc[GC], dlp_by))
    end
    chk(gw.gap_noise <= GATE[:gap_noise_max], @sprintf("MH too slow to certify the cloud: noise floor of the extrapolated gap %.3f sd > %.2f (min relaxation %.4f) over %s", gw.gap_noise, GATE[:gap_noise_max], gw.relax_min, win))
    chk(gw.gap_rms <= GATE[:gap_rms_max], @sprintf("extrapolated RMS gap %.3f sd > %.2f (per-parameter ρ_j; RMS drift %.4f, ρ̄ %.4f, noise floor %.3f) over %s", gw.gap_rms, GATE[:gap_rms_max], gw.rms_drift_B2, gw.rho_bar, gw.gap_noise, win))
    chk(gw.gap_max <= GATE[:gap_max_max], @sprintf("extrapolated max gap %.3f sd (%s) > %.2f (max drift %.4f) over %s", gw.gap_max, gw.gap_max_para, GATE[:gap_max_max], gw.max_drift_B2, win))
    chk(gw.n_z3_B2 <= GATE[:nz3_share_max] * length(BG), @sprintf("%d of %d rest-block parameters with |z| > 3 over %s", gw.n_z3_B2, length(BG), win))
    chk(rc.nonfinite == 0 && rc.esjd_B2_med > 0, "drift check: non-finite loglh or frozen particles")
    println(@sprintf("  gated window (%s): Δlogpost %+.3f (se %.3f), RMS drift %.4f, max drift %.4f, ρ̄ %.4f, relaxation %.4f (min per parameter %.4f), gap RMS %.3f / max %.3f sd (%s), noise floor %.3f, signal part %.3f, kernel max %d-pass corr %.4f (%s), #|z|>3 %d",
                     win, gw.dlogpost, gw.se_dlogpost, gw.rms_drift_B2, gw.max_drift_B2, gw.rho_bar, gw.relax, gw.relax_min, gw.gap_rms, gw.gap_max, gw.gap_max_para, gw.gap_noise, gw.gap_signal, NMH, gw.maxcorr_kernel, gw.maxcorr_kernel_para, gw.n_z3_B2))
    pass = isempty(reasons)

    # ---------------- outputs: the next seed, md, marker, cumulative summary
    cs_save = [kinds[g] == :log ? cs0[g] : cs[g] for g in 1:NB]   # STEP1FIX: model-space scales for the next bridge
    cl = deepcopy(cloud); cl.particles = P; cl.particles[:, N-2] .= 0.0; cl.ESS = [Float64(NP)]; cl.c = cs_save[end]
    # written atomically (.tmp, then mv; RTWALK104-REVIEW §5.4); c_block_labels name the groups of c_blocks (§3)
    tmp = joinpath(stepdir, "cloud_health.jld2.tmp")
    jldsave(tmp; cloud = cl, c_blocks = cs_save, c_block_labels = HL, c_blocks_health = cs, health_kernels = string.(kinds), passes = npass + polish, idx = idx,
            note = "RTWALK104 $(info[:old]) -> $(info[:new]): final bridge cloud, systematic resample to $NP equal weights, + $(npass + polish) blocked MH passes at φ = 1 on the new data; old_loglh zeroed")
    mv(tmp, joinpath(stepdir, "cloud_health.jld2"); force = true)
    o = sortperm(abs.(dmean[keepj]), rev = true)
    row = (step = info[:step], old = info[:old], new = info[:new], vint_new = info[:vint_new], n_part = NP, n_free = NFREE, n_new_slots = length(thetatilde),
           stages = S, resamples = nres, ess_N = essN, acc_rest = acc_rest, acc_rsw = acc_rsw, c_rest_end = cblk("rest"), c_rsw_end = cblk("ρ_σ_w"),
           bridge_groups = join(BL, "/"), acc_other = join([@sprintf("%s %.3f", l, a) for (l, a) in acc_other], "/"), health_groups = join(HL, "/"), c_blocks_seed = join([@sprintf("%.4f", x) for x in cs], "/"),
           max_corr = maxcorr, log_mdd_incr = get(info, :log_mdd_incr, NaN),
           mh_passes = npass, polish = polish, dlogpost = gw.dlogpost, se_dlogpost = gw.se_dlogpost, rms_drift = gw.rms_drift_B2, n_z3 = gw.n_z3_B2,
           max_drift = gw.max_drift_B2, rho_bar = gw.rho_bar, gap_rms = gw.gap_rms, gap_max = gw.gap_max, gap_noise = gw.gap_noise, gap_signal = gw.gap_signal, maxcorr_kernel = gw.maxcorr_kernel,
           dlogpost_by_group = join([@sprintf("%.4f", x) for x in gw.dlogpost_by_group], "/"), covid_n = gw.cv_n, covid_n_z4 = gw.cv_n_z4, covid_n_off = length(gw.cv_off), covid_net_gap_max = gw.cv_net_max,
           covid_gap_rms = gw.cv_gap_rms, covid_gap_max = gw.cv_gap_max, covid_noise = gw.cv_noise, covid_maxcorr_kernel = gw.cv_maxcorr_kernel, covid_maxcorr_bridge = maxcorr_covid,
           esjd_med = rc.esjd_B2_med, max_abs_dmean_prev = isempty(o) ? NaN : abs(dmean[keepj][o[1]]),
           max_dmean_para = isempty(o) ? "" : nm(keepj[o[1]]), rsw_q05_prev = q_prev[2], rsw_q05_new = q_new[2],
           bridge_wall_h = get(info, :bridge_wall_h, NaN), health_wall_h = tH / 3600, slots = nprocs(), verdict = pass ? "PASS" : "FAIL")
    open(joinpath(stepdir, "health.md"), "w") do io
        println(io, "# RTWALK104 step $(info[:step]): $(info[:old]) -> $(info[:new]) (vintage $(info[:vint_old]) -> $(info[:vint_new]))\n")
        println(io, "**Verdict: $(pass ? "PASS" : "FAIL")**", pass ? "" : " — " * join(reasons, "; "), "\n")
        isempty(warns) || println(io, "Warnings: ", join(warns, "; "), "\n")
        println(io, "Seed: `$(info[:seed])`. Bridge cloud: `$(info[:cloudfile])`. New parameter-regimes drawn from the prior: ",
                isempty(thetatilde) ? "none" : join(["$(k)[$r]" for (k, r) in thetatilde], ", "), ".\n")
        println(io, "## Gate thresholds\n"); for (k, v) in sort(collect(GATE)); println(io, "- `$k` = $v"); end
        println(io, "\n## Bridge stage health\n")
        println(io, @sprintf("- %d stages, %d resamples, final ESS/N %.3f, weight on non-finite particles %.3g", S, nres, essN, nonfin_mass))
        println(io, @sprintf("- block groups %s; acceptance (mean of last 5 stages): rest %.3f, ρ_σ_w %.3f%s; c at the end: %s", join(BL, ", "), acc_rest, acc_rsw, join([@sprintf(", %s %.3f", l, a) for (l, a) in acc_other]), join(["$l $(round(cblk(l), digits = 4))" for l in BL], ", ")))
        println(io, @sprintf("- last stage: distinct mutated %.3f, max correlation %.4f (%s)", distinct, maxcorr, maxcorr_para))
        println(io, "- φ path: ", join([@sprintf("%.4g", x) for x in h["phi"]], " "), "\n- full table: `health_stage.csv`")
        println(io, "\n## Movement vs the previous origin (seed cloud; parameters free at both origins)\n")
        println(io, "| parameter | mean prev | mean new | Δmean (prev sd) | sd ratio |\n|---|---|---|---|---|")
        for k in o[1:min(15, end)]
            j = keepj[k]; println(io, @sprintf("| %s | %.4g | %.4g | %+.2f | %.3f |", nm(j), ms[j], mn[j], dmean[j], sdr[j]))
        end
        println(io, @sprintf("\nρ_σ_w quantiles (1, 5, 50, 95%%): prev %s, new %s", join([@sprintf("%.4f", x) for x in q_prev], " "), join([@sprintf("%.4f", x) for x in q_new], " ")))
        println(io, @sprintf("median |Δmean| %.3f prev-sd; median sd ratio %.3f", median(abs.(dmean[keepj])), median(sdr[keepj])))
        println(io, "\n## MH drift check at φ = 1 ($npass passes, $ntune tuning, then frozen; $polish polishing passes)\n")
        println(io, "Groups: ", join(HL, " / "), "\n")
        println(io, "| pass | c by group | acc by group | Δlogpost (SE) | Δloglh (SE) | RMS drift B2 | max drift B2 | #abs(z)>3 | ESJD med (by group) |\n|---|---|---|---|---|---|---|---|---|")
        for r in PLOG
            println(io, @sprintf("| %d%s | %s | %s | %+.3f (%.3f) | %+.3f (%.3f) | %.4f | %.4f | %d | %.4g (%s) |", r.pass, r.polish ? "p" : "", r.c_blocks,
                                 r.acc_blocks, r.dlogpost, r.se_dlogpost, r.dloglh, r.se_dloglh, r.rms_drift_B2, r.max_drift_B2, r.n_z3_B2, r.esjd_B2_med, r.esjd_med_blocks))
        end
        println(io, @sprintf("\n**Gated window** (%s; the last %d passes, ending at the cloud that becomes the next seed): Δlogpost %+.3f (SE %.3f), RMS / max drift %.4f / %.4f start-sd, ρ̄ %.4f (relaxation 1 − ρ̄^%d = %.4f; per parameter min %.4f), **extrapolated gap (per-parameter ρ_j) RMS %.3f / max %.3f sd (%s)** (gates %.2f / %.2f; median-ρ̄ formula: RMS %.3f), **noise floor %.3f** (gate %.2f), max %d-pass kernel correlation %.4f (%s), #abs(z)>3 %d of %d.",
                             win, npass, gw.dlogpost, gw.se_dlogpost, gw.rms_drift_B2, gw.max_drift_B2, gw.rho_bar, npass, gw.relax, gw.relax_min, gw.gap_rms, gw.gap_max, gw.gap_max_para,
                             GATE[:gap_rms_max], GATE[:gap_max_max], gw.gap_rms_median_rho, gw.gap_noise, GATE[:gap_noise_max], NMH, gw.maxcorr_kernel, gw.maxcorr_kernel_para, gw.n_z3_B2, length(BG)))
        if !isempty(BC)      # STEP1FIX-VERIFY: the covid group's own check
            println(io, @sprintf("\n## Covid group (own check, %d parameters, %s kernel)\n", gw.cv_n, string(kinds[GC])))
            println(io, @sprintf("Hard only if clearly off: acceptance in [%.2f, %.2f] (last pass %.3f); #|z|>4 %d of %d (max share %.2f); parameters with |z| > %.1f and net-of-noise gap > %.2f sd: %s. Not gated (warnings): gap RMS %.3f / max %.3f sd, noise floor %.3f, max %d-pass kernel correlation %.4f, bridge last-stage max correlation %.4f. Window Δlogpost by block: %s.\n",
                                 GATE[:covid_acc_lo], GATE[:covid_acc_hi], last_acc[GC], gw.cv_n_z4, gw.cv_n, GATE[:covid_z4_share_max], GATE[:covid_net_zmin], GATE[:covid_net_gap_max],
                                 isempty(gw.cv_off) ? "none" : join(nm.(gw.cv_off), ", "), gw.cv_gap_rms, gw.cv_gap_max, gw.cv_noise, NMH, gw.cv_maxcorr_kernel, maxcorr_covid, dlp_by))
            println(io, "| parameter | ρ_j per pass | drift (start sd) | z | extrapolated gap | noise floor | net-of-noise gap |\n|---|---|---|---|---|---|---|")
            for r in gw.cv_rows
                println(io, @sprintf("| %s | %.5f | %+.4f | %+.2f | %.3f | %.3f | %.3f |", r.para, r.rho, r.drift_sd, r.z, r.gap, r.noise, r.net_gap))
            end
        end
        println(io, @sprintf("\nHealth check: %.2f h wall on %d processes. Next seed: `cloud_health.jld2` (c_blocks %s = %s; health kernels %s, final health scales %s).", tH / 3600, nprocs(), string(HL), string(round.(cs_save, digits = 4)), string(string.(kinds)), string(round.(cs, digits = 4))))
    end
    sumf = info[:summary_csv]
    df = isfile(sumf) ? CSV.read(sumf, DataFrame; stringtype = String) : DataFrame()
    df = nrow(df) == 0 ? DataFrame([row]) : vcat(df[string.(df.new) .!= info[:new], :], DataFrame([row]); cols = :union)
    sort!(df, :step); CSV.write(sumf, df)
    # the verdict marker LAST, atomically (the old one was removed at the start of run_health)
    mk = joinpath(stepdir, pass ? "HEALTH_PASS" : "HEALTH_FAIL")
    open(mk * ".tmp", "w") do io
        println(io, now(), "  ", info[:old], " -> ", info[:new]); foreach(r -> println(io, r), reasons); foreach(r -> println(io, "warn: ", r), warns)
    end
    mv(mk * ".tmp", mk; force = true)
    println("HEALTH ", pass ? "PASS" : "FAIL", isempty(reasons) ? "" : ": " * join(reasons, "; "))
    return pass, row
end
end # myid() == 1
