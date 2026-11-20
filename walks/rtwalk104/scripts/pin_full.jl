# RTWALK104 FINAL STEP (13 Nov 2026): build the full-sample pin and run the guards (SGE only; NW workers for G6).
# Modelled on work/rtwalk114/scripts/pin_full.jl. See full_data.jl.
#   G0 walk version guards (setup.jl check_versions: DSGE 33ebacbb7 + pinned trunc/reuse patch, SMC 270e283 + pinned
#      initial_c_blocks patch, SSR 26929ea, ModelConstructors 232e4b0) -- they run on include, here re-run explicitly
#   G1 pin === ss104v3 frozen 28 x 268 (sha256-checked) === ss104v2 reference matrix (NaN-aware); regime 28 = 2026Q2
#   G2 walk loglh at the defaults (all 101 free; ss104v2/v3 lik!), trunc on and off, === -6177.968200893426
#   G3 stored loglh AND logprior of 8 particles each of the ss104v3 final cloud, the ss104v3 health cloud and the
#      ss104v3mh5 stage-650 checkpoint recompute on the pin with the walk model (loglh: bitwise expected, rtol 1e-12
#      accepted and reported; logprior: same prior = the MDD comparison is like for like)
#   G4 the 2 slots first free at 2026Q2 (fixed_at("2026Q1")) = σ_condgdp[2], σ_condcorepce[2]: perturbed to prior
#      draws at 3 bases x 3 reps; ΔL on the full data (reported; expected 0: conditional-forecast regime only)
#   G5 truncated measurement solve === full solve on the full data, 8 θ (ss104v3 cloud)
#   G6 projection: Δ = log p(Y_full|θ) − log p(Y_2026Q1|θ) on NPRE particles of the step-25 cloud_health (θ̃ redrawn
#      from the prior), static adaptive schedule (= preflight.jl PART B); calibration: the same for step 25 (seed =
#      step-24 cloud_health, data 2025Q4 -> 2026Q1; actual 8 stages)
#   I  revisions 260710 pin -> full data (first 267 columns)
using Distributed
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk104/scripts/setup.jl")
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk104/scripts/rt_data.jl")
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk104/scripts/full_data.jl")
using Printf, Statistics, Random
const LDEF = -6177.968200893426
fails = String[]
chk(c, msg) = (println(c ? "  PASS  " : "  FAIL  ", msg); c || push!(fails, msg))

println("== G0 versions"); check_versions(verbose = true); chk(true, "walk version guards (setup.jl EXPECTED)")

println("== G1 data")
dV3, shV3, oW = full_frozen_checked(m)
if !isfile(FULL_PIN)
    tmp = FULL_PIN * ".tmp$(getpid())"
    jldsave(tmp; data = dV3, vint = FULL_V, sample_end = "2026Q2", source = V3_FROZEN, sha_ss104v3_frozen = shV3,
            ss104v2_ref = V2_FROZEN_REF, observables = string.(oW), built_at = string(now()))
    mv(tmp, FULL_PIN)
    open(FULL_PIN_SHA, "w") do io; println(io, bytes2hex(open(sha256, FULL_PIN)), "  ", basename(FULL_PIN)); end
    println("  pinned ", FULL_PIN)
end
dF, _ = full_data_pinned(m)
chk(isequal(dF, dV3) && size(dF) == (28, 268), "pin === ss104v3 frozen 28 x 268 === ss104v2 ref (sha256 $shV3)")
dts = DSGE.quarter_range(date_presample_start(m), FULL_T)
println("  dates ", first(dts), " .. ", last(dts), ";  last column (2026Q2) non-NaN rows: ", [string(oW[i]) for i in 1:28 if !isnan(dF[i, end])])
nr = get_setting(m, :n_regimes); rd = get_setting(m, :regime_dates)
println("  n_regimes ", nr, ";  regime dates 26..29: ", [(r, get(rd, r, missing)) for r in 26:29])
chk(try nr == 29 && Dates.lastdayofquarter(rd[28]) == FULL_T && Dates.lastdayofquarter(rd[29]) == Date(2026, 9, 30) catch; false end,
    "regime 28 = 2026Q2 (last in-sample quarter), regime 29 = 2026Q3 (forecast)")
println("  in-sample regime count at 2026Q2 (k, 0 = full solve): ", DSGE.insample_regime_count(m, FULL_T; tvis = haskey(DSGE.get_settings(m), :tvis_information_set), compute_pseudo = false))

println("== G2 defaults")
configure_freeset!(m, Set{Tuple{Symbol, Int}}())
chk(nfree() == 101, "all 101 parameter-regimes free (= ss104v3)")
l_on = lik!(m, VALS0, dF)
m <= Setting(:likelihood_truncated_regimes, false); l_off = lik!(m, VALS0, dF); m <= Setting(:likelihood_truncated_regimes, true)
chk(l_on === LDEF && l_off === LDEF, "loglh at defaults: trunc on $(repr(l_on)), off $(repr(l_off)), ss104v2/v3 $(repr(LDEF))")

println("== G3 stored loglh / logprior")
const RAWV3 = "/data/dsge_data_dir/estimation-AI/work/ss104v3/saveroot/output_data/m1002/ss104/estimate/raw"
const CV3  = joinpath(RAWV3, "smc_cloud_fcastdate=2026Q3_mhsteps=2_nparts=15000_phi=0.95_ss104v3_vint=261013.jld2")
const CV3H = "/data/dsge_data_dir/estimation-AI/work/ss104v3/health/cloud_health.jld2"
const CMH5S = "/data/dsge_data_dir/estimation-AI/work/ss104v3mh5/saveroot/output_data/m1002/ss104/estimate/raw/smc_cloud_fcastdate=2026Q3_mhsteps=5_nparts=15000_phi=0.95_ss104v3mh5_vint=261013_stage=650.jld2"
function prior_of(θ)
    ModelConstructors.update!(m.parameters, Vector{Float64}(θ)); lp = ModelConstructors.prior(m.parameters)
    ModelConstructors.toggle_regime!(m.parameters, 1); return lp
end
nbit = 0; ntot = 0
for (lab, f) in (("ss104v3 final", CV3), ("ss104v3 health cloud", CV3H), ("ss104v3mh5 stage 650", CMH5S))
    global nbit, ntot
    P = load(f, "cloud").particles; n = size(P, 2)
    @assert n - 5 == length(VALS0)
    idx = unique(vcat(argmax(P[:, n-4]), 1, rand(MersenneTwister(13), 1:size(P, 1), 6)))
    rec = [LIK(m, P[i, 1:n-5], dF) for i in idx]; d = rec .- P[idx, n-4]
    nbit += count(i -> rec[i] === P[idx[i], n-4], eachindex(idx)); ntot += length(idx)
    dp = [prior_of(P[i, 1:n-5]) - P[i, n-3] for i in idx]
    chk(all(isfinite, d) && maximum(abs.(d)) <= 1e-12 * 7000, @sprintf("%s: stored loglh at %d particles, bitwise %d, max |Δ| %.3g", lab, length(idx), count(i -> rec[i] === P[idx[i], n-4], eachindex(idx)), maximum(abs.(d))))
    chk(maximum(abs.(dp)) <= 1e-9, @sprintf("%s: stored logprior = walk prior, max |Δ| %.3g", lab, maximum(abs.(dp))))
end
println("  bitwise loglh: $nbit of $ntot")

println("== G4 slots first free at 2026Q2")
fx = fixed_at("2026Q1")
println("  fixed_at(2026Q1) = ", fx, ";  freeset keys: ", first(sort(collect(keys(load(FREESET_PIN, "fixed_at"))))), " .. ", last(sort(collect(keys(load(FREESET_PIN, "fixed_at"))))))
chk(Set(fx) == Set([(:σ_condgdp, 2), (:σ_condcorepce, 2)]), "fixed at 2026Q1 = σ_condgdp[2], σ_condcorepce[2]")
m2p = get_setting(m, :model2para_regime)
for k in (:σ_condgdp, :σ_condcorepce); println("  model2para_regime[$k]: ", haskey(m2p, k) ? sort(collect(m2p[k])) : "none"); end
Pm = load(CV3, "cloud").particles; nm_ = size(Pm, 2)
bases = [VALS0, Pm[argmax(Pm[:, nm_-4]), 1:nm_-5], Pm[7, 1:nm_-5]]
sl = [slot_index(k, r) for (k, r) in fx]
dmax = 0.0
for (b, θ0) in enumerate(bases), rep in 1:3
    global dmax
    θ = copy(θ0); Random.seed!(1000b + rep); dr = ModelConstructors.rand_regime_switching(m.parameters); ModelConstructors.toggle_regime!(m.parameters, 1)
    θ[sl] = dr[sl]
    l0 = b == 1 ? lik!(m, θ0, dF) : LIK(m, θ0, dF); l1 = b == 1 ? lik!(m, θ, dF) : LIK(m, θ, dF)
    println(@sprintf("  base %d rep %d: %s -> %s: ΔL %.3g", b, rep, string(round.(θ0[sl], digits = 4)), string(round.(θ[sl], digits = 4)), l1 - l0))
    dmax = max(dmax, abs(l1 - l0))
end
println(@sprintf("  INFO σ_condgdp[2], σ_condcorepce[2] on log p(Y_2026Q2): max |ΔL| %.3g (ENTER_TOL %g) -> %s", dmax, ENTER_TOL,
                 dmax <= ENTER_TOL ? "do not enter (θ̃ from the prior = their posterior; = ss104v3, where they are free too)" : "ENTER: θ̃ from the prior, the bridge learns them"))

println("== G5 trunc vs full solve")
dd = Float64[]
for i in vcat(argmax(Pm[:, nm_-4]), rand(MersenneTwister(15), 1:size(Pm, 1), 7))
    a = LIK(m, Pm[i, 1:nm_-5], dF)
    m <= Setting(:likelihood_truncated_regimes, false); b = LIK(m, Pm[i, 1:nm_-5], dF); m <= Setting(:likelihood_truncated_regimes, true)
    push!(dd, a === b ? 0.0 : abs(a - b))
end
chk(all(==(0.0), dd), "truncated === full solve at 8 θ on the full data")

println("== I revisions 260710 -> 261013")
d1, v1 = rt_data_pinned(m, Date(2026, 3, 31))
@assert v1 == "260710" && size(d1, 2) == 267
for i in 1:28
    a = d1[i, :]; b = dF[i, 1:267]
    pat = count(isnan.(a) .!= isnan.(b)); both = .!isnan.(a) .& .!isnan.(b)
    nd = count(a[both] .!= b[both]); mx = nd == 0 ? 0.0 : maximum(abs.(a[both] .- b[both]))
    println(@sprintf("  %-26s cells revised %3d (max |Δ| %.4f), NaN-pattern changes %d, 2026Q2 %s", string(oW[i]), nd, mx, pat, isnan(dF[i, end]) ? "NaN" : @sprintf("%.4f", dF[i, end])))
end

println("== G6 projection (static adaptive schedule, target 0.95, resample 0.5)")
ess(w) = sum(w)^2 / sum(w .^ 2)
function simulate_schedule(Δ; target = 0.95, thr = 0.5, rng = MersenneTwister(1))   # = preflight.jl PART B
    n = length(Δ); w = ones(n); d = copy(Δ); d[.!isfinite.(d)] .= -1e300
    φ = 0.0; essprev = ess(w); resampled = true; stages = 0; nres = 0
    while φ < 1.0 && stages < 5000
        stages += 1
        bar = resampled ? target * n : target * essprev
        f(x) = (lw = log.(w) .+ (x - φ) .* d; e = exp.(lw .- maximum(lw)); ess(e) - bar)
        φn = if f(1.0) >= 0
            1.0
        else
            lo, hi = φ, 1.0
            for _ in 1:200; mid = (lo + hi) / 2; f(mid) >= 0 ? (lo = mid) : (hi = mid); end
            lo
        end
        lw = log.(w) .+ (φn - φ) .* d; e = exp.(lw .- maximum(lw)); w = e ./ sum(e) .* n
        essprev = ess(w); φ = φn; resampled = false
        if essprev < thr * n
            cw = cumsum(w) ./ n; u = (rand(rng) .+ (0:n-1)) ./ n
            idx = [min(searchsortedfirst(cw, x), n) for x in u]
            d = d[idx]; w = ones(n); resampled = true; nres += 1; essprev = n
        end
    end
    return stages, nres
end
NPRE = parse(Int, get(ENV, "NPRE", "1000"))
@everywhere begin
    myid() == 1 || configure_freeset!(m, Set{Tuple{Symbol, Int}}())   # all free (master: done in G2): LIK takes the full vector
    const DATA_G6 = Dict("2025Q4" => rt_data_pinned(m, Date(2025, 12, 31))[1], "2026Q1" => rt_data_pinned(m, Date(2026, 3, 31))[1],
                         "2026Q2" => full_data_pinned(m)[1])
    g6lik(θ, lab) = LIK(m, θ, DATA_G6[lab])
end
function proj(seedf, lo, hi, thetatilde, tag)
    P = load(seedf, "cloud").particles; n = size(P, 2); w = P[:, end] ./ sum(P[:, end])
    idx = SMC.resample(w; n_parts = NPRE, method = :systematic, rng = Xoshiro(20261113))
    Θ = P[idx, 1:n-5]; Lold_st = P[idx, n-4]
    sl = [slot_index(k, r) for (k, r) in thetatilde]
    if !isempty(sl)
        for i in 1:NPRE
            Random.seed!(20261113 + i); d = ModelConstructors.rand_regime_switching(m.parameters); Θ[i, sl] = d[sl]
        end
        ModelConstructors.toggle_regime!(m.parameters, 1)
    end
    t0 = time()
    Lnew = pmap(θ -> g6lik(θ, hi), [Θ[i, :] for i in 1:NPRE])
    Lold = pmap(θ -> g6lik(θ, lo), [Θ[i, :] for i in 1:20])
    tt = time() - t0
    Δ = Lnew .- Lold_st; fin = isfinite.(Δ)
    st, nres = simulate_schedule(Δ)
    lw = copy(Δ); lw[.!fin] .= -Inf; e = exp.(lw .- maximum(lw[fin]))
    println(@sprintf("  %s: %s -> %s, %d particles: mean Δ %.3f, sd Δ %.3f, non-finite %d, ESS/N at φ=1 %.4f, IS log mean exp %.3f; static stages %d (%d resamples); old loglh stored vs recomputed (20): max |Δ| %.3g; %.0f s",
                     tag, lo, hi, NPRE, mean(Δ[fin]), std(Δ[fin]), count(!, fin), ess(e) / NPRE, maximum(lw[fin]) + log(mean(e)), st, nres, maximum(abs.(Lold .- Lold_st[1:20])), tt))
    return st
end
st25 = proj(joinpath(W, "steps", "2025Q3_to_2025Q4", "cloud_health.jld2"), "2025Q4", "2026Q1", Tuple{Symbol, Int}[], "calibration step 25")
st26 = proj(joinpath(W, "steps", "2025Q4_to_2026Q1", "cloud_health.jld2"), "2026Q1", "2026Q2", fx, "FINAL step 26")
ratio = 8 / st25
proj_st = ceil(Int, st26 * max(ratio, 1.0))
proj_h = 2.65 + 0.28 * proj_st      # steps 21-25: health ≈ 2.46-2.61 h (30 passes), bridge ≈ 0.27-0.28 h per stage
println(@sprintf("  calibration: step 25 actual 8 stages vs static %d (ratio %.2f). Final step projection: %d stages -> %.1f h wall, %.0f slot-h on 97 slots (budget 700)",
                 st25, ratio, proj_st, proj_h, proj_h * 97))
chk(proj_h * 97 <= 700, @sprintf("projected production step %.0f slot-h <= 700", proj_h * 97))

println("\n", isempty(fails) ? "ALL GUARDS PASS" : "GUARD FAILURES: " * join(fails, " | "))
exit(isempty(fails) ? 0 : 1)
