# RTWALK114 FINAL STEP (13 Nov 2026): build the full-sample pin and run the guards (1 slot, SGE). See full_data.jl.
#   G1 pin === ss114 frozen 26 x 268 (sha256-checked) === ss104v3 frozen 28 x 268 minus obs_pgap / obs_ygap (NaN-aware)
#   G2 walk LIK at the defaults (all 93 free), trunc on and off, === -6192.287399926011 (ss114 build check, 344099)
#   G3 stored loglh of 8 particles each of the ss114 (344099) final cloud, the ss114mh5 final cloud and its
#      health cloud recompute on the pin (rtol 1e-12; reported max |Δ|)
#   G4 the 2 slots first free at 2026Q2 (fixed_at("2026Q1")) = σ_condgdp[2], σ_condcorepce[2]; perturbed to prior
#      draws at 3 bases, ΔL on the full data (expected 0: they enter only the 2026Q3 conditional regime, SS114 l.124)
#   G5 truncated measurement solve === full solve on the full data, 8 θ
#   I  info: revisions 260710 pin -> full data (first 267 columns), the new column, and the rev-convention rebuild
#      from the revised 261013 frame (rt_data.jl construction) vs the frozen matrix
include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk114/scripts/setup.jl")
include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk114/scripts/rt_data.jl")
include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk114/scripts/full_data.jl")
using Printf
const LDEF = -6192.287399926011
fails = String[]
chk(c, msg) = (println(c ? "  PASS  " : "  FAIL  ", msg); c || push!(fails, msg))

println("== G1 data")
d114, sh114, o114 = full_frozen_checked(m)
if !isfile(FULL_PIN)
    tmp = FULL_PIN * ".tmp$(getpid())"
    jldsave(tmp; data = d114, vint = FULL_V, sample_end = "2026Q2", source = S114_FROZEN, sha_ss114_frozen = sh114,
            ss104v3_frozen = V3_FROZEN, sha_ss104v3_frozen = bytes2hex(open(sha256, V3_FROZEN)),
            observables = string.(o114), built_at = string(now()))
    mv(tmp, FULL_PIN)
    open(FULL_PIN_SHA, "w") do io; println(io, bytes2hex(open(sha256, FULL_PIN)), "  ", basename(FULL_PIN)); end
    println("  pinned ", FULL_PIN)
end
dF, _ = full_data_pinned(m)
chk(isequal(dF, d114) && size(dF) == (26, 268), "pin === ss114 frozen 26 x 268 (and === ss104v3 frozen minus gap rows; sha256 $sh114)")
println("  non-NaN cells: ", count(!isnan, dF), ";  last column (2026Q2) non-NaN rows: ", [string(o114[i]) for i in 1:26 if !isnan(dF[i, end])])

println("== G2 defaults")
configure_freeset!(m, Set{Tuple{Symbol, Int}}())
@assert nfree() == 93
l_on = LIK(m, VALS0, dF)
m <= Setting(:likelihood_truncated_regimes, false); l_off = LIK(m, VALS0, dF); m <= Setting(:likelihood_truncated_regimes, true)
chk(l_on === LDEF && l_off === LDEF, "loglh at defaults: trunc on $(repr(l_on)), off $(repr(l_off)), ss114 $(repr(LDEF))")

println("== G3 stored loglh")
const C114 = "/data/dsge_data_dir/estimation-AI/work/ss114/saveroot/output_data/m1002/ss114/estimate/raw/smc_cloud_fcastdate=2026Q3_mhsteps=2_nparts=15000_phi=0.95_ss114_vint=261013.jld2"
const C114H = "/data/dsge_data_dir/estimation-AI/work/ss114/health/cloud_health.jld2"
const CMH5 = "/data/dsge_data_dir/estimation-AI/work/ss114mh5/saveroot/output_data/m1002/ss114/estimate/raw/smc_cloud_fcastdate=2026Q3_mhsteps=5_nparts=15000_phi=0.95_ss114mh5_vint=261013.jld2"
const CMH5H = "/data/dsge_data_dir/estimation-AI/work/ss114mh5/health/cloud_health.jld2"
for (lab, f) in (("ss114 344099 final", C114), ("ss114 health cloud", C114H), ("ss114mh5 final", CMH5), ("ss114mh5 health cloud", CMH5H))
    P = load(f, "cloud").particles; n = size(P, 2)
    idx = unique(vcat(argmax(P[:, n-4]), 1, rand(MersenneTwister(13), 1:size(P, 1), 6)))
    d = [LIK(m, P[i, 1:n-5], dF) - P[i, n-4] for i in idx]
    chk(all(isfinite, d) && maximum(abs.(d)) <= 1e-12 * 7000, @sprintf("%s: stored loglh recomputed at %d particles, max |Δ| %.3g", lab, length(idx), maximum(abs.(d))))
end

println("== G4 slots first free at 2026Q2")
fx = fixed_at("2026Q1")
println("  fixed_at(2026Q1) = ", fx, ";  freeset keys present: ", sort(collect(keys(load(FREESET_PIN, "fixed_at")))))
chk(Set(fx) == Set([(:σ_condgdp, 2), (:σ_condcorepce, 2)]), "fixed at 2026Q1 = σ_condgdp[2], σ_condcorepce[2]")
Pm = load(C114, "cloud").particles; nm_ = size(Pm, 2)
bases = [VALS0, Pm[argmax(Pm[:, nm_-4]), 1:nm_-5], Pm[7, 1:nm_-5]]
sl = [slot_index(k, r) for (k, r) in fx]
dmax = 0.0
for (b, θ0) in enumerate(bases), rep in 1:3
    global dmax
    θ = copy(θ0); Random.seed!(1000b + rep); dr = ModelConstructors.rand_regime_switching(m.parameters); ModelConstructors.toggle_regime!(m.parameters, 1)
    θ[sl] = dr[sl]
    l0 = LIK(m, θ0, dF); l1 = LIK(m, θ, dF)
    println(@sprintf("  base %d rep %d: %s -> %s: ΔL %.3g", b, rep, string(θ0[sl]), string(θ[sl]), l1 - l0))
    dmax = max(dmax, abs(l1 - l0))
end
chk(dmax <= ENTER_TOL, @sprintf("σ_condgdp[2], σ_condcorepce[2] do not enter log p(Y_2026Q2): max |ΔL| %.3g (tol %g)", dmax, ENTER_TOL))

println("== G5 trunc vs full solve")
dd = Float64[]
for i in vcat(argmax(Pm[:, nm_-4]), rand(MersenneTwister(15), 1:size(Pm, 1), 7))
    a = LIK(m, Pm[i, 1:nm_-5], dF)
    m <= Setting(:likelihood_truncated_regimes, false); b = LIK(m, Pm[i, 1:nm_-5], dF); m <= Setting(:likelihood_truncated_regimes, true)
    push!(dd, a === b ? 0.0 : abs(a - b))
end
chk(all(==(0.0), dd), "truncated === full solve at 8 θ on the full data")

println("== I revisions and construction")
d1, v1 = rt_data_pinned(m, Date(2026, 3, 31))
@assert v1 == "260710" && size(d1, 2) == 267
for i in 1:26
    a = d1[i, :]; b = dF[i, 1:267]
    pat = count(isnan.(a) .!= isnan.(b)); both = .!isnan.(a) .& .!isnan.(b)
    nd = count(a[both] .!= b[both]); mx = nd == 0 ? 0.0 : maximum(abs.(a[both] .- b[both]))
    lastrev = nd == 0 ? "-" : string(DSGE.quarter_range(date_presample_start(m), Date(2026, 3, 31))[findlast(j -> both[j] && a[j] != b[j], 1:267)])
    println(@sprintf("  %-26s cells revised %3d (max |Δ| %.4f, latest %s), NaN-pattern changes %d, 2026Q2 value %s", string(o114[i]), nd, mx, lastrev, pat, isnan(dF[i, end]) ? "NaN" : @sprintf("%.4f", dF[i, end])))
end
rv = data_matrix(m, REV_DF, FULL_T)
r = m.observables[:obs_nominalrate]
for (j, dte) in enumerate(DSGE.quarter_range(date_presample_start(m), FULL_T)); ZLB_NAN[1] <= dte <= ZLB_NAN[2] && (rv[r, j] = NaN); end
println("  rev-convention rebuild from the revised 261013 frame (", basename(FROZEN_DF), ") === frozen matrix: ", isequal(rv, dF),
        isequal(rv, dF) ? "" : string("; rows differing: ", [string(o114[i]) for i in 1:26 if !isequal(rv[i, :], dF[i, :])]))

println("\n", isempty(fails) ? "ALL GUARDS PASS" : "GUARD FAILURES: " * join(fails, " | "))
exit(isempty(fails) ? 0 : 1)
