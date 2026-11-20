# RTWALK104 FINAL STEP comparison (13 Nov 2026; RTWALK104.md "13 Nov 2026: final step"). 1 slot. Reads only; writes
# out/compare_full*.{md,csv} under work/rtwalk104. Same layout and conventions as work/rtwalk114/scripts/compare_full.jl
# (the ss114 counterpart), so the two reports read side by side.
#   W  = the walk's end point on the full sample: steps/2026Q1_to_2026Q2/cloud_health.jld2 (health-polished), and its
#        raw bridge cloud (Wraw);
#   A  = ss104v3mh5 health cloud (work/ss104v3mh5/health/cloud_health.jld2), and its raw final cloud (Araw). Used ONLY if
#        work/ss104v3mh5/health/health.md says "Verdict (walk thresholds): PASS" (override: CMP_FORCE_A=1, labelled);
#   B  = ss104v3 (2 MH steps, health FAIL) health cloud and its raw final cloud (Braw).
# Over all 101 free parameter-regimes (non-covid 63 = ρ_σ_w + rest, the walk's gated group; covid 38): weighted mean /
#   sd, Δmean / pooled posterior sd (compareAB.jl convention sqrt((sd1² + sd2²)/2)), sd ratio X / W, Δmean / prior sd,
#   z_noise = Δmean / sqrt(se1² + se2²) with se = noise floor × sd (the health checks' MC-noise floor of a mean in sd
#   units, per parameter where the health CSV has it, else the group RMS of the summary row; = the "honest" MC error,
#   it includes the autocorrelation of the MH kernel), and z_ess = Δmean / sqrt(sd1²/ESS1 + sd2²/ESS2) (optimistic:
#   iid particles).
# Thresholds = work/rtwalk114/scripts/compareAB.jl: max |Δmean|/post sd ≤ 0.35 and median ≤ 0.10; non-covid sd ratio in
#   [0.80, 1.25]; covid sd ratios reported, not gated. MDD: compareAB's |Δ| ≤ 0.30 is reported but see the MDD notes.
# Log MDD: the walk's chained estimate = root (est2019blk, direct SMC from the prior on the revised 2019Q4 data)
#   + rtvint104 (revised -> real-time 200410) + the 26 walk increments; compared with A's and B's direct smc estimates.
# Env: CMP_W (default the final step's health cloud), CMP_TAG (default "full"), CMP_TEST=1 (code test before the final
#   step exists: W = step 25's cloud on the 2026Q1 sample; output tagged "test", NOT a result), CMP_FORCE_A=1.
include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk104/scripts/setup.jl")
using CSV, DataFrames, Printf, Statistics, StatsBase
const WK = "/data/dsge_data_dir/estimation-AI/work"
const CMP_TEST = get(ENV, "CMP_TEST", "0") == "1"
const TAG = get(ENV, "CMP_TAG", CMP_TEST ? "test" : "full")
const SD = joinpath(W, "steps", CMP_TEST ? "2025Q4_to_2026Q1" : "2026Q1_to_2026Q2")
const R104 = joinpath(W, "saveroot/output_data/m1002/ss104/estimate/raw")
const RAWA = joinpath(WK, "ss104v3mh5/saveroot/output_data/m1002/ss104/estimate/raw")
const RAWB = joinpath(WK, "ss104v3/saveroot/output_data/m1002/ss104/estimate/raw")
const HEALTH_A = joinpath(WK, "ss104v3mh5/health/health.md")
mkpath(joinpath(W, "out"))
CL = Dict(
  "W"    => get(ENV, "CMP_W", joinpath(SD, "cloud_health.jld2")),
  "Wraw" => joinpath(R104, "smc_cloud_bridge=true_mhsteps=3_nparts=15000_phi=0.95_priorwt=0.0_rtwalk104_sampleend=" *
                           (CMP_TEST ? "2026Q1_vint=260710" : "2026Q2_vint=261013") * ".jld2"),
  "A"    => joinpath(WK, "ss104v3mh5/health/cloud_health.jld2"),
  "Araw" => joinpath(RAWA, "smc_cloud_fcastdate=2026Q3_mhsteps=5_nparts=15000_phi=0.95_ss104v3mh5_vint=261013.jld2"),
  "B"    => joinpath(WK, "ss104v3/health/cloud_health.jld2"),
  "Braw" => joinpath(RAWB, "smc_cloud_fcastdate=2026Q3_mhsteps=2_nparts=15000_phi=0.95_ss104v3_vint=261013.jld2"))
HP = Dict("A" => joinpath(WK, "ss104v3mh5/health/health_params.csv"), "B" => joinpath(WK, "ss104v3/health/health_params.csv"))
# ---- A only after ss104v3mh5 finished AND its health check passed
verdA = isfile(HEALTH_A) ? (mt = match(r"Verdict \(walk thresholds\): (\w+)", read(HEALTH_A, String)); mt === nothing ? "unreadable" : mt[1]) : "missing"
A_OK = verdA == "PASS" || get(ENV, "CMP_FORCE_A", "0") == "1"
A_OK || (delete!(CL, "A"); delete!(CL, "Araw"))
println("ss104v3mh5 health verdict: ", verdA, A_OK ? (verdA == "PASS" ? " -> A used" : " -> A used (CMP_FORCE_A=1: NOT a passed benchmark)") : " -> A NOT used")
isfile(joinpath(dirname(CL["W"]), "HEALTH_PASS")) || println("WARNING: no HEALTH_PASS beside ", CL["W"])

configure_freeset!(m, Set{Tuple{Symbol, Int}}())
ModelConstructors.toggle_regime!(m.parameters, 1)
FREE = ModelConstructors.get_free_para_inds(m.parameters; regime_switching = true); @assert length(FREE) == 101
COVSYM = Set(string.(only([s for (l, s) in walk_block_groups(m) if l == "covid"]))); @assert length(COVSYM) == 38
NAMES = string.(SLOTSYMS[FREE]); ISCOV = [x in COVSYM for x in NAMES]
println("free 101: non-covid ", count(!, ISCOV), ", covid ", count(ISCOV))
# W's two θ̃ (σ_condgdp[2], σ_condcorepce[2]) were drawn from the prior at the final step (pin_full.jl G4: they do not
# enter log p(Y_2026Q2)); in A and B they are free from φ = 0. Both are prior draws; kept in the covid group.
THETA_FINAL = Set(["σ_condgdp_reg2", "σ_condcorepce_reg2"])

function prior_sd(s)
    sym = SLOTSYMS[s]; mt = match(r"^(.*)_reg(\d+)$", string(sym))
    key, r = mt === nothing ? (sym, 1) : (Symbol(mt[1]), parse(Int, mt[2]))
    p = m[key]; pr = (haskey(p.regimes, :prior) && haskey(p.regimes[:prior], r)) ? p.regimes[:prior][r] : p.prior
    d = get(pr); sd = try std(d) catch; NaN end
    isfinite(sd) && return sd
    x = rand(MersenneTwister(s), d, 200_000); return (quantile(x, 0.75) - quantile(x, 0.25)) / 1.349
end
PSD = [prior_sd(s) for s in FREE]

# ---- MC-noise floors: W from its summary row (group RMS); A/B per parameter from their health CSVs
S104 = CSV.read(joinpath(W, "rtwalk104_summary.csv"), DataFrame; stringtype = String)
sumF = CMP_TEST ? S104[1:0, :] : CSV.read(joinpath(W, "rtwalk104_full_summary.csv"), DataFrame; stringtype = String)
wrow = CMP_TEST ? S104[findfirst(==(25), S104.step), :] : sumF[findfirst(==(26), sumF.step), :]
NOISE = Dict{String, Vector{Float64}}()
NOISE["W"] = [c ? Float64(wrow.covid_noise) : Float64(wrow.gap_noise) for c in ISCOV]
for k in ("A", "B")
    haskey(CL, k) && isfile(HP[k]) || continue
    h = CSV.read(HP[k], DataFrame); d = Dict(String(r.para) => Float64(r.noise) for r in eachrow(h))
    nc = sqrt(mean(abs2, [d[x] for x in NAMES[.!ISCOV] if haskey(d, x)])); cv = sqrt(mean(abs2, [d[x] for x in NAMES[ISCOV] if haskey(d, x)]))
    println("$k: per-parameter noise for $(count(x -> haskey(d, x), NAMES)) of 101; group RMS non-covid $(round(nc, digits = 4)), covid $(round(cv, digits = 4))")
    NOISE[k] = [get(d, x, ISCOV[j] ? cv : nc) for (j, x) in enumerate(NAMES)]
end
for k in ("W", "A", "B"); haskey(NOISE, k) && (NOISE[k * "raw"] = NOISE[k]); end   # raw clouds: the same floors (approximation)

MOM = Dict{String, Any}()
for (k, f) in CL
    isfile(f) || (println("missing $k: $f"); continue)
    c = load(f, "cloud"); P = c.particles; n = size(P, 2)
    @assert n - 5 == length(VALS0)
    w = P[:, n] ./ sum(P[:, n]); wt = Weights(w); ess = 1 / sum(abs2, w)
    @assert all(s -> all(==(VALS0[s]), P[:, s]), setdiff(eachindex(VALS0), FREE)) "$k: a fixed slot is off its value"
    MOM[k] = (mu = [mean(P[:, s], wt) for s in FREE], sd = [std(P[:, s], wt; corrected = false) for s in FREE], ess = ess,
              loglh = mean(P[:, n-4], wt), logpost = mean(P[:, n-4] .+ P[:, n-3], wt), phi = c.tempering_schedule[end], file = f)
    println(@sprintf("%-5s ESS %.0f  φ %.3f  mean loglh %.3f  mean logpost %.3f  %s", k, ess, MOM[k].phi, MOM[k].loglh, MOM[k].logpost, f))
end
for k in keys(NOISE); haskey(MOM, k) || delete!(NOISE, k); end

function compare(a, b)    # b relative to a
    A = MOM[a]; B = MOM[b]
    pool = sqrt.((A.sd .^ 2 .+ B.sd .^ 2) ./ 2); d = B.mu .- A.mu
    df = DataFrame(slot = NAMES, group = [c ? "covid" : "noncovid" for c in ISCOV],
                   mean_1 = A.mu, mean_2 = B.mu, sd_1 = A.sd, sd_2 = B.sd, prior_sd = PSD,
                   dmean_post_sd = d ./ pool, sd_ratio = B.sd ./ A.sd, dmean_prior_sd = d ./ PSD,
                   z_noise = d ./ sqrt.((NOISE[a] .* A.sd) .^ 2 .+ (NOISE[b] .* B.sd) .^ 2),
                   z_ess = d ./ sqrt.(A.sd .^ 2 ./ A.ess .+ B.sd .^ 2 ./ B.ess))
    sort!(df, :dmean_post_sd, by = abs, rev = true)
    CSV.write(joinpath(W, "out", "compare_$(TAG)_$(a)_vs_$(b)_params.csv"), df)
    return df
end
io = open(joinpath(W, "out", "compare_$(TAG).md"), "w")
pr(x...) = (println(x...); println(io, x...))
pr("# RTWALK104 final-step comparison ($TAG), ", now(), "\n")
CMP_TEST && pr("**CODE TEST ONLY: W is the 2026Q1 walk cloud, not the full-sample end point. Not a result.**\n")
pr("ss104v3mh5 health verdict: **", verdA, "**", A_OK ? (verdA == "PASS" ? " (A included)" : " (A included by CMP_FORCE_A=1: NOT a passed benchmark)") : " (A excluded)", "\n")
for (k, v) in sort(collect(MOM), by = first)
    pr(@sprintf("- %s: ESS %.0f, φ %.3f, mean loglh %.3f, mean logpost %.3f, `%s`", k, v.ess, v.phi, v.loglh, v.logpost, v.file))
end
pr("\nThresholds (compareAB.jl): max |Δmean|/post sd ≤ 0.35, median ≤ 0.10; non-covid sd ratio in [0.80, 1.25] (covid not gated). ",
   "z_noise uses the health checks' MC-noise floors (honest, autocorrelation included); z_ess assumes iid particles (optimistic).\n")
pr("| pair (2 vs 1) | group | n | max abs Δ/post sd (slot) | median | #>0.10 | #>0.20 | #>0.35 | sd ratio min..max | #outside [0.80,1.25] | #abs z_noise>3 | #abs z_ess>3 | verdict |")
pr("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
PAIRS = [("W", "A"), ("W", "B"), ("W", "Braw"), ("Wraw", "Araw"), ("Wraw", "Braw"), ("A", "B"), ("A", "Braw")]
for (a, b) in PAIRS
    (haskey(MOM, a) && haskey(MOM, b)) || continue
    df = compare(a, b)
    for (g, sel) in (("all", trues(nrow(df))), ("noncovid", df.group .== "noncovid"), ("covid", df.group .== "covid"))
        x = df[sel, :]; ad = abs.(x.dmean_post_sd); i = argmax(ad)
        mpass = maximum(ad) <= 0.35 && median(ad) <= 0.10
        ok = g == "covid" ? "(sd not gated)" : (g == "all" ? (mpass ? "means PASS" : "means FAIL") :
             (minimum(x.sd_ratio) >= 0.80 && maximum(x.sd_ratio) <= 1.25 ? "sd PASS" : "sd FAIL") * (mpass ? ", means PASS" : ", means FAIL"))
        pr(@sprintf("| %s vs %s | %s | %d | %.3f (%s) | %.3f | %d | %d | %d | %.3f..%.3f | %d | %d | %d | %s |", b, a, g, nrow(x), ad[i], x.slot[i], median(ad),
                    count(>(0.10), ad), count(>(0.20), ad), count(>(0.35), ad), minimum(x.sd_ratio), maximum(x.sd_ratio),
                    count(r -> !(0.80 <= r <= 1.25), x.sd_ratio), count(>(3), abs.(x.z_noise)), count(>(3), abs.(x.z_ess)), ok))
    end
end
for (a, b) in [("W", "A"), ("W", "B"), ("A", "B")]
    (haskey(MOM, a) && haskey(MOM, b)) || continue
    df = CSV.read(joinpath(W, "out", "compare_$(TAG)_$(a)_vs_$(b)_params.csv"), DataFrame)
    pr("\n## $b vs $a: the 15 largest |Δmean|/post sd\n\n| slot | group | mean $a | mean $b | sd $a | sd $b | Δ/post sd | sd ratio | Δ/prior sd | z_noise | z_ess |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in eachrow(df[1:15, :])
        pr(@sprintf("| %s | %s | %.4f | %.4f | %.4f | %.4f | %+.3f | %.3f | %+.3f | %+.1f | %+.1f |", r.slot, r.group, r.mean_1, r.mean_2, r.sd_1, r.sd_2, r.dmean_post_sd, r.sd_ratio, r.dmean_prior_sd, r.z_noise, r.z_ess))
    end
    x = df[df.group .== "noncovid", :]; o = sortperm(abs.(log.(x.sd_ratio)), rev = true)
    pr("\nnon-covid sd ratios furthest from 1: ", join([@sprintf("%s %.3f", x.slot[i], x.sd_ratio[i]) for i in o[1:6]], ", "))
    x = df[df.group .== "covid", :]; o = sortperm(abs.(log.(x.sd_ratio)), rev = true)
    pr("covid sd ratios furthest from 1: ", join([@sprintf("%s %.3f", x.slot[i], x.sd_ratio[i]) for i in o[1:6]], ", "))
    y = df[in.(df.slot, Ref(THETA_FINAL)), :]
    pr("final-step θ̃ (prior draws in W): ", join([@sprintf("%s Δ/post sd %+.3f, sd ratio %.3f, Δ/prior sd %+.3f", r.slot, r.dmean_post_sd, r.sd_ratio, r.dmean_prior_sd) for r in eachrow(y)], "; "))
end

# ---------------- log MDD
pr("\n## Log MDD\n")
pr("Is the walk's chained MDD comparable with a direct φ = 0 estimate? The chain telescopes: root log p(Y_2019Q4^rev) + ",
   "[log p(Y_2019Q4^200410) − log p(Y_2019Q4^rev)] + Σ_t [log p(Y_t+1^v(t+1)) − log p(Y_t^v(t))] = log p(Y_2026Q2^261013). ",
   "Each link is SMC's incremental-weight estimate with μ = 0 and no log_prob_old_data (smc_level NaN), so no link double-counts. ",
   "The root (est2019blk) is a direct SMC run from the prior with 63 free slots; the 38 others enter as θ̃ drawn from their proper priors at the ",
   "step where they first enter, and they do not enter the earlier likelihoods (pinned perturbation test; ≤ 3.5e-10), so ",
   "∫ p(Y_t|θ1) p(θ1) p(θ̃) dθ = p(Y_t) and the prior is that of the direct runs (pin_full.jl G3: logprior identical). ",
   "So the level is comparable in definition. Its error is NOT the increments' lineage NSE alone: the root has only 2 eves (lineage NSE ",
   "unusable) and two 2019Q4 runs differed by 3 log points (EST-2019Q4-BLK: −984.0 vs −987.0); the direct full-sample smc ",
   "estimates have their own between-run noise (ss104v2: smc −1326.8 vs mhm −1271.4). Read walk − direct against a few log points, not 0.30.\n")
mddof(f) = jldopen(ff -> haskey(ff, "log_mdd_estimates") ? ff["log_mdd_estimates"] : Dict(), f, "r")
g(d, k) = d isa AbstractDict && haskey(d, k) ? Float64(d[k]) : NaN
rows = []
root = joinpath(WK, "est2019blk/saveroot/output_data/m1002/ss104/estimate/raw/smc_cloud_mhsteps=3_nparts=15000_phi=0.95_sampleend=2019Q4_ss104_s2019q4blk_vint=261013.jld2")
d = mddof(root); push!(rows, (link = "root: ss104 2019Q4 revised-261013, direct SMC from the prior (est2019blk)", smc = g(d, "smc"), nse = g(d, "smc_lineage_nse"), mhm = g(d, "mhm"), n_eves = g(d, "n_eves")))
d = mddof(joinpath(WK, "rtvint104/saveroot/output_data/m1002/ss104/estimate/raw/smc_cloud_bridge=true_mhsteps=3_nparts=15000_phi=0.95_priorwt=0.0_rtvint104revois_sampleend=2019Q4_vint=200410.jld2"))
push!(rows, (link = "2019Q4: revised -> real-time 200410 (rtvint104)", smc = g(d, "smc"), nse = g(d, "smc_lineage_nse"), mhm = NaN, n_eves = g(d, "n_eves")))
SW = sort(vcat(S104, sumF; cols = :union), :step)
for r in eachrow(SW)
    f = only(Base.filter(x -> occursin("rtwalk104_sampleend=$(r.new)_", x) && endswith(x, ".jld2") && !occursin("stage=", x), readdir(R104)))
    d = mddof(joinpath(R104, f)); @assert isapprox(g(d, "smc"), Float64(r.log_mdd_incr); atol = 1e-9) "MDD of $f != summary row $(r.step)"
    push!(rows, (link = "walk step $(r.step): $(r.old) -> $(r.new) (vint $(r.vint_new))", smc = g(d, "smc"), nse = g(d, "smc_lineage_nse"), mhm = NaN, n_eves = g(d, "n_eves")))
end
dm = DataFrame(rows); CSV.write(joinpath(W, "out", "compare_$(TAG)_mdd.csv"), dm)
pr("| link | smc | lineage NSE | n_eves |\n|---|---|---|---|")
for r in eachrow(dm); pr(@sprintf("| %s | %.4f | %.4f | %.0f |", r.link, r.smc, r.nse, r.n_eves)); end
tot = sum(dm.smc); nse_inc = sqrt(sum(abs2, dm.nse[2:end][isfinite.(dm.nse[2:end])]))
pr(@sprintf("\n**walk chained log p_104(Y_%s) = %.3f**; NSE of the increments (√Σ lineage NSE²) %.3f; root lineage NSE %.3f (n_eves %.0f); root mhm %.3f",
            CMP_TEST ? "2026Q1, 260710" : "2026Q2, 261013", tot, nse_inc, dm.nse[1], dm.n_eves[1], dm.mhm[1]))
for (k, f) in (("A (ss104v3mh5)", get(CL, "Araw", "")), ("B (ss104v3)", CL["Braw"]))
    isfile(f) || (pr("- $k: not available"); continue)
    d = mddof(f); pr(@sprintf("- %s direct: smc %.3f, lineage NSE %.3f, n_eves %.0f, mhm %.3f;  walk − direct = %+.3f%s", k, g(d, "smc"), g(d, "smc_lineage_nse"), g(d, "n_eves"), g(d, "mhm"), tot - g(d, "smc"),
                      CMP_TEST ? " (TEST: different samples, meaningless)" : ""))
end
close(io)
println("\nwritten ", joinpath(W, "out", "compare_$(TAG).md"))
