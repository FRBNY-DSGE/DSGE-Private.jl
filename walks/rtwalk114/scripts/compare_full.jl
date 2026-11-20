# RTWALK114 FINAL STEP comparison (13 Nov 2026; RTWALK114-FULLCOMPARE.md). 1 slot. Reads only.
#   W  = the walk's end point on the full sample: steps/2026Q1_to_2026Q2/cloud_health.jld2 (health-polished), and its raw
#        bridge cloud;
#   A  = ss114mh5 health-polished cloud (health PASS), and its raw final cloud;
#   B  = ss114 (344099) health-polished cloud (health FAIL) and its raw final cloud.
# Over all 93 free parameters (non-covid 63 = ρ_σ_w + rest, the walk's gated group; covid 30): weighted mean / sd,
#   Δmean / pooled posterior sd (compareAB.jl convention sqrt((sd1² + sd2²)/2)), sd ratio X / W, Δmean / prior sd,
#   z_noise = Δmean / sqrt(se1² + se2²) with se = noise floor × sd (health checks: per-parameter where the health CSV
#   has it, else the group RMS from the summary), and z_ess = Δmean / sqrt(sd1²/ESS1 + sd2²/ESS2) (optimistic).
# Thresholds = compareAB.jl: max |Δmean|/post sd ≤ 0.35, median ≤ 0.10; non-covid sd ratio in [0.80, 1.25].
# Also the log MDD chain of the walk and the direct estimations' log MDD (out/compare_full_mdd.csv).
# Env: CMP_W (default the health cloud of the final step), CMP_TAG (default "full").
include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk114/scripts/setup.jl")
using CSV, DataFrames, Printf, Statistics, StatsBase
const WK = "/data/dsge_data_dir/estimation-AI/work"
const TAG = get(ENV, "CMP_TAG", "full")
const SD = joinpath(W, "steps", "2026Q1_to_2026Q2")
const RAW114 = joinpath(W, "saveroot/output_data/m1002/ss114/estimate/raw")
CL = Dict(
  "W"    => get(ENV, "CMP_W", joinpath(SD, "cloud_health.jld2")),
  "Wraw" => joinpath(RAW114, "smc_cloud_bridge=true_mhsteps=3_nparts=15000_phi=0.95_priorwt=0.0_rtwalk114_sampleend=2026Q2_vint=261013.jld2"),
  "A"    => joinpath(WK, "ss114mh5/health/cloud_health.jld2"),
  "Araw" => joinpath(WK, "ss114mh5/saveroot/output_data/m1002/ss114/estimate/raw/smc_cloud_fcastdate=2026Q3_mhsteps=5_nparts=15000_phi=0.95_ss114mh5_vint=261013.jld2"),
  "B"    => joinpath(WK, "ss114/health/cloud_health.jld2"),
  "Braw" => joinpath(WK, "ss114/saveroot/output_data/m1002/ss114/estimate/raw/smc_cloud_fcastdate=2026Q3_mhsteps=2_nparts=15000_phi=0.95_ss114_vint=261013.jld2"))
HP = Dict("A" => joinpath(WK, "ss114mh5/health/health_params.csv"), "B" => joinpath(WK, "ss114/health/health_params.csv"))
configure_freeset!(m, Set{Tuple{Symbol, Int}}())
ModelConstructors.toggle_regime!(m.parameters, 1)
FREE = ModelConstructors.get_free_para_inds(m.parameters; regime_switching = true); @assert length(FREE) == 93
COVSYM = Set(string.(only([s for (l, s) in walk_block_groups(m) if l == "covid"]))); @assert length(COVSYM) == 30
NAMES = string.(SLOTSYMS[FREE]); ISCOV = [x in COVSYM for x in NAMES]
println("free 93: non-covid ", count(!, ISCOV), ", covid ", count(ISCOV))

function prior_sd(s)
    sym = SLOTSYMS[s]; mt = match(r"^(.*)_reg(\d+)$", string(sym))
    key, r = mt === nothing ? (sym, 1) : (Symbol(mt[1]), parse(Int, mt[2]))
    p = m[key]; pr = (haskey(p.regimes, :prior) && haskey(p.regimes[:prior], r)) ? p.regimes[:prior][r] : p.prior
    d = get(pr); sd = try std(d) catch; NaN end
    isfinite(sd) && return sd
    x = rand(MersenneTwister(s), d, 200_000); return (quantile(x, 0.75) - quantile(x, 0.25)) / 1.349
end
PSD = [prior_sd(s) for s in FREE]

# W's noise floors (group RMS, from its health summary row); A/B per parameter from their health CSVs
const CMP_TEST = get(ENV, "CMP_TEST", "0") == "1"    # code test only (before the final step exists): step 25's summary row; output tagged, not a result
sumF = CMP_TEST ? CSV.read(joinpath(W, "rtwalk114_summary.csv"), DataFrame; stringtype = String)[1:0, :] : CSV.read(joinpath(W, "rtwalk114_full_summary.csv"), DataFrame; stringtype = String)
wrow = CMP_TEST ? (s = CSV.read(joinpath(W, "rtwalk114_summary.csv"), DataFrame); s[findfirst(==(25), s.step), :]) : sumF[findfirst(==(26), sumF.step), :]
NOISE = Dict{String, Vector{Float64}}()
NOISE["W"] = [c ? Float64(wrow.covid_noise) : Float64(wrow.gap_noise) for c in ISCOV]
for k in ("A", "B")
    h = CSV.read(HP[k], DataFrame); d = Dict(String(r.para) => Float64(r.noise) for r in eachrow(h))
    nc = sqrt(mean(abs2, [d[x] for x in NAMES[.!ISCOV] if haskey(d, x)])); cv = sqrt(mean(abs2, [d[x] for x in NAMES[ISCOV] if haskey(d, x)]))
    println("$k: per-parameter noise for $(count(x -> haskey(d, x), NAMES)) of 93; group RMS non-covid $(round(nc, digits = 4)), covid $(round(cv, digits = 4))")
    NOISE[k] = [get(d, x, ISCOV[j] ? cv : nc) for (j, x) in enumerate(NAMES)]
end
NOISE["Wraw"] = NOISE["W"]; NOISE["Araw"] = NOISE["A"]; NOISE["Braw"] = NOISE["B"]   # raw clouds: the same floors (approximation)

MOM = Dict{String, Any}()
for (k, f) in CL
    isfile(f) || (println("missing $k: $f"); continue)
    c = load(f, "cloud"); P = c.particles; n = size(P, 2)
    w = P[:, n] ./ sum(P[:, n]); wt = Weights(w); ess = 1 / sum(abs2, w)
    @assert all(s -> all(==(VALS0[s]), P[:, s]), setdiff(eachindex(VALS0), FREE)) "$k: a fixed slot is off its value"
    MOM[k] = (mu = [mean(P[:, s], wt) for s in FREE], sd = [std(P[:, s], wt; corrected = false) for s in FREE], ess = ess,
              loglh = mean(P[:, n-4], wt), logpost = mean(P[:, n-4] .+ P[:, n-3], wt), phi = c.tempering_schedule[end], file = f)
    println(@sprintf("%-5s ESS %.0f  φ %.3f  mean loglh %.3f  mean logpost %.3f  %s", k, ess, MOM[k].phi, MOM[k].loglh, MOM[k].logpost, f))
end

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
pr("# Final-step comparison ($TAG), ", now(), "\n")
for (k, v) in sort(collect(MOM), by = first)
    pr(@sprintf("- %s: ESS %.0f, mean loglh %.3f, mean logpost %.3f, `%s`", k, v.ess, v.loglh, v.logpost, v.file))
end
pr("\nThresholds (compareAB.jl): max |Δmean|/post sd ≤ 0.35, median ≤ 0.10; non-covid sd ratio in [0.80, 1.25].\n")
pr("| pair (2 vs 1) | group | n | max abs Δ/post sd (slot) | median | #>0.10 | #>0.20 | #>0.35 | sd ratio min..max | #outside [0.80,1.25] | #abs z_noise>3 | #abs z_ess>3 | verdict |")
pr("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
PAIRS = [("W", "A"), ("W", "B"), ("W", "Braw"), ("Wraw", "Araw"), ("A", "B"), ("A", "Braw")]
for (a, b) in PAIRS
    (haskey(MOM, a) && haskey(MOM, b)) || continue
    df = compare(a, b)
    for (g, sel) in (("all", trues(nrow(df))), ("noncovid", df.group .== "noncovid"), ("covid", df.group .== "covid"))
        x = df[sel, :]; ad = abs.(x.dmean_post_sd); i = argmax(ad)
        ok = g == "covid" ? "(sd not gated)" : (g == "all" ? (maximum(ad) <= 0.35 && median(ad) <= 0.10 ? "means PASS" : "means FAIL") :
             (minimum(x.sd_ratio) >= 0.80 && maximum(x.sd_ratio) <= 1.25 ? "sd PASS" : "sd FAIL") * (maximum(ad) <= 0.35 && median(ad) <= 0.10 ? ", means PASS" : ", means FAIL"))
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
end

# ---------------- log MDD
pr("\n## Log MDD\n")
mddof(f) = jldopen(ff -> haskey(ff, "log_mdd_estimates") ? ff["log_mdd_estimates"] : Dict(), f, "r")
g(d, k) = d isa AbstractDict && haskey(d, k) ? Float64(d[k]) : NaN
rows = []
root = joinpath(WK, "est2019blk/saveroot/output_data/m1002/ss104/estimate/raw/smc_cloud_mhsteps=3_nparts=15000_phi=0.95_sampleend=2019Q4_ss104_s2019q4blk_vint=261013.jld2")
d = mddof(root); push!(rows, (link = "root: ss104 2019Q4 revised-261013, direct SMC from the prior (342899)", smc = g(d, "smc"), nse = g(d, "smc_lineage_nse"), mhm = g(d, "mhm"), n_eves = g(d, "n_eves")))
d = mddof(joinpath(WK, "rtvint104/saveroot/output_data/m1002/ss104/estimate/raw/smc_cloud_bridge=true_mhsteps=3_nparts=15000_phi=0.95_priorwt=0.0_rtvint104revois_sampleend=2019Q4_vint=200410.jld2"))
push!(rows, (link = "ss104 2019Q4: revised -> real-time 200410 (rtvint104)", smc = g(d, "smc"), nse = g(d, "smc_lineage_nse"), mhm = NaN, n_eves = g(d, "n_eves")))
R104 = joinpath(WK, "rtwalk104/saveroot/output_data/m1002/ss104/estimate/raw")
S104 = CSV.read(joinpath(WK, "rtwalk104/rtwalk104_summary.csv"), DataFrame)
for r in eachrow(S104[S104.step .<= 10, :])
    f = only(Base.filter(x -> occursin("rtwalk104_sampleend=$(r.new)_", x) && endswith(x, ".jld2") && !occursin("stage=", x), readdir(R104)))
    d = mddof(joinpath(R104, f)); @assert isapprox(g(d, "smc"), Float64(r.log_mdd_incr); atol = 1e-9)
    push!(rows, (link = "ss104 walk step $(r.step): $(r.old) -> $(r.new)", smc = g(d, "smc"), nse = g(d, "smc_lineage_nse"), mhm = NaN, n_eves = g(d, "n_eves")))
end
d = mddof(joinpath(RAW114, "smc_cloud_bridge=true_mhsteps=3_nparts=15000_phi=0.95_priorwt=0.0_rtwalk114B_sampleend=2022Q2_vint=221010.jld2"))
push!(rows, (link = "model bridge ss104 -> ss114 at 2022Q2 (B 2022Q2, seeds the ss114 chain)", smc = g(d, "smc"), nse = g(d, "smc_lineage_nse"), mhm = NaN, n_eves = g(d, "n_eves")))
S114 = vcat(CSV.read(joinpath(W, "rtwalk114_summary.csv"), DataFrame; stringtype = String), sumF; cols = :union)
for r in eachrow(sort(S114[S114.step .>= 11, :], :step))
    f = only(Base.filter(x -> occursin("rtwalk114_sampleend=$(r.new)_", x) && endswith(x, ".jld2") && !occursin("stage=", x), readdir(RAW114)))
    d = mddof(joinpath(RAW114, f)); @assert isapprox(g(d, "smc"), Float64(r.log_mdd_incr); atol = 1e-9)
    push!(rows, (link = "ss114 walk step $(r.step): $(r.old) -> $(r.new)", smc = g(d, "smc"), nse = g(d, "smc_lineage_nse"), mhm = NaN, n_eves = g(d, "n_eves")))
end
dm = DataFrame(rows); CSV.write(joinpath(W, "out", "compare_$(TAG)_mdd.csv"), dm)
pr("| link | smc | lineage NSE | n_eves |\n|---|---|---|---|")
for r in eachrow(dm); pr(@sprintf("| %s | %.4f | %.4f | %.0f |", r.link, r.smc, r.nse, r.n_eves)); end
tot = sum(dm.smc); nse_inc = sqrt(sum(abs2, dm.nse[2:end][isfinite.(dm.nse[2:end])]))
pr(@sprintf("\n**walk chained log p_114(Y_2026Q2, 261013) = %.3f**; NSE of the increments (√Σ lineage NSE²) %.3f; root's lineage NSE %.3f (n_eves %.0f); root mhm %.3f", tot, nse_inc, dm.nse[1], dm.n_eves[1], dm.mhm[1]))
for (k, f) in (("A (ss114mh5)", CL["Araw"]), ("B (ss114 344099)", CL["Braw"]))
    d = mddof(f); pr(@sprintf("- %s direct: smc %.3f, lineage NSE %.3f, n_eves %.0f, mhm %.3f;  walk − direct = %+.3f", k, g(d, "smc"), g(d, "smc_lineage_nse"), g(d, "n_eves"), g(d, "mhm"), tot - g(d, "smc")))
end
close(io)
