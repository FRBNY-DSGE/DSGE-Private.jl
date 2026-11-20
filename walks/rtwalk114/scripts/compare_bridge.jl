# RTWALK114: compare the two candidate ss114 2020Q3 seeds (Marco, 8 Nov).
#   A = seed/ss114_2020Q3_seed.jld2                          (ss104 2020Q3 cloud, mapped + importance-reweighted)
#   B = steps_bridge/2020Q2_to_2020Q3/cloud_health.jld2      (ss114 bridged 2020Q1 -> Q2 -> Q3 from the mapped ss104 2020Q1 cloud)
#   also ss104's own 2020Q3 cloud (rtwalk104 step 3) for reference.
# Outputs out/compare_bridge.md and out/compare_bridge_params.csv:
#   - per free parameter at 2020Q3: weighted mean and sd of A and B, (mean_B − mean_A) / prior sd (IQR/1.349 where the
#     prior has no finite sd), sd_B / sd_A,
#     and (mean_B − mean_A) / sqrt(sd_A² + sd_B²) · sqrt(ESS) style z (MC noise scale; informative only);
#   - the 2020Q3 loglh distribution (stored loglh = ss114 on the 2020Q3 vintage in both; recomputed at 20 particles);
#   - log MDD increments of the two ss114 bridge steps vs ss104's (rtwalk104_summary.csv steps 2, 3): expected
#     ≈ +7.84 at 2020Q2 (ss104's two gap pseudo-observations) and ≈ 0 at 2020Q3.
# usage: run.sh compare_bridge.jl (1 slot); runs automatically after rtw114_2020Q3 (held job, RTWALK114.md).
include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk114/scripts/setup.jl")
include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk114/scripts/rt_data.jl")
using CSV, DataFrames, Printf, Statistics, StatsBase
const FA = joinpath(W, "seed", "ss114_2020Q3_seed.jld2")
const FB = get(ENV, "CMP_B", joinpath(W, "steps_bridge", "2020Q2_to_2020Q3", "cloud_health.jld2"))   # CMP_B: test override only
const OUTP = get(ENV, "CMP_OUT", "compare_bridge")
const F104 = joinpath(W, "seed", "ss104_2020Q3_cloud_health.jld2")
isfile(joinpath(dirname(FB), "HEALTH_PASS")) || println("WARNING: no HEALTH_PASS beside $FB (", isfile(joinpath(dirname(FB), "HEALTH_FAIL")) ? "HEALTH_FAIL" : "no marker", ")")
configure_freeset!(m, Set(fixed_at("2020Q3")))
ModelConstructors.toggle_regime!(m.parameters, 1)
FREE = ModelConstructors.get_free_para_inds(m.parameters; regime_switching = true)
A = load(FA, "cloud").particles; B = load(FB, "cloud").particles; C = load(F104, "cloud").particles; n = size(A, 2)
wA = Weights(A[:, n] ./ sum(A[:, n])); wB = Weights(B[:, n] ./ sum(B[:, n])); wC = Weights(C[:, n] ./ sum(C[:, n]))
essA = 1 / sum(abs2, wA); essB = 1 / sum(abs2, wB)
# prior sd of each slot (regime-specific prior if the parameter has one)
function prior_sd(s)
    sym = SLOTSYMS[s]; mt = match(r"^(.*)_reg(\d+)$", string(sym))
    key, r = mt === nothing ? (sym, 1) : (Symbol(mt[1]), parse(Int, mt[2]))
    p = m[key]
    pr = (haskey(p.regimes, :prior) && haskey(p.regimes[:prior], r)) ? p.regimes[:prior][r] : p.prior
    d = get(pr)
    sd = try std(d) catch; NaN end
    isfinite(sd) && return sd
    # heavy-tailed priors without a finite sd (e.g. RootInverseGamma with ν <= 2): robust scale IQR / 1.349
    x = rand(MersenneTwister(s), d, 200_000)
    return (quantile(x, 0.75) - quantile(x, 0.25)) / 1.349
end
rows = []
for s in FREE
    μa, μb, μc = mean(A[:, s], wA), mean(B[:, s], wB), mean(C[:, s], wC)
    sa, sb = std(A[:, s], wA; corrected = false), std(B[:, s], wB; corrected = false)
    ps = prior_sd(s)
    push!(rows, (slot = string(SLOTSYMS[s]), mean_A = μa, mean_B = μb, mean_104 = μc, sd_A = sa, sd_B = sb, prior_sd = ps,
                 dmean_prior_sd = (μb - μa) / ps, dmean_post_sd = (μb - μa) / sqrt((sa^2 + sb^2) / 2), sd_ratio = sb / sa,
                 dmean104_prior_sd = (μa - μc) / ps))
end
df = DataFrame(rows); sort!(df, :dmean_post_sd, by = abs, rev = true)
CSV.write(joinpath(W, "out", OUTP * "_params.csv"), df)
lA = A[:, n-4]; lB = B[:, n-4]
qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
qA = quantile(lA, wA, qs); qB = quantile(lB, wB, qs)
d3, _ = rt_data_pinned(m, Date(2020, 9, 30))
rec = [LIK(m, B[i, 1:n-5], d3) - lB[i] for i in rand(MersenneTwister(5), 1:size(B, 1), 20)]
S104 = CSV.read("/data/dsge_data_dir/estimation-AI/work/rtwalk104/rtwalk104_summary.csv", DataFrame)
S114f = joinpath(W, "rtwalk114_bridge_summary.csv")
S114 = isfile(S114f) ? CSV.read(S114f, DataFrame) : DataFrame()
mdd(S, st) = (r = findfirst(==(st), S.step); r === nothing ? NaN : Float64(S.log_mdd_incr[r]))
open(joinpath(W, "out", OUTP * ".md"), "w") do io
    println(io, "# ss114 2020Q3: reweighted seed (A) vs bridged cloud (B)\n\n", now(), "\n")
    println(io, "A = `$FA` (ESS $(round(essA, digits = 1)));  B = `$FB` (ESS $(round(essB, digits = 1)))\n")
    println(io, "## Log MDD increments\n\n| step | ss104 (rtwalk104) | ss114 bridge | ss114 − ss104 | expected |\n|---|---|---|---|---|")
    for (st, lab, ex) in ((2, "2020Q1 → 2020Q2", "≈ +7.84 (gap pseudo-obs.)"), (3, "2020Q2 → 2020Q3", "≈ 0"))
        a = mdd(S104, st); b = isempty(S114) ? NaN : mdd(S114, st)
        println(io, @sprintf("| %s | %.4f | %.4f | %+.4f | %s |", lab, a, b, b - a, ex))
    end
    println(io, "\n## 2020Q3 loglh (stored = ss114 on vintage 210110)\n\n| quantile | A (weighted) | B | B − A |\n|---|---|---|---|")
    for (q, a, b) in zip(qs, qA, qB); println(io, @sprintf("| %.2f | %.3f | %.3f | %+.3f |", q, a, b, b - a)); end
    println(io, @sprintf("\nmean A %.3f, B %.3f; B stored loglh recomputed at 20 particles: max |Δ| %.3g\n", mean(lA, wA), mean(lB, wB), maximum(abs.(rec))))
    println(io, "## Parameters (free at 2020Q3: $(length(FREE)))\n")
    println(io, @sprintf("|Δmean|/prior sd: max %.3f (%s), median %.4f;  |Δmean|/posterior sd: max %.3f (%s), median %.3f, #>0.2 %d, #>0.5 %d;  sd_B/sd_A: %.3f .. %.3f\n",
            maximum(abs.(df.dmean_prior_sd)), df.slot[argmax(abs.(df.dmean_prior_sd))], median(abs.(df.dmean_prior_sd)),
            maximum(abs.(df.dmean_post_sd)), df.slot[1], median(abs.(df.dmean_post_sd)), count(>(0.2), abs.(df.dmean_post_sd)),
            count(>(0.5), abs.(df.dmean_post_sd)), minimum(df.sd_ratio), maximum(df.sd_ratio)))
    println(io, "| slot | mean A | mean B | sd A | sd B | Δmean / prior sd | Δmean / post sd | sd B / sd A |\n|---|---|---|---|---|---|---|---|")
    for r in eachrow(df[1:min(25, nrow(df)), :])
        println(io, @sprintf("| %s | %.4f | %.4f | %.4f | %.4f | %+.4f | %+.3f | %.3f |", r.slot, r.mean_A, r.mean_B, r.sd_A, r.sd_B, r.dmean_prior_sd, r.dmean_post_sd, r.sd_ratio))
    end
    println(io, "\n(all $(nrow(df)) rows: out/compare_bridge_params.csv; sorted by |Δmean / posterior sd|)")
end
println(read(joinpath(W, "out", OUTP * ".md"), String))
