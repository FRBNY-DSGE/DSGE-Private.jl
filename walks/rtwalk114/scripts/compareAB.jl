# RTWALK114: option A vs option B at sample end 2020Q4 (Marco, 9 Nov).
#   A = steps/2020Q3_to_2020Q4/cloud_health.jld2     (sequential walk: reweighted ss114 2020Q3 seed, bridged one quarter)
#   B = steps_modelbridge/2020Q4/cloud_health.jld2   (model bridge ss104 -> ss114 at 2020Q4 from the ss104 walk cloud)
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
const FA = get(ENV, "CMP_A", joinpath(W, "steps", "2020Q3_to_2020Q4", "cloud_health.jld2"))
const FB = get(ENV, "CMP_B", joinpath(W, "steps_modelbridge", "2020Q4", "cloud_health.jld2"))
const OUTP = get(ENV, "CMP_OUT", "compare_AB_2020Q4")
const F104 = "/data/dsge_data_dir/estimation-AI/work/rtwalk104/steps/2020Q3_to_2020Q4/cloud_health.jld2"
isfile(joinpath(dirname(FB), "HEALTH_PASS")) || println("WARNING: no HEALTH_PASS beside $FB (", isfile(joinpath(dirname(FB), "HEALTH_FAIL")) ? "HEALTH_FAIL" : "no marker", ")")
configure_freeset!(m, Set(fixed_at("2020Q4")))
isfile(joinpath(dirname(FA), "HEALTH_PASS")) || println("WARNING: no HEALTH_PASS beside $FA")
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
d3, _ = rt_data_pinned(m, Date(2020, 12, 31))
rec = [LIK(m, B[i, 1:n-5], d3) - lB[i] for i in rand(MersenneTwister(5), 1:size(B, 1), 20)]
S104 = CSV.read("/data/dsge_data_dir/estimation-AI/work/rtwalk104/rtwalk104_summary.csv", DataFrame)
rdcsv(f) = isfile(f) ? CSV.read(f, DataFrame) : DataFrame(step = Int[], log_mdd_incr = Float64[], stages = Int[], bridge_wall_h = Float64[], health_wall_h = Float64[], verdict = String[])
S114 = rdcsv(joinpath(W, "rtwalk114_summary.csv"))                 # option A (step 4)
SB = rdcsv(joinpath(W, "rtwalk114_modelbridge_summary.csv"))        # option B (step 4 = 2020Q4)
mdd(S, st) = (r = findfirst(==(st), S.step); r === nothing ? NaN : Float64(S.log_mdd_incr[r]))
sd0 = load(joinpath(W, "seed", "ss114_2020Q3_seed.jld2")); P0 = sd0["cloud"].particles; n0 = size(P0, 2)
dl = P0[:, n0-4] .- sd0["loglh_104"]; c_rw = maximum(dl) + log(mean(exp.(dl .- maximum(dl))))   # log p114(Y3) − log p104(Y3)
incrA = mdd(S114, 4); incr104 = mdd(S104, 4); mddB = mdd(SB, 4)
predB = c_rw + incrA - incr104
open(joinpath(W, "out", OUTP * ".md"), "w") do io
    println(io, "# ss114 at 2020Q4: option A (walk) vs option B (model bridge)\n\n", now(), "\n")
    println(io, "A = `$FA` (ESS $(round(essA, digits = 1)));  B = `$FB` (ESS $(round(essB, digits = 1)))\n")
    println(io, "## Log MDD\n\nB gives log p114(Y_2020Q4) − log p104(Y_2020Q4) directly. Via A: [log p114(Y_2020Q3) − log p104(Y_2020Q3)] (2020Q3 reweight constant) + A's 2020Q4 increment − ss104's 2020Q4 increment.\n")
    println(io, @sprintf("| quantity | value |\n|---|---|\n| 2020Q3 reweight constant log mean exp(L114 − L104) | %.5f |\n| A increment 2020Q3 → 2020Q4 (ss114) | %.5f |\n| ss104 walk increment 2020Q3 → 2020Q4 | %.5f |\n| implied log p114 − log p104 at 2020Q4 (via A) | %.5f |\n| **B: log p114 − log p104 at 2020Q4** | **%.5f** |\n| B − implied | %+.5f |", c_rw, incrA, incr104, predB, mddB, mddB - predB))
    println(io, "\n## 2020Q4 loglh (stored = ss114 on vintage 210410)\n\n| quantile | A | B | B − A |\n|---|---|---|---|")
    for (q, a, b) in zip(qs, qA, qB); println(io, @sprintf("| %.2f | %.3f | %.3f | %+.3f |", q, a, b, b - a)); end
    println(io, @sprintf("\nmean A %.3f, B %.3f; B stored loglh recomputed at 20 particles: max |Δ| %.3g\n", mean(lA, wA), mean(lB, wB), maximum(abs.(rec))))
    println(io, "## Parameters (free at 2020Q4: $(length(FREE)); dmean104 = A vs ss104 cloud in the CSV)\n")
    println(io, @sprintf("|Δmean|/prior sd: max %.3f (%s), median %.4f;  |Δmean|/posterior sd: max %.3f (%s), median %.3f, #>0.2 %d, #>0.5 %d;  sd_B/sd_A: %.3f .. %.3f\n",
            maximum(abs.(df.dmean_prior_sd)), df.slot[argmax(abs.(df.dmean_prior_sd))], median(abs.(df.dmean_prior_sd)),
            maximum(abs.(df.dmean_post_sd)), df.slot[1], median(abs.(df.dmean_post_sd)), count(>(0.2), abs.(df.dmean_post_sd)),
            count(>(0.5), abs.(df.dmean_post_sd)), minimum(df.sd_ratio), maximum(df.sd_ratio)))
    println(io, "| slot | mean A | mean B | sd A | sd B | Δmean / prior sd | Δmean / post sd | sd B / sd A |\n|---|---|---|---|---|---|---|---|")
    for r in eachrow(df[1:min(25, nrow(df)), :])
        println(io, @sprintf("| %s | %.4f | %.4f | %.4f | %.4f | %+.4f | %+.3f | %.3f |", r.slot, r.mean_A, r.mean_B, r.sd_A, r.sd_B, r.dmean_prior_sd, r.dmean_post_sd, r.sd_ratio))
    end
    println(io, "\n(all $(nrow(df)) rows: out/$(OUTP)_params.csv; sorted by |Δmean / posterior sd|)")
end
println(read(joinpath(W, "out", OUTP * ".md"), String))

# ---- GATE for the all-t option-B runs (Marco, 9 Nov; revised 10 Nov): B at 2020Q4 agrees with A, health passes (A and B at 2020Q4, B at 2022Q2)
# Thresholds (fixed 9 Nov, before the results):
#   MDD:     |B − (2020Q3 reweight constant + A increment − ss104 increment)| <= 0.30   (SMC MDD noise ≈ 0.05–0.1)
#   params:  max |Δmean| / posterior sd <= 0.35 and median <= 0.10 over the free parameters at 2020Q4;
#            sd_B / sd_A within [0.80, 1.25] for every parameter
#   health:  HEALTH_PASS for A at 2020Q4, B at 2020Q4 and B at 2022Q2 (checked by gateAB.sh before this script runs)
#   cost:    (dropped 10 Nov, Marco: arbitrary) — B 2022Q2 stages and wall are reported, not gated
g = String[]
abs(mddB - predB) <= 0.30 || push!(g, @sprintf("MDD: B − implied = %+.4f (|·| > 0.30)", mddB - predB))
maximum(abs.(df.dmean_post_sd)) <= 0.35 || push!(g, @sprintf("params: max |Δmean|/post sd %.3f > 0.35 (%s)", maximum(abs.(df.dmean_post_sd)), df.slot[1]))
median(abs.(df.dmean_post_sd)) <= 0.10 || push!(g, @sprintf("params: median |Δmean|/post sd %.3f > 0.10", median(abs.(df.dmean_post_sd))))
# sd-ratio test on the NON-covid parameters only (Marco, 10 Nov: the COVID slots are weakly identified, as the near-identical
# MDDs show, so their posterior sds are not robust across approaches; = the walk health gate's treatment of the covid group).
# The covid ratios are reported as information.
COVSYM = Set(string.(only([s for (l, s) in walk_block_groups(m) if l == "covid"])))
nc = [!(s in COVSYM) for s in df.slot]
(minimum(df.sd_ratio[nc]) >= 0.80 && maximum(df.sd_ratio[nc]) <= 1.25) || push!(g, @sprintf("params: non-covid sd ratio %.3f .. %.3f outside [0.80, 1.25]", minimum(df.sd_ratio[nc]), maximum(df.sd_ratio[nc])))
println(@sprintf("info: covid-group sd ratio %.3f .. %.3f (%d slots; not gated)", minimum(df.sd_ratio[.!nc]), maximum(df.sd_ratio[.!nc]), count(!, nc)))
r10 = findfirst(==(10), SB.step)
if r10 === nothing
    get(ENV, "CMP_PRE", "0") == "1" ? println("CMP_PRE: B 2022Q2 not finished yet (not gated in this preliminary run)") : push!(g, "no option-B row for 2022Q2 in rtwalk114_modelbridge_summary.csv")
else
    st10 = SB.stages[r10]; wh10 = Float64(SB.bridge_wall_h[r10]) + Float64(SB.health_wall_h[r10])
    String(SB.verdict[r10]) == "PASS" || push!(g, "health: B 2022Q2 verdict $(SB.verdict[r10])")
    println(@sprintf("info: B 2022Q2 %d stages, bridge + health wall %.2f h", st10, wh10))
end
open(joinpath(W, "out", OUTP * "_gate.txt"), "w") do io
    println(io, isempty(g) ? "GATE PASS" : "GATE FAIL")
    foreach(x -> println(io, x), g)
    println(io, @sprintf("values: B−implied %+.4f; max/median |Δmean|/post sd %.3f / %.3f; sd ratio all %.3f..%.3f, non-covid %.3f..%.3f, covid %.3f..%.3f (covid not gated)", mddB - predB,
            maximum(abs.(df.dmean_post_sd)), median(abs.(df.dmean_post_sd)), minimum(df.sd_ratio), maximum(df.sd_ratio), minimum(df.sd_ratio[nc]), maximum(df.sd_ratio[nc]), minimum(df.sd_ratio[.!nc]), maximum(df.sd_ratio[.!nc])))
end
println(read(joinpath(W, "out", OUTP * "_gate.txt"), String))
