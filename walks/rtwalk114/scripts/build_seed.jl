# RTWALK114 seed: the ss104 2020Q3 real-time cloud (rtwalk104 step 3, health-checked, HEALTH_PASS), mapped to ss114 and
# importance-reweighted to the ss114 posterior at 2020Q3.
#
#   source  seed/ss104_2020Q3_cloud_health.jld2 = byte copy of work/rtwalk104/steps/2020Q2_to_2020Q3/cloud_health.jld2
#           (sha256 pinned below), 15,000 equal-weight particles of p_104(θ, ρ_ait_rm | Y_2020Q3)
#   map     the parameter vectors are the same (130 parameters, 184 values, same order: asserted). Every slot ss114
#           fixes (incl. the 8 AIT slots) and every walk slot fixed at 2020Q3 is set to its ss114 fixed value. In the
#           source only ρ_ait_rm (free in ss104 from 2020Q2) differs from it (asserted / reported).
#   weight  w_i ∝ exp(loglh_114(θ_i) − loglh_104(θ_i, ρ_i)): proposal p(θ)p(ρ)L_104(θ,ρ), target p(θ)L_114(θ) × p(ρ)
#           (ρ then dropped) — the ss104 prior of ρ_ait_rm cancels. loglh_104 = the source's stored loglh (checked
#           against an ss104 build on this tree at NCHK particles).
#   GATE    (Marco, 8 Nov 2026) PASS iff all loglh finite, ESS/N >= 0.90 AND max − min of (loglh_114 − loglh_104)
#           <= 0.1. On PASS the seed is written (seed/ss114_2020Q3_seed.jld2) with SEED_OK (sha256 + gate values);
#           on FAIL no seed is written, any old seed / SEED_OK is removed, SEED_GATE_FAIL records the values, exit 3.
#           (Fallback, not built: start ss114 from the ss104 2020Q1 cloud and bridge 2020Q1 -> Q2 -> Q3.)
#   stored  loglh = loglh_114 on the 2020Q3 vintage, logprior = the ss114 prior (free set of 2020Q3), old_loglh = 0
#           (as every cloud_health.jld2; SMC overwrites it with loglh at the bridge start), weight = w (mean 1), ESS.
#           c_blocks / c_block_labels carried from the source by label. loglh_104 is kept.
# usage: run.sh build_seed.jl with NW workers (julia -p NW)
using Distributed
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk114/scripts/setup.jl")
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk114/scripts/rt_data.jl")
using SHA, Printf, Statistics, StatsBase
# RTWALK114 (8 Nov, Marco): SEED_T = 2020Q3 (default; the main-chain seed) or 2020Q1 (the validation-bridge seed, source
# work/rtwalk104/steps/2019Q4_to_2020Q1/cloud_health.jld2; ss114 = ss104 there up to rounding, so the 2020Q1 gate adds
# max |loglh_114 − loglh_104| <= 1e-8). Everything below is the same for both.
const SEED_T = get(ENV, "SEED_T", "2020Q3"); @assert SEED_T in ("2020Q3", "2020Q1")
const SRCINFO = Dict("2020Q3" => ("33350cf1000059d9a6b628ab37539cc1d6bf18ce4beafef0fb07969458d13383", "2020Q2_to_2020Q3"),
                     "2020Q1" => ("aa251e1afb0e516b966529b4dc9383cff6511c3448cf15f15aabbd27c37eabf1", "2019Q4_to_2020Q1"))
const SRC = joinpath(W, "seed", "ss104_$(SEED_T)_cloud_health.jld2")
const SRC_SHA = SRCINFO[SEED_T][1]
const SRC_ORIG = "/data/dsge_data_dir/estimation-AI/work/rtwalk104/steps/$(SRCINFO[SEED_T][2])/cloud_health.jld2"
const DST = joinpath(W, "seed", "ss114_$(SEED_T)_seed.jld2")
const OKF = joinpath(W, "seed", SEED_T == "2020Q3" ? "SEED_OK" : "SEED_OK_$(SEED_T)"); const FAILF = joinpath(W, "seed", SEED_T == "2020Q3" ? "SEED_GATE_FAIL" : "SEED_GATE_FAIL_$(SEED_T)")
const GATE_MAXABS = SEED_T == "2020Q1" ? 1e-8 : Inf     # 2020Q1: ss114 = ss104 up to rounding
const GATE_ESSN_MIN = 0.90; const GATE_RANGE_MAX = 0.1
const NCHK = parse(Int, get(ENV, "NCHK", "20"))
const T3 = qdate(SEED_T)        # (name kept from the 2020Q3 version)
println("host ", gethostname(), "  nworkers ", nworkers(), "  ", now())
@assert bytes2hex(open(sha256, SRC)) == SRC_SHA "source seed copy changed"
isfile(SRC_ORIG) && println("rtwalk104 original sha256 ", bytes2hex(open(sha256, SRC_ORIG)) == SRC_SHA ? "equal" : "DIFFERS (copy is the pinned one)")
@assert isfile(joinpath(W, "seed", "ss104_$(SEED_T)_HEALTH_PASS"))

# ---- data (pins the ss114 2020Q3 matrix; checked against rtwalk104's pin minus the gap rows)
d3, v3 = rt_data_pinned(m, T3)
@everywhere workers() D3 = rt_data_pinned(m, $T3)[1]
D3 = d3
println("$SEED_T vintage ", v3, ", ", size(d3, 1), " x ", size(d3, 2))
const FIX3 = fixed_at(SEED_T)
@everywhere configure_freeset!(m, Set($FIX3))
println("ss114 free at $SEED_T: ", nfree(), " (fixed walk slots ", length(FIX3), ")")

# ---- the ss104 build on this tree (= rtwalk104's build): layout and defaults
m104 = build_walk_model(Date(2026, 9, 30), "ss104"; avg = false, antmu = 0)
m104 <= Setting(:likelihood_insample_regimes_only, true)
m104 <= Setting(:likelihood_truncated_regimes, true)
ModelConstructors.toggle_regime!(m104.parameters, 1)
V104 = copy(ModelConstructors.get_values(m104.parameters))
# the rtwalk104 free set of SEED_T on m104 (a raw ss104 build frees every CAND38 slot; e.g. ρ_meas_π[2] sits at 0.0 in
# a 2020Q1 ss104 particle, outside its free bounds; verifier I2). ρ_ait_rm (no regimes) can stay free: its value is in bounds.
let fx = Set(load(joinpath(RTW104_PIN_DIR, "freeset.jld2"), "fixed_at")[SEED_T])
    for key in unique(first.(collect(fx)))
        p = m104[key]; isempty(p.regimes) && continue
        p.regimes[:fixed] = OrderedDict{Int, Bool}(r => (regfixed(p, r) || (key, r) in fx) for r in 1:length(p.regimes[:value]))
    end
    ModelConstructors.toggle_regime!(m104.parameters, 1)
end
SYM104 = let s = Symbol[θ.key for θ in m104.parameters]
    for θ in m104.parameters
        if !isempty(θ.regimes) && haskey(θ.regimes, :value); for i in 2:length(θ.regimes[:value]); push!(s, Symbol(θ.key, "_reg$i")); end; end
    end; s
end
@assert SYM104 == SLOTSYMS "ss104 and ss114 parameter layouts differ"
dv = findall(V104 .!= VALS0)
println("defaults ss104 vs ss114: ", isempty(dv) ? "identical (184 values)" : "differ at $(SLOTSYMS[dv])")
@assert isempty(dv)

# ---- map
src = load(SRC); c = src["cloud"]; P = copy(c.particles); NP, NC = size(P); n = NC
@assert NC - 5 == 184 && c.tempering_schedule[end] >= 1.0
w0 = P[:, end]; @assert all(==(w0[1]), w0) "source weights not equal"
setcols = sort(unique(vcat(BASE_FIXED_SLOTS, [slot_index(k, r) for (k, r) in FIX3])))
changed = [s for s in setcols if any(!=(VALS0[s]), P[:, s])]
println("columns set to their ss114 fixed value: ", length(setcols), "; of these not already there in the source: ",
        [string(SLOTSYMS[s]) for s in changed])
for s in changed
    println(@sprintf("  %s: source mean %.5f sd %.5f min %.5f max %.5f -> %.5f", SLOTSYMS[s], mean(P[:, s]), std(P[:, s]),
                     minimum(P[:, s]), maximum(P[:, s]), VALS0[s]))
end
@assert Set(SLOTSYMS[changed]) ⊆ Set([:ρ_ait_rm]) "unexpected columns differ from the ss114 fixed values: $(SLOTSYMS[changed])"
l104 = P[:, n-4]
P[:, setcols] .= VALS0[setcols]'

# ---- check stored loglh_104 against ss104 on this tree (rtwalk104's ss104 pin of 2020Q3: 28 rows)
p104 = load(joinpath(RTW104_PIN_DIR, "rt_ss104_$(SEED_T)_vint=$(v3)_revois.jld2"), "data")
chk = unique(vcat(argmax(l104), argmin(l104), rand(MersenneTwister(31), 1:NP, NCHK)))[1:NCHK]
d104 = [LIK(m104, c.particles[i, 1:n-5], p104) - l104[i] for i in chk]
println(@sprintf("ss104 recomputed on this tree vs stored loglh_104 at %d particles: max |Δ| %.3g (bitwise %d)", length(chk), maximum(abs.(d104)), count(==(0.0), d104)))
@assert maximum(abs.(d104)) <= 1e-9 "stored loglh_104 does not reproduce"

# ---- ss114 loglh and logprior on 2020Q3, all particles
@everywhere function lik_prior(θ)
    L = LIK(m, θ, D3)
    ModelConstructors.update!(m.parameters, θ); lp = ModelConstructors.prior(m.parameters); ModelConstructors.toggle_regime!(m.parameters, 1)
    return (L, lp)
end
t0 = time()
res = pmap(lik_prior, [P[i, 1:n-5] for i in 1:NP]; batch_size = 50)
println(@sprintf("ss114 loglh + logprior for %d particles in %.1f s on %d workers", NP, time() - t0, nworkers()))
l114 = first.(res); lp114 = last.(res)
Δ = l114 .- l104
fin = all(isfinite, l114) && all(isfinite, lp114)
lw = Δ .- maximum(Δ); wt = exp.(lw); wt .*= NP / sum(wt)
essN = sum(wt)^2 / sum(abs2, wt) / NP
rng = maximum(Δ) - minimum(Δ)
q = quantile(Δ, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
println(@sprintf("loglh_114 − loglh_104: mean %.6f sd %.6f min %.6f max %.6f max−min %.6f", mean(Δ), std(Δ), minimum(Δ), maximum(Δ), rng))
println("  quantiles 0/1/5/25/50/75/95/99/100%: ", join([@sprintf("%.5f", x) for x in q], " "))
println("  corr(Δ, ρ_ait_rm source) ", round(cor(Δ, c.particles[:, slot_index(:ρ_ait_rm, 1)]), digits = 3),
        "; corr(Δ, loglh_104) ", round(cor(Δ, l104), digits = 3))
println(@sprintf("ESS/N %.8f (ESS %.1f); max weight / mean %.5f", essN, essN * NP, maximum(wt)))
lpd = lp114 .- P[:, n-3]
println(@sprintf("logprior ss114 − ss104 stored: mean %.4f sd %.4g (the ρ_ait_rm prior drops out)", mean(lpd), std(lpd)))
# Δ distribution to out/
mkpath(joinpath(W, "out"))
CSV.write(joinpath(W, "out", SEED_T == "2020Q3" ? "seed_reweight.csv" : "seed_reweight_$(SEED_T).csv"), DataFrame(particle = 1:NP, loglh_104 = l104, loglh_114 = l114, d = Δ,
          rho_ait_rm_104 = c.particles[:, slot_index(:ρ_ait_rm, 1)], weight = wt))

gate = Dict("essN" => essN, "mean" => mean(Δ), "sd" => std(Δ), "range" => rng, "min" => minimum(Δ), "max" => maximum(Δ),
            "finite" => fin, "essN_min" => GATE_ESSN_MIN, "range_max" => GATE_RANGE_MAX, "maxabs" => maximum(abs.(Δ)), "maxabs_max" => GATE_MAXABS,
            "pass" => fin && essN >= GATE_ESSN_MIN && rng <= GATE_RANGE_MAX && maximum(abs.(Δ)) <= GATE_MAXABS)
gl = @sprintf("ESS/N %.8f (gate >= %.2f)  sd %.3g  max-min %.3g (gate <= %.2f)  max|Δ| %.3g (gate <= %.0e)  finite %s", essN, GATE_ESSN_MIN, std(Δ), rng, GATE_RANGE_MAX, maximum(abs.(Δ)), GATE_MAXABS, fin)
if !gate["pass"]
    for f in (DST, OKF); isfile(f) && rm(f); end
    write(FAILF, "SEED GATE FAIL $(now())\n$gl\n")
    println("SEED GATE FAIL: ", gl, "\nno seed written (fallback: ss114 walk from the ss104 2020Q1 cloud, bridges 2020Q1 -> Q2 -> Q3)")
    exit(3)
end
isfile(FAILF) && rm(FAILF)
println("SEED GATE PASS: ", gl)

# ---- write the seed
P[:, n-4] = l114; P[:, n-3] = lp114; P[:, n-2] .= 0.0; P[:, n] = wt
cl = deepcopy(c); cl.particles = P; cl.ESS = [essN * NP]
labs = haskey(src, "c_block_labels") ? Vector{String}(src["c_block_labels"]) : ["ρ_σ_w", "rest"]
isfile(OKF) && rm(OKF)
tmp = DST * ".tmp"
jldsave(tmp; cloud = cl, c_blocks = Vector{Float64}(src["c_blocks"]), c_block_labels = labs, loglh_104 = l104,
        seed_kind = "ss104_reweighted_$(SEED_T)", gate = gate, source = SRC, sha_source = SRC_SHA, source_orig = SRC_ORIG,
        data_vint = v3, mapped_columns = string.(SLOTSYMS[changed]),
        note = "RTWALK114 seed: rtwalk104 $SEED_T cloud_health mapped to ss114 (AIT slots at the ss114 fixed values), weights exp(loglh_114 - loglh_104) (mean 1), loglh = ss114 on vintage $v3, logprior = ss114, old_loglh = 0",
        built_at = string(now()))
mv(tmp, DST; force = true)
# re-read and check
cc = load(DST, "cloud"); @assert isequal(cc.particles, P)
for i in chk[1:5]; @assert LIK(m, cc.particles[i, 1:n-5], d3) === l114[i]; end
sh = bytes2hex(open(sha256, DST))
write(OKF, "$sh  $(basename(DST))\n$gl\nbuilt $(now())\n")
println("seed written: ", DST, "\nsha256 ", sh, "\nc_blocks ", labs, " = ", src["c_blocks"])
