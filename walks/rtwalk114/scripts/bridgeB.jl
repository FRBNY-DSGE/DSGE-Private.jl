# =============================================================================
# RTWALK114 option B: MODEL bridge ss104 -> ss114 at a FIXED sample end t (Marco, 9 Nov 2026).
#   old_cloud = the ss104 walk cloud at t (work/rtwalk104/steps/<t-1>_to_<t>/cloud_health.jld2; needs its HEALTH_PASS)
#   old_model = ss104 (rtwalk104's build, rtwalk104's free set at t), old_data = rtwalk104's ss104 pin of t (28 rows)
#   model     = mB: ss114 (the walk build, the walk's free set at t) PLUS, as auxiliary free variables with their ss104
#               priors, the AIT slots that ss104 frees at t (the ss104 Parameter objects are copied in). So mB's free set
#               = ss104's at t, slot for slot, and π_0 = L_104 p(θ) p(a) = the ss104 posterior exactly.
#   data      = the ss114 pin of t (26 rows), the SAME vintage.
#   Tempered target π_φ ∝ L_114(θ)^φ L_104(θ, a)^(1−φ) p(θ) p(a); incremental weights (L_114 / L_104)^Δφ: the AIT prior
#   cancels. MH moves a for φ < 1 (in the covid group, as ss104's walk does). At φ = 1, a does not enter L_114 (up to
#   rounding: zero-variance states), so the θ-marginal is the ss114 posterior: a is set to ss114's fixed values, loglh and
#   logprior are recomputed on the walk model (reported max |Δ|), and the walk's health check + gate run on that cloud.
#   log MDD increment = log p_114(Y_t) − log p_104(Y_t).
# Machinery: DSGE.estimate(mB, data; old_cloud, old_data, old_model = m104): smc.jl's _my_old_likelihood_dsge copies
#   mB's values into m104 by position and asserts equality (same layout, same free set, same fixed defaults: holds);
#   key_del / reg_del are empty; the :preprocessed_transitions memo is cleared on old_model (DSGE edaedbe6f, in this tree).
#   old_model !== mB, so no system reuse: each proposal costs L_114 + L_104.
# Env: SAMPLE_END (t, required), SEED_CLOUD (default: the rtwalk104 cloud at t), FRBNY_NW / NW, TAG (rtwalk114B),
#   H_PASSES/H_NTUNE/H_NCHUNK/H_POLISH, PHI_TARGET (0.95), CONTINUE_FROM (-1 = newest loadable stage), DRY_RUN, SMOKE
#   (NSMOKE particles, NPHI_SMOKE fixed stages, SCRATCH), HEALTH_REDO. Output: steps_modelbridge/<t>/ (health files,
#   cloud_health.jld2, cloud_B_mapped.jld2, B_info.txt), summary rtwalk114_modelbridge_summary.csv.
# Exit 0 = PASS, 2 = HEALTH_FAIL, 3 = the ss104 step at t has no HEALTH_PASS (nothing done).
# =============================================================================
const T0 = time()
const LB = ENV["SAMPLE_END"]
const SMOKE = get(ENV, "SMOKE", "0") == "1"
const W0 = "/data/dsge_data_dir/estimation-AI/work/rtwalk114"
const R104 = "/data/dsge_data_dir/estimation-AI/work/rtwalk104"
qd(s) = (y = parse(Int, s[1:4]); q = parse(Int, s[6:6]); Dates.lastdayofquarter(Dates.Date(y, 3q, 1)))
using Dates
# FULL mode (SAMPLE_END=full; Marco, 10 Nov): the full-sample bridge ss104v3 -> ss114 on vintage 261013 (the frozen data
# of both estimations, through 2026Q2), every parameter free (101 / 93), the estimations' 2 MH steps and seed 20261028.
# Seed: work/ss104v3/health/cloud_health.jld2, needs "Verdict (walk thresholds): PASS" in work/ss104v3/health/health.md.
const FULL = LB == "full"
const TB = FULL ? Date(2026, 6, 30) : qd(LB); const TPREV = Dates.lastdayofquarter(Dates.firstdayofquarter(TB) - Dates.Day(1))
const LPREV = string(Dates.year(TPREV)) * "Q" * string(Dates.quarterofyear(TPREV))
const V3 = "/data/dsge_data_dir/estimation-AI/work/ss104v3"; const S114 = "/data/dsge_data_dir/estimation-AI/work/ss114"
const SEED_CLOUD = get(ENV, "SEED_CLOUD", FULL ? joinpath(V3, "health", "cloud_health.jld2") : joinpath(R104, "steps", "$(LPREV)_to_$(LB)", "cloud_health.jld2"))
if FULL
    hm = joinpath(V3, "health", "health.md")
    if !SMOKE && !(isfile(hm) && occursin("Verdict (walk thresholds): PASS", read(hm, String)))
        println("ss104v3 health check has not passed ($hm): nothing to do"); exit(3)
    end
elseif !SMOKE && !isfile(joinpath(dirname(SEED_CLOUD), "HEALTH_PASS"))
    println("ss104 walk step $(LPREV) -> $(LB) has no HEALTH_PASS ($(dirname(SEED_CLOUD))): nothing to do"); exit(3)
end
const SCR = get(ENV, "SCRATCH", "scratch_B")
const STEPDIR = joinpath(W0, SMOKE ? SCR : "", "steps_modelbridge", LB)
const SUMMARY = joinpath(W0, SMOKE ? SCR : "", "rtwalk114_modelbridge_summary.csv")
if get(ENV, "HEALTH_REDO", "0") != "1" && !SMOKE
    for f in ("HEALTH_PASS", "HEALTH_FAIL")
        isfile(joinpath(STEPDIR, f)) && (println("already checked: ", joinpath(STEPDIR, f)); exit(f == "HEALTH_PASS" ? 0 : 2))
    end
end

using Distributed
const FRBNY_NW = parse(Int, get(ENV, "FRBNY_NW", "0"))
if FRBNY_NW > 0
    using ClusterManagers, frbny
    ENV["frbnyjuliamemory"] = "8G,h_rt=" * get(ENV, "WORKER_HRT", "38:00:00")
    envdir = dirname(Base.active_project())
    mkpath(joinpath(W0, "workers")); cd(joinpath(W0, "workers"))
    addprocs_frbny(FRBNY_NW; exeflags = `--project=$envdir --startup-file=no --compiled-modules=existing`)
    cd(W0)
    @assert nworkers() == FRBNY_NW
end
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk114/scripts/setup.jl")
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk114/scripts/rt_data.jl")
@everywhere begin
    const LBX = $LB
    const FIX114 = LBX == "full" ? Tuple{Symbol, Int}[] : fixed_at(LBX)                  # full sample: all 93 free (= the ss114 estimation)
    const FIX104 = LBX == "full" ? Tuple{Symbol, Int}[] : load("/data/dsge_data_dir/estimation-AI/work/rtwalk104/pinned/freeset.jld2", "fixed_at")[LBX]   # full: all 101 (= ss104v3)
    @assert Set(FIX114) == setdiff(Set(FIX104), Set(AIT_SLOTS))
    # walk model m (AIT fixed): the health check and the φ = 1 cloud
    configure_freeset!(m, Set(FIX114))
    m <= Setting(:smc_param_blocks, walk_param_blocks(m))
    if LBX == "full"       # the estimations' SMC kernel settings (ss104v3 / ss114: 2 MH steps, seed 20261028); health uses NMH from m
        m <= Setting(:n_mh_steps_smc, 2, true, "mhsteps", ""); m <= Setting(:smc_seed, 20261028)
    end
    # old model: ss104 with rtwalk104's free set at t (= rtwalk104 configure_freeset!)
    function build_m104()
        mm = build_walk_model(Date(2026, 9, 30), "ss104"; avg = false, antmu = 0)
        mm <= Setting(:likelihood_insample_regimes_only, true)
        mm <= Setting(:likelihood_truncated_regimes, get(ENV, "TRUNC_REGIMES", "1") == "1")
        ModelConstructors.toggle_regime!(mm.parameters, 1)
        fx = Set(FIX104)
        for key in unique(first.(CAND38_104))
            p = mm[key]
            if isempty(p.regimes)
                @assert key == :ρ_ait_rm
                (key, 1) in fx && (mm <= parameter(p.key, p.value, p.valuebounds, p.transform_parameterization, p.transform, get(p.prior);
                                                    fixed = true, description = p.description, tex_label = p.tex_label))
            else
                p.regimes[:fixed] = OrderedDict{Int, Bool}(r => (regfixed(p, r) || (key, r) in fx) for r in 1:length(p.regimes[:value]))
            end
        end
        ModelConstructors.toggle_regime!(mm.parameters, 1)
        @assert nfree(mm) == 101 - length(FIX104)
        @assert ModelConstructors.get_values(mm.parameters) == VALS0
        return mm
    end
    const m104 = build_m104()
    # bridge model mB: ss114 walk build + the AIT slots ss104 frees at t, as auxiliary variables (ss104 Parameter objects)
    function build_mB()
        mm = build_walk_model(Date(2026, 9, 30), SUBSPEC; avg = false, antmu = 0)
        mm <= Setting(:likelihood_insample_regimes_only, true)
        mm <= Setting(:likelihood_truncated_regimes, get(ENV, "TRUNC_REGIMES", "1") == "1")
        ModelConstructors.toggle_regime!(mm.parameters, 1)
        configure_freeset!(mm, Set(FIX114))
        for key in unique(first.(AIT_SLOTS))
            mm <= deepcopy(m104[key])
        end
        ModelConstructors.toggle_regime!(mm.parameters, 1)
        @assert ModelConstructors.get_values(mm.parameters) == VALS0
        set_smc_settings!(mm)
        LBX == "full" && (mm <= Setting(:n_mh_steps_smc, 2, true, "mhsteps", ""); mm <= Setting(:smc_seed, 20261028))
        return mm
    end
    const mB = build_mB()
    freeinds(mm) = (ModelConstructors.toggle_regime!(mm.parameters, 1); f = ModelConstructors.get_free_para_inds(mm.parameters; regime_switching = true); ModelConstructors.toggle_regime!(mm.parameters, 1); f)
    const FREE_B = freeinds(mB); const FREE_W = freeinds(m)
    @assert FREE_B == freeinds(m104) "mB's free set differs from ss104's at $LBX"
    const AUX = sort(setdiff(FREE_B, FREE_W))     # the auxiliary AIT slots (columns)
    @assert Set(SLOTSYMS[AUX]) == Set([SLOTSYMS[slot_index(k, r)] for (k, r) in AIT_SLOTS if !((k, r) in FIX104)])
    # mB's block groups = the ss104 walk's: [ρ_σ_w], covid = CAND38 slots free in mB (incl. the aux AIT slots), rest
    const B_GROUPS = [("ρ_σ_w", [:ρ_σ_w]), ("covid", [SLOTSYMS[slot_index(k, r)] for (k, r) in CAND38_104 if slot_index(k, r) in FREE_B])]
    mB <= Setting(:smc_param_blocks, [s for (_, s) in B_GROUPS])
end
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk114/scripts/health.jl")
const B_LABELS = vcat(first.(B_GROUPS), "rest")
@assert B_LABELS == health_block_labels()
const STEPNO = FULL ? 99 : length(DSGE.quarter_range(STEP_BASE, TB)) - 1
if FULL
    # the frozen data of the two estimations (vintage 261013, 1959Q3-2026Q2): ss114 = ss104v3 minus obs_pgap/obs_ygap
    global D114F = joinpath(S114, "frozen", "data_ss114.jld2"); global D104F = joinpath(V3, "frozen", "data_ss104v3.jld2")
    @assert bytes2hex(open(sha256, D114F)) == strip(read(joinpath(S114, "frozen", "data_ss114.sha256"), String)) "ss114 frozen data changed"
    global d114 = load(D114F, "data"); global d104 = load(D104F, "data"); global V = "261013"; global f104 = basename(D104F)
    @assert size(d114) == (26, 268) && size(d104) == (28, 268)
    o104 = sort(collect(keys(m104.observables)), by = k -> m104.observables[k]); o114 = sort(collect(keys(m.observables)), by = k -> m.observables[k])
    @assert isequal(d104[[m104.observables[k] for k in o114], :], d114) "ss114 data != ss104v3 data minus the gap rows"
    @everywhere HEALTH_DATA[] = load($D114F, "data")
else
    tmp_ = rt_data_pinned(m, TB); global d114 = tmp_[1]; global V = tmp_[2]
    global f104 = only(Base.filter(x -> startswith(x, "rt_ss104_$(LB)_vint=$(V)_"), readdir(RTW104_PIN_DIR)))
    global d104 = load(joinpath(RTW104_PIN_DIR, f104), "data")
    @everywhere HEALTH_DATA[] = rt_data_pinned(m, $TB)[1]
end
const TAG = get(ENV, "TAG", "rtwalk114B")
println("="^90, "\nRTWALK114 option B (model bridge ss104 -> ss114) at sample end $LB   host ", gethostname(), "  nworkers ", nworkers(), "  SMOKE ", SMOKE, "  ", now())
println("seed (ss104 walk cloud) ", SEED_CLOUD, "\nfree: mB ", length(FREE_B), " (= ss104), walk ", length(FREE_W), "; auxiliary AIT slots: ", join(string.(SLOTSYMS[AUX]), " "))

# ---- data: new = ss114 pin of t, old = rtwalk104's ss104 pin of t (same vintage)
println("data: vintage $V; ss114 $(size(d114)), ss104 $(size(d104)) ($f104)")

# ---- settings on mB (as step.jl)
mkpath(STEPDIR)
mB <= Setting(:fcast_date, LB, true, "sampleend", "")
mB <= Setting(:use_parallel_workers, nworkers() > 0 && nprocs() > 1)
mB <= Setting(:saveroot, joinpath(W0, SMOKE ? SCR : "saveroot") * "/")
mB <= Setting(:previous_data_vintage, V)
mB <= Setting(:bridge, true, true, "bridge", "")
mB <= Setting(:tempered_update_prior_weight, 0.0, true, "priorwt", "")
mB <= Setting(:data_vintage, V)
mB <= Setting(:smc_reuse_system_new_params, false)
const PHI_TARGET = parse(Float64, get(ENV, "PHI_TARGET", "0.95"))
mB <= Setting(:adaptive_tempering_target_smc, PHI_TARGET, true, "phi", "")
INCREMENT = 1
if SMOKE
    mB <= Setting(:use_fixed_schedule, true); mB <= Setting(:n_Φ, parse(Int, get(ENV, "NPHI_SMOKE", "4"))); mB <= Setting(:λ, 1.0)
end

# ---- seed: the ss104 cloud, unchanged (fresh copy every call)
NSMOKE = parse(Int, get(ENV, "NSMOKE", "300"))
function seed_cloud()
    c = load(SEED_CLOUD, "cloud"); P = c.particles
    @assert size(P, 2) - 5 == length(VALS0)
    get(ENV, "ALLOW_PHI_LT1", "0") == "1" || @assert c.tempering_schedule[end] >= 1.0   # ALLOW_PHI_LT1: DRY_RUN tests on a stage file only
    nf = setdiff(eachindex(VALS0), FREE_B)
    @assert all(s -> all(==(VALS0[s]), P[:, s]), nf) "a slot fixed in ss104 at $LB is not at its fixed value in the seed"
    if SMOKE
        w = SMC.get_weights(c); cw = cumsum(w ./ sum(w)); u = (rand(MersenneTwister(20261109)) .+ (0:NSMOKE-1)) ./ NSMOKE
        c.particles = c.particles[[min(searchsortedfirst(cw, x), length(cw)) for x in u], :]; SMC.reset_weights!(c)
    end
    w = SMC.get_weights(c); c.ESS = [sum(w)^2 / sum(abs2, w)]
    return c
end
sc = seed_cloud(); NP = length(sc)
mB <= Setting(:n_particles, NP, true, "nparts", "")
let P = sc.particles, n = size(P, 2)
    imod = argmax(P[:, n-4] .+ P[:, n-3]); idx = unique(vcat(imod, 1, rand(MersenneTwister(9), 1:NP, 10)))
    dmax = 0.0; dl = Float64[]
    for i in idx
        st = P[i, n-4]; r104 = LIK(m104, P[i, 1:n-5], d104); r114 = LIK(mB, P[i, 1:n-5], d114)
        θf = P[i, 1:n-5]; θf[AUX] .= VALS0[AUX]; r114f = LIK(m, θf, d114)
        println(@sprintf("  seed #%d: stored L104 %.10f, recomputed L104 diff %.3g;  L114 %.6f (L114 − L104 %+.5f);  L114 with a fixed − with a free %.3g",
                         i, st, r104 - st, r114, r114 - st, r114f - r114))
        (isfinite(r104) && abs(r104 - st) <= 1e-12 * max(1.0, abs(st))) || error("SEED: stored L104 does not reproduce at particle $i")
        push!(dl, r114 - st); dmax = max(dmax, abs(r114f - r114))
    end
    println(@sprintf("  L114 − L104 at %d seed particles: mean %+.4f sd %.4f;  max |L114(a fixed) − L114(a free)| %.3g", length(dl), mean(dl), std(dl), dmax))
end

# ---- output path and stage resume (= step.jl)
savepath = rawpath(mB, "estimate", "smc_cloud.jld2", [TAG])
RAW_DIR = dirname(savepath); step_prefix = replace(basename(savepath), ".jld2" => "")
stage_files() = isdir(RAW_DIR) ? sort([parse(Int, f[length(step_prefix)+8:end-5]) for f in readdir(RAW_DIR) if startswith(f, step_prefix * "_stage=") && endswith(f, ".jld2")]) : Int[]
SMOKE && isdir(RAW_DIR) && rm(RAW_DIR; recursive = true)
mkpath(RAW_DIR)
stage_path(k) = rawpath(mB, "estimate", "smc_cloud", [TAG]) * "_stage=$(k).jld2"
stage_loads(k) = try f = stage_path(k); load(f, "cloud"); load(f, "W"); load(f, "j"); true catch; false end
function prune_stages(keep::Int)
    st = stage_files(); kept = Int[]
    for s in reverse(st); length(kept) == keep && break; stage_loads(s) && push!(kept, s); end
    for s in st; s in kept || rm(stage_path(s)); end
end
function newest_loadable_stage()
    for s in reverse(stage_files()); stage_loads(s) && return s; println("  stage file $s does not load: skipped"); end
    return 0
end
CONTINUE_FROM = parse(Int, get(ENV, "CONTINUE_FROM", "-1"))
resume_stage = CONTINUE_FROM >= 0 ? CONTINUE_FROM : newest_loadable_stage()
!SMOKE && !isfile(savepath) && prune_stages(3)
println("output: ", savepath, "\nstage files: ", stage_files(), "; resume stage ", resume_stage)
function check_pair(cl, i, what)
    P = cl.particles; n = size(P, 2)
    a = LIK(mB, P[i, 1:n-5], d114); b = LIK(m104, P[i, 1:n-5], d104)
    println("  $what #$i: loglh stored $(P[i, n-4]) recomputed $a;  old_loglh stored $(P[i, n-2]) recomputed $b")
    (a === P[i, n-4] || abs(a - P[i, n-4]) <= 1e-12 * abs(a)) && (b === P[i, n-2] || abs(b - P[i, n-2]) <= 1e-12 * abs(b)) ||
        error("$what: stored likelihoods do not reproduce at particle $i")
end

# ---- c_blocks by label (the seed's: ρ_σ_w / covid / rest of the ss104 walk = mB's groups)
cbs = jldopen(f -> (Vector{Float64}(f["c_blocks"]), haskey(f, "c_block_labels") ? Vector{String}(f["c_block_labels"]) : B_LABELS), SEED_CLOUD, "r")   # no labels (SMC stage file): the run order ρ_σ_w / covid / rest
cc, notes = carry_c_blocks(cbs[2], cbs[1], B_LABELS); foreach(x -> println("  c carry: ", x), notes)
mB <= Setting(:step_size_smc, cc[end])
mB <= Setting(:smc_initial_c_blocks, cc)
println("initial c_blocks ", B_LABELS, " = ", cc, ";  N ", NP, ", target ", PHI_TARGET)
flush(stdout)
get(ENV, "DRY_RUN", "0") == "1" && (println("DRY_RUN: checks passed (", round((time() - T0) / 60, digits = 2), " min)"); exit())

est(k) = DSGE.estimate(mB, d114; old_cloud = seed_cloud(), old_data = d104, old_model = m104,
                       verbose = :low, filestring_addl = [TAG], save_intermediate = true, intermediate_stage_increment = INCREMENT,
                       run_csminwel = false, continue_intermediate = k > 0, intermediate_stage_start = k > 0 ? k : 1)
tb = time()
if isfile(savepath) && !SMOKE
    println("bridge already complete: $savepath")
else
    resume_stage > 0 && (println("RESUMING from stage $resume_stage"); check_pair(load(stage_path(resume_stage), "cloud"), 1, "resume stage $resume_stage"))
    est(SMOKE ? 0 : resume_stage)
    SMOKE || prune_stages(3)
end
bridge_wall = time() - tb
full = load(savepath); fc = full["cloud"]; sh = full["stage_health"]
lmdd = (lm = get(full, "log_mdd_estimates", Dict()); lm isa AbstractDict ? get(lm, "smc", NaN) : NaN)
S = length(sh["stage"])
println(@sprintf("\nBRIDGE B %s done: %d stages, %.2f h; log MDD (log p_114(Y) − log p_104(Y)) = %.6f", LB, S, bridge_wall / 3600, lmdd))
check_pair(fc, 1, "final cloud")

# ---- φ = 1: the auxiliary AIT slots to ss114's fixed values; loglh / logprior on the walk model
P = copy(fc.particles); n = size(P, 2); wf = P[:, n] ./ sum(P[:, n])
for s in AUX
    s0 = sc.particles[:, s]
    println(@sprintf("  aux %s at φ = 1: mean %.4f sd %.4f (ss104 seed: mean %.4f sd %.4f) -> set to %.4f", SLOTSYMS[s], sum(wf .* P[:, s]),
                     sqrt(sum(wf .* (P[:, s] .- sum(wf .* P[:, s])) .^ 2)), mean(s0), std(s0), VALS0[s]))
end
Pa = copy(P); Pa[:, AUX] .= VALS0[AUX]'
@everywhere lp_walk(θ) = (L = LIK(m, θ, HEALTH_DATA[]); ModelConstructors.update!(m.parameters, θ); lp = ModelConstructors.prior(m.parameters); ModelConstructors.toggle_regime!(m.parameters, 1); (L, lp))
res = pmap(lp_walk, [Pa[i, 1:n-5] for i in 1:NP]; batch_size = 25)
Lw = first.(res); lpw = last.(res)
dfix = Lw .- P[:, n-4]
fin = isfinite.(Lw) .& isfinite.(P[:, n-4])
println(@sprintf("  loglh(a fixed, walk model) − loglh(a, mB) over %d particles: max |Δ| %.3g (non-finite: %d / %d)", NP, maximum(abs.(dfix[fin])), count(!isfinite, Lw), count(!isfinite, P[:, n-4])))
Pa[:, n-4] = Lw; Pa[:, n-3] = lpw
mapped = deepcopy(fc); mapped.particles = Pa
jldsave(joinpath(STEPDIR, "cloud_B_mapped.jld2"); cloud = mapped, aux_slots = string.(SLOTSYMS[AUX]), max_abs_dfix = maximum(abs.(dfix[fin])), log_mdd_114_104 = lmdd)
# stage-health per-parameter rows are mB's free parameters: keep the walk's (the aux slots are dropped)
sh2 = deepcopy(sh)
if haskey(sh2, "corr") && sh2["corr"] isa AbstractMatrix
    rows = [findfirst(==(s), FREE_B) for s in FREE_W]; sh2["corr"] = sh2["corr"][rows, :]
end
cs0 = haskey(sh, "c_blocks") ? Vector{Float64}(sh["c_blocks"][:, end]) : fill(fc.c, length(B_LABELS))
open(joinpath(STEPDIR, "B_info.txt"), "w") do io
    println(io, "sample_end=$LB vint=$V stages=$S bridge_wall_h=$(bridge_wall / 3600) log_mdd_114_minus_104=$lmdd max_abs_dfix=$(maximum(abs.(dfix[fin]))) nprocs=$(nprocs())")
    println(io, "ESS path: ", join(round.(Float64.(get(sh, "ess", fill(NaN, S))), digits = 1), " "))
    println(io, "phi path: ", join(round.(Float64.(sh["phi"]), sigdigits = 4), " "))
end
info = Dict(:step => STEPNO, :old => "ss104@$LB", :new => "ss114@$LB", :vint_old => V, :vint_new => V, :seed => SEED_CLOUD,
            :cloudfile => savepath, :summary_csv => SUMMARY, :bridge_wall_h => bridge_wall / 3600, :log_mdd_incr => lmdd)
ok, row = run_health(; cloud = mapped, seed = seed_cloud(), stage_health = sh2, thetatilde = Tuple{Symbol, Int}[], stepdir = STEPDIR, info = info,
                     cs0 = cs0, bridge_labels = B_LABELS,
                     npass = parse(Int, get(ENV, "H_PASSES", "10")), ntune = parse(Int, get(ENV, "H_NTUNE", "2")),
                     nchunk = parse(Int, get(ENV, "H_NCHUNK", "5")), polish = parse(Int, get(ENV, "H_POLISH", "0")))
println(@sprintf("option B %s: %s  (job total %.2f h on %d processes)", LB, ok ? "PASS" : "FAIL", (time() - T0) / 3600, nprocs()))
exit(ok ? 0 : 2)
