# =============================================================================
# RTWALK104 FINAL STEP (13 Nov 2026; RTWALK104.md "13 Nov 2026: final step") = a copy of step.jl (sha256 d321d6fc...)
# with the changes marked FULLSTEP (as work/rtwalk114/scripts/step_full.jl): the one step 2026Q1 (vintage 260710) ->
# 2026Q2 on the FULL-SAMPLE data (vintage 261013 = the frozen ss104v3 / ss104v3mh5 estimation matrix, pinned by
# pin_full.jl / full_data.jl); every slot free (101 = ss104v3; θ̃ = σ_condgdp[2], σ_condcorepce[2] from the prior);
# summary rtwalk104_full_summary.csv (the walk summary is left as is). Settings otherwise = steps 21-25.
# RTWALK104: ONE step of the quarter-by-quarter real-time SMC bridge walk for ss104.
#   step t -> t+1:  old_data  = real-time vintage of sample end t   (rev OIS convention, rt_data.jl)
#                   data      = real-time vintage of sample end t+1
#                   old_cloud = SEED_CLOUD (the previous step's health-checked cloud, or the 2019Q4 RT cloud)
#                   same model, μ = 0 (tempered_update_prior_weight = 0), no log_prob_old_data (bug B1)
#                   model = the 2026Q3 ss104 build with the free set of t+1 (FIXED_AT[t+1], pinned by preflight A)
#                   old_model = m (the same object; see RTWALK104.md §3 for why a separate origin-t model with the
#                   new slots fixed is NOT usable with DSGE 33ebacbb7: smc.jl's old-likelihood callback asserts
#                   that old_model's values equal the particle's, which fails for every parameter fixed in old_model)
#   new slots θ̃ = FIXED_AT[t] \ FIXED_AT[t+1]: drawn from their priors in the seed cloud before SMC (BRIDGE-NOTES B3/C3);
#   they do not enter log p(Y_t|θ) (preflight A3, bitwise), and step.jl re-checks the stored old loglh after the redraw.
# Then the health check (health.jl) runs on the same workers; the step writes HEALTH_PASS or HEALTH_FAIL into its step
# directory, and cloud_health.jld2 there is the next step's seed. A step refuses to start from a walk cloud whose step
# directory has no HEALTH_PASS.
#
# Adapted from work/rtvint104/scripts/bridge.jl (itself from MP's run_vintbridge_ss207.jl, with BW-4/BW-6/BW-8 fixes).
#
# Environment (qsub -v):
#   ORIGIN        sample end t of the OLD data, e.g. 2019Q4 (required). The step targets t+1.
#   SEED_CLOUD    path of the old cloud (required). JLD2 with key "cloud" (SMC.Cloud), loglh column = log p(Y_t|θ).
#   NW            workers (`julia -p NW`, run.sh)
#   TAG           estim string (default rtwalk104)
#   CONTINUE_FROM -1 (default) auto-resume from the NEWEST stage file that loads (cloud, W, j); 0 fresh; k stage k
#   INCREMENT     stage-file increment (default 1; stage files older than the newest KEEP_STAGES = 3 are deleted at the
#                 start of a job and after the bridge completes)
#   REUSE_SYSTEM  1 = DSGE setting :smc_reuse_system_new_params (merged on DSGE main): the old-data
#                 likelihood reuses the new-data system of the same proposal (RTWALK104-REVIEW §7); 0 = off (original).
#                 Default: 1 iff the old sample needs the full solve (k(t) = 0, steps 5-25), else 0 (steps 1-4)
#   PHI_TARGET    adaptive tempering target (default 0.95, = the seed run; SMOKE: TARGET_SMOKE wins)
#   N_PART        particles (default = the seed's size, 15,000); if it differs from the seed's, the seed is
#                 systematically resampled to N_PART equal-weight particles (as SMOKE does)
#   WORKER_HRT    h_rt of the FRBNY worker job (default 38:00:00; frbnyjuliamemory = "8G,h_rt=...")
#   SCRATCH       SMOKE output directory under work/rtwalk104 (default scratch)
#   H_PASSES, H_NTUNE, H_NCHUNK, H_POLISH   health-check MH passes (10), tuning passes (2), chunks (5), polishing (0)
#   H_<GATE>      gate overrides (health.jl GATE)
#   HEALTH_REDO   1 = re-run the health check even if a marker exists
#   WALK_BLOCKS   MH block groups (RTWALK104-STEP1FIX, setup.jl): covid (default) = [ρ_σ_w], [the CAND38 slots free at
#                 t+1], rest; two = [ρ_σ_w], rest (the pre-fix groups). c_blocks are carried between steps BY LABEL.
#   C_NEW_INIT    starting scale of a block group the seed lacks (default: the seed's rest scale)
#   H_C_NEW       same, for the health check of a bridge file written with other groups (HEALTH_REDO; default: the
#                 bridge's rest scale; the H_NTUNE tuning passes adapt it)
#   DRY_RUN       1 = all checks, no SMC
#   SMOKE         1 = NSMOKE particles resampled from the seed, adaptive φ target TARGET_SMOKE (or fixed NPHI_SMOKE
#                 stages), scratch saveroot/steps, every stage saved, resume from RESUME_SMOKE must be bitwise
# =============================================================================
const T0 = time()
const ORIGIN = ENV["ORIGIN"]; const SEED_CLOUD = ENV["SEED_CLOUD"]
const SMOKE = get(ENV, "SMOKE", "0") == "1"
const W0 = "/data/dsge_data_dir/estimation-AI/work/rtwalk104"
# ---- gate on the seed BEFORE any setup: a walk cloud needs its step's HEALTH_PASS
if occursin("/rtwalk104/", SEED_CLOUD) && !SMOKE
    isfile(joinpath(dirname(SEED_CLOUD), "HEALTH_PASS")) ||
        error("GATE: seed $(SEED_CLOUD) has no HEALTH_PASS in its step directory; the previous step failed or did not finish")
end
isfile(SEED_CLOUD) || error("seed cloud $(SEED_CLOUD) not found")

using Distributed
# workers: `julia -p NW` (one node, run.sh) or FRBNY_NW > 0 = addprocs_frbny (multi-node, as work/rtvint104 bridge.jl;
# the SGE job then needs only 1 slot; the workers come as a separate `-pe julia` job, 16G each by the site default)
const FRBNY_NW = parse(Int, get(ENV, "FRBNY_NW", "0"))
if FRBNY_NW > 0
    using ClusterManagers, frbny
    ENV["frbnyjuliamemory"] = "8G,h_rt=" * get(ENV, "WORKER_HRT", "38:00:00")   # every job gets an h_rt (RTWALK104-REVIEW §6 b)
    envdir = dirname(Base.active_project())
    mkpath(joinpath(W0, "workers")); cd(joinpath(W0, "workers"))
    addprocs_frbny(FRBNY_NW; exeflags = `--project=$envdir --startup-file=no --compiled-modules=existing`)
    cd(W0)
    @assert nworkers() == FRBNY_NW
end
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk104/scripts/setup.jl")
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk104/scripts/rt_data.jl")
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk104/scripts/full_data.jl")   # FULLSTEP
const T_OLD = qdate(ORIGIN); const T_NEW = nextq(T_OLD)
const L_OLD = qlabel(T_OLD); const L_NEW = qlabel(T_NEW)
@assert T_OLD == WALK_LAST && T_NEW == FULL_T "FULLSTEP: ORIGIN must be $(qlabel(WALK_LAST)) (got $ORIGIN)"   # FULLSTEP
const FIX_OLD = fixed_at(L_OLD); const FIX_NEW = Tuple{Symbol, Int}[]      # FULLSTEP: full sample = all 101 free (= ss104v3)
@assert Set(FIX_OLD) == Set([(:σ_condgdp, 2), (:σ_condcorepce, 2)]) "FULLSTEP: unexpected fixed set at $L_OLD: $FIX_OLD"
@assert Set(FIX_NEW) ⊆ Set(FIX_OLD) "a parameter-regime free at $L_OLD is fixed at $L_NEW: not supported"
const THETA_TILDE = sort(setdiff(FIX_OLD, FIX_NEW))
@everywhere configure_freeset!(m, Set($FIX_NEW))
# MH block groups for this step's free set (RTWALK104-STEP1FIX; setup.jl walk_block_groups): [ρ_σ_w], ["covid" = the
# CAND38 slots free at t+1], rest. Same value on every process; health.jl uses the same groups.
@everywhere m <= Setting(:smc_param_blocks, walk_param_blocks(m))
@everywhere include("/home/rcemzp01/.julia/dev/DSGE-fix_1020_7_8_9_10/walks/rtwalk104/scripts/health.jl")

const TAG = get(ENV, "TAG", "rtwalk104")
CONTINUE_FROM = parse(Int, get(ENV, "CONTINUE_FROM", "-1"))
INCREMENT = parse(Int, get(ENV, "INCREMENT", "1"))
const KEEP_STAGES = 3
const DRY_RUN = get(ENV, "DRY_RUN", "0") == "1"
const STEPNO = length(DSGE.quarter_range(WALK_FIRST, T_NEW)) - 1
const SCR = get(ENV, "SCRATCH", "scratch")
const STEPDIR = joinpath(W0, SMOKE ? SCR : "", "steps", "$(L_OLD)_to_$(L_NEW)")
const SUMMARY = joinpath(W0, SMOKE ? SCR : "", "rtwalk104_full_summary.csv")   # FULLSTEP: a new file
# system reuse for the old-data likelihood (DSGE patch, RTWALK104-REVIEW §7); same value on every process
# Default: ON iff the old data need the full solve (in-sample regime count k(t) = 0, i.e. t >= 2020Q4: steps 5-25).
# Measured (tests/test_reuse.jl, job 343098; bitwise in every case): 2022Q4->2023Q1 1.68 -> 0.61 s per proposal;
# 2019Q4->2020Q1 0.085 -> 0.081 s; but 2020Q3->2020Q4 (step 4: old data in-sample k = 4, new data full solve)
# 0.61 -> 1.15 s, because the old data then run on the reused full 29-regime system instead of their 4-regime one.
# [RTWALK104-N123 (FIXVERIFY N3): that step-4 slowdown does not reproduce. Paired, same θ, 40 draws x 2 reps, alternating
# order (jobs 343112/343113): 0.62 s both ways, reuse ~1.5% faster (old-data part 0.042 -> 0.035 s), bitwise 160/160.
# The default stays off at step 4; chain_qsub.txt passes REUSE_SYSTEM=1 on the step-4 line.]
const K_OLD = DSGE.insample_regime_count(m, T_OLD; tvis = haskey(DSGE.get_settings(m), :tvis_information_set), compute_pseudo = false)
const REUSE_SYSTEM = get(ENV, "REUSE_SYSTEM", K_OLD == 0 ? "1" : "0") == "1"
@everywhere m <= Setting(:smc_reuse_system_new_params, $REUSE_SYSTEM)
const PHI_TARGET = parse(Float64, get(ENV, "PHI_TARGET", "0.95"))
@everywhere m <= Setting(:adaptive_tempering_target_smc, $PHI_TARGET, true, "phi", "")
println("="^90, "\nRTWALK104 step $STEPNO: $L_OLD -> $L_NEW   host ", gethostname(), "  nworkers ", nworkers(), "  SMOKE ", SMOKE, "  ", now())
println("seed ", SEED_CLOUD, "\nfree parameter-regimes: ", nfree(), " (fixed: ", length(FIX_NEW), ");  new at this step (θ̃, from the prior): ",
        isempty(THETA_TILDE) ? "none" : join(["$(k)[$r]" for (k, r) in THETA_TILDE], " "))
mkpath(STEPDIR)
if get(ENV, "HEALTH_REDO", "0") != "1" && !SMOKE
    for f in ("HEALTH_PASS", "HEALTH_FAIL")
        isfile(joinpath(STEPDIR, f)) && (println("step already checked: ", joinpath(STEPDIR, f), "\n", read(joinpath(STEPDIR, f), String));
                                         exit(f == "HEALTH_PASS" ? 0 : 2))
    end
end

# ---- data
new_dat, V_NEW = full_data_pinned(m)          # FULLSTEP: the ss104v3 frozen full-sample matrix (vintage 261013)
old_dat, V_OLD = rt_data_pinned(m, T_OLD)
@assert size(new_dat, 2) == size(old_dat, 2) + 1
@everywhere HEALTH_DATA[] = full_data_pinned(m)[1]   # FULLSTEP
println("old data: sample end $L_OLD, vintage $V_OLD, T = $(size(old_dat, 2));  new data: $L_NEW, vintage $V_NEW, T = $(size(new_dat, 2))")

# ---- settings (SMC settings = the seed run's, setup.jl), bridge tags
m <= Setting(:fcast_date, L_NEW, true, "sampleend", "")
m <= Setting(:use_parallel_workers, nworkers() > 0 && nprocs() > 1)
m <= Setting(:saveroot, joinpath(W0, SMOKE ? SCR : "saveroot") * "/")
m <= Setting(:previous_data_vintage, V_OLD)
m <= Setting(:bridge, true, true, "bridge", "")
m <= Setting(:tempered_update_prior_weight, 0.0, true, "priorwt", "")
m <= Setting(:data_vintage, V_NEW)
if SMOKE && haskey(ENV, "TARGET_SMOKE")      # before rawpath: the φ target is in the file name
    m <= Setting(:adaptive_tempering_target_smc, parse(Float64, ENV["TARGET_SMOKE"]), true, "phi", ""); INCREMENT = 1
elseif SMOKE
    m <= Setting(:use_fixed_schedule, true); m <= Setting(:n_Φ, parse(Int, get(ENV, "NPHI_SMOKE", "4"))); m <= Setting(:λ, 1.0); INCREMENT = 1
end

# ---- the seed cloud: fresh copy every time (DSGE.estimate mutates old_cloud), θ̃ redrawn deterministically
NSMOKE = parse(Int, get(ENV, "NSMOKE", "300"))
function seed_cloud()
    c = load(SEED_CLOUD, "cloud")
    P = c.particles; N = size(P, 2)
    @assert N - 5 == length(VALS0) "seed has $(N-5) slots, model $(length(VALS0))"
    @assert c.tempering_schedule[end] >= 1.0 "seed cloud is at φ = $(c.tempering_schedule[end])"
    for (k, r) in FIX_OLD                      # every slot fixed at t sits at its fixed (default) value in the seed
        s = slot_index(k, r); @assert all(==(VALS0[s]), P[:, s]) "seed column $k[$r] is not at its fixed value"
    end
    NT = SMOKE ? NSMOKE : parse(Int, get(ENV, "N_PART", string(size(P, 1))))   # default: the seed's own size, no resampling
    if SMOKE || NT != size(P, 1)
        w = SMC.get_weights(c); cw = cumsum(w ./ sum(w)); u = (rand(MersenneTwister(20261104)) .+ (0:NT-1)) ./ NT
        c.particles = c.particles[[min(searchsortedfirst(cw, x), length(cw)) for x in u], :]
        SMC.reset_weights!(c)
    end
    w = SMC.get_weights(c); c.ESS = [sum(w)^2 / sum(abs2, w)]     # bug B2 workaround (initialization.jl:206)
    if !isempty(THETA_TILDE)
        sl = [slot_index(k, r) for (k, r) in THETA_TILDE]
        for i in 1:size(c.particles, 1)
            Random.seed!(SMC.stream_seed(UInt64(20261107 + STEPNO), :added_parameter_particle, 0, i))
            d = ModelConstructors.rand_regime_switching(m.parameters)
            c.particles[i, sl] = d[sl]
        end
        ModelConstructors.toggle_regime!(m.parameters, 1)
    end
    return c
end
sc = seed_cloud()
m <= Setting(:n_particles, length(sc), true, "nparts", "")
NP = length(sc)

# ---- old_cloud must reproduce on old_data, after the θ̃ redraw (BW-4, R1: modal, particle 1, 10 random)
function check_loglh(cloud, dat, col::Symbol, i::Int, what; rtol = 1e-9, atol = 0.0)
    P = cloud.particles; n = size(P, 2)
    stored = P[i, col === :loglh ? n - 4 : n - 2]
    rec = LIK(m, P[i, 1:n-5], dat)
    println("  $what ($col, particle #$i): stored $(repr(stored)), recomputed $(repr(rec)), diff $(rec - stored)")
    (isfinite(rec) && abs(rec - stored) <= max(atol, rtol * max(1.0, abs(stored)))) ||
        error("DATA / CLOUD MISMATCH ($what, $col, particle $i): stored $stored, recomputed $rec")
end
let P = sc.particles, n = size(P, 2)
    imod = argmax(P[:, n-4] .+ P[:, n-3])
    for i in unique(vcat(imod, 1, rand(MersenneTwister(7), 1:NP, 10)))
        check_loglh(sc, old_dat, :loglh, i, "old_cloud on old data"; atol = THETA_ATOL)   # setup.jl: max(1e-5, ENTER_TOL); default pin: ρ_ait_rm (θ̃ at step 2) moves log p(Y_2020Q1) by ≤ 6e-9
    end
    println("  new-data loglh at the modal particle: ", LIK(m, P[imod, 1:n-5], new_dat))
    nInf = count(i -> !isfinite(LIK(m, P[i, 1:n-5], new_dat)), rand(MersenneTwister(8), 1:NP, min(NP, 20)))
    println("  new-data loglh non-finite at $nInf of $(min(NP, 20)) random seed particles")
end

# ---- output path and stage-level resume (= rtvint104 bridge.jl)
savepath = rawpath(m, "estimate", "smc_cloud.jld2", [TAG])
RAW_DIR  = dirname(savepath)
step_prefix = replace(basename(savepath), ".jld2" => "")
stage_files() = sort([parse(Int, f[length(step_prefix)+8:end-5]) for f in readdir(RAW_DIR)
                      if startswith(f, step_prefix * "_stage=") && endswith(f, ".jld2")])
println("output: ", savepath, "\nstep dir: ", STEPDIR)
SMOKE && get(ENV, "SMOKE_KEEP_RAW", "0") != "1" && isdir(RAW_DIR) && rm(RAW_DIR; recursive = true)   # SMOKE_KEEP_RAW=1: tests of the resume choice (with DRY_RUN)
mkpath(RAW_DIR)
stage_path(k) = rawpath(m, "estimate", "smc_cloud", [TAG]) * "_stage=$(k).jld2"
# resume from the NEWEST stage file that loads (RTWALK104-REVIEW §6 c): a file cut short by a kill fails to load and is
# skipped; with INCREMENT = 1 at most one stage is lost per interruption
stage_loads(k) = try f = stage_path(k); load(f, "cloud"); load(f, "W"); load(f, "j"); true catch; false end
function prune_stages(keep::Int)           # keep the newest `keep` stage files that load; delete the rest without loading them
    st = stage_files(); kept = Int[]        # (never the final file; RTWALK104-FIXVERIFY N2: loads at most keep + #unloadable)
    for s in reverse(st)
        length(kept) == keep && break
        stage_loads(s) && push!(kept, s)
    end
    for s in st; s in kept || (rm(stage_path(s)); println("  deleted stage file $s")); end
end
function newest_loadable_stage()           # a function, not a top-level loop (soft scope)
    for s in reverse(stage_files())
        stage_loads(s) && return s
        println("  stage file $s does not load: skipped")
    end
    return 0
end
resume_stage = CONTINUE_FROM >= 0 ? CONTINUE_FROM : newest_loadable_stage()
!SMOKE && !isfile(savepath) && prune_stages(KEEP_STAGES)
println("stage files present: ", stage_files(), ";  resume stage: ", resume_stage, CONTINUE_FROM >= 0 ? " (CONTINUE_FROM)" : " (newest loadable)")
function resume_guard(k)
    rc = load(rawpath(m, "estimate", "smc_cloud", [TAG]) * "_stage=$(k).jld2", "cloud")
    check_loglh(rc, new_dat, :loglh, 1, "resume stage $k"; rtol = 1e-12)
    check_loglh(rc, old_dat, :old_loglh, 1, "resume stage $k"; rtol = 1e-12)
end

# ---- BW-6 + R3: start every block group at the seed's own final scale. c_blocks = [ρ_σ_w, rest] comes from the seed
# file (key "c_blocks": written by SMC into every final cloud file and by health.jl into cloud_health.jld2), or from
# env C_BLOCKS="a,b". DSGE forwards the :smc_initial_c_blocks setting to SMC.smc as
# the initial_c_blocks keyword. On a resume the checkpoint's scales are used instead.
# this run's block groups (RTWALK104-STEP1FIX: named groups, setup.jl walk_block_labels; SMC orders them the same way,
# fixed groups then the rest); health.jl uses the same groups
const RUN_BLOCK_LABELS = walk_block_labels(m)
@assert length(RUN_BLOCK_LABELS) == length(get_setting(m, :smc_param_blocks)) + 1
@assert RUN_BLOCK_LABELS == health_block_labels() "bridge block groups $RUN_BLOCK_LABELS differ from the health check's $(health_block_labels())"
# C_NEW_INIT: starting scale of a group the seed does not have (default: the seed's rest scale)
const C_NEW_INIT = haskey(ENV, "C_NEW_INIT") ? parse(Float64, ENV["C_NEW_INIT"]) : nothing
const SEED_C_BLOCKS = if haskey(ENV, "C_BLOCKS")       # positional, in the order of RUN_BLOCK_LABELS ("a:b:c" or "a,b,c")
    parse.(Float64, split(ENV["C_BLOCKS"], r"[,:]"))
else
    # "c_blocks": SMC final files and cloud_health.jld2; "cs": work/rtvint104-health/cloud_mh.jld2 (frozen MH scales)
    f = jldopen(SEED_CLOUD, "r")
    cb = haskey(f, "c_blocks") ? Vector{Float64}(f["c_blocks"]) : haskey(f, "cs") ? Vector{Float64}(f["cs"]) : nothing
    # carry c_blocks BY BLOCK LABEL (RTWALK104-REVIEW §3): "c_block_labels" (cloud_health.jld2, health.jl), else SMC's
    # "param_blocks" (final/stage files: Vector{Vector{Symbol}}, the fixed groups then the rest group; labelled as SMC's
    # block_labels: a one-parameter group by its name, the last group "rest"), else the rtvint104-health files ("cs":
    # [ρ_σ_w, rest], work/rtvint104-health/mh.jl:144)
    labs = haskey(f, "c_block_labels") ? Vector{String}(f["c_block_labels"]) :
           haskey(f, "param_blocks")   ? label_param_blocks(f["param_blocks"], m) :
           ["ρ_σ_w", "rest"]
    close(f)
    cb === nothing && @warn "seed file has no c_blocks: every block group starts at the seed's c = $(sc.c) (R3 not handled)"
    # RTWALK104-STEP1FIX: carried BY LABEL, so groups may appear (e.g. "covid" at step 1: starts at C_NEW_INIT, default
    # the seed's rest scale) or be merged into the rest (scale dropped); the scales of common labels are carried
    if cb === nothing
        fill(sc.c > 0 ? sc.c : 0.5, length(RUN_BLOCK_LABELS))
    else
        cc, notes = carry_c_blocks(labs, cb, RUN_BLOCK_LABELS; cnew = C_NEW_INIT)
        println("seed c_blocks ", labs, " = ", cb)
        foreach(n -> println("  c_blocks carry-over: ", n), notes)
        cc
    end
end
@assert length(SEED_C_BLOCKS) == length(RUN_BLOCK_LABELS) && all(>(0), SEED_C_BLOCKS)
c0 = SEED_C_BLOCKS[end]
m <= Setting(:step_size_smc, c0)
m <= Setting(:smc_initial_c_blocks, SEED_C_BLOCKS)
println("initial c_blocks ", RUN_BLOCK_LABELS, " = ", SEED_C_BLOCKS, " (from the seed);  step_size_smc = ", c0)
println("settings: N = ", get_setting(m, :n_particles), ", tempering target ", get_setting(m, :adaptive_tempering_target_smc), ", REUSE_SYSTEM ", REUSE_SYSTEM, " (k_old = ", K_OLD, "), INCREMENT ", INCREMENT, ", free-set pin ", basename(FREESET_PIN), " (ENTER_TOL ", ENTER_TOL, ")")
flush(stdout)
DRY_RUN && (println("DRY_RUN: checks passed, no SMC  (", round((time() - T0) / 60, digits = 2), " min)"); exit())

est(k) = DSGE.estimate(m, new_dat;
                       old_cloud = seed_cloud(), old_data = old_dat,
                       old_model = m,                  # same object (see header)
                       verbose = :low, filestring_addl = [TAG],
                       save_intermediate = true, intermediate_stage_increment = INCREMENT,
                       run_csminwel = false,           # no log_prob_old_data (B1); cholesky_fix = :none
                       continue_intermediate = k > 0, intermediate_stage_start = k > 0 ? k : 1)

tb = time()
if isfile(savepath) && (!SMOKE || get(ENV, "SMOKE_REUSE_BRIDGE", "0") == "1")   # SMOKE_REUSE_BRIDGE (+ SMOKE_KEEP_RAW): tests only
    println("bridge already complete: $savepath")
else
    resume_stage > 0 && (println("RESUMING from stage $resume_stage"); resume_guard(resume_stage))
    est(SMOKE ? 0 : resume_stage)
    SMOKE || prune_stages(KEEP_STAGES)
end
bridge_wall = time() - tb
# SMOKE only: keep a copy of the final bridge file (reuse on/off comparison, tests/test_reuse.jl)
SMOKE && haskey(ENV, "SMOKE_SAVE") && (cp(savepath, ENV["SMOKE_SAVE"]; force = true); println("bridge file copied to ", ENV["SMOKE_SAVE"]))
println(@sprintf("bridge wall %.1f s (REUSE_SYSTEM %s)", bridge_wall, REUSE_SYSTEM))
full = load(savepath)
println(@sprintf("\nBRIDGE %s -> %s done: %d stages, %.2f h this job  %s", L_OLD, L_NEW, length(full["stage_health"]["stage"]), bridge_wall / 3600, string(now())))
println("log MDD estimates (smc = log p(Y_new)/p(Y_old) increment): ", get(full, "log_mdd_estimates", "n/a"))
let fc = full["cloud"], n = size(fc.particles, 2)
    check_loglh(fc, new_dat, :loglh, 1, "final cloud"; rtol = 1e-12)
    check_loglh(fc, old_dat, :old_loglh, 1, "final cloud"; rtol = 1e-12)
end
if SMOKE && haskey(ENV, "RESUME_SMOKE")
    k = parse(Int, ENV["RESUME_SMOKE"]); println("\nSMOKE resume from stage $k:"); resume_guard(k)
    est(k); res = load(savepath)
    strip_time(d) = Dict(kk => v for (kk, v) in d if kk != "time")
    same = isequal(res["cloud"].particles, full["cloud"].particles) && isequal(res["w"], full["w"]) && isequal(res["W"], full["W"]) &&
           isequal(strip_time(res["stage_health"]), strip_time(full["stage_health"])) && isequal(res["c_blocks"], full["c_blocks"])
    println("resume from stage $k = uninterrupted run, bitwise: ", same); @assert same
end

# ---- health check + gate
fc = full["cloud"]
sh = full["stage_health"]
# the bridge's final scales, carried BY LABEL to the health check's groups (RTWALK104-STEP1FIX): with HEALTH_REDO on a
# bridge run with other groups (step 1: [ρ_σ_w, rest] bridge, [ρ_σ_w, covid, rest] health), a group the bridge lacks
# starts at H_C_NEW (default: the bridge's rest scale) and is tuned in the H_NTUNE passes
const BRIDGE_LABELS = haskey(full, "param_blocks") ? label_param_blocks(full["param_blocks"], m) : ["ρ_σ_w", "rest"]   # groups of the bridge run
cs0 = if haskey(sh, "c_blocks")
    cc, notes = carry_c_blocks(BRIDGE_LABELS, Vector{Float64}(sh["c_blocks"][:, end]), health_block_labels();
                               cnew = haskey(ENV, "H_C_NEW") ? parse(Float64, ENV["H_C_NEW"]) : nothing)
    println("bridge c_blocks ", BRIDGE_LABELS, " = ", sh["c_blocks"][:, end], " -> health ", health_block_labels(), " = ", cc)
    foreach(n -> println("  c_blocks carry-over: ", n), notes)
    cc
else
    fill(fc.c, length(health_block_labels()))
end
info = Dict(:step => STEPNO, :old => L_OLD, :new => L_NEW, :vint_old => V_OLD, :vint_new => V_NEW, :seed => SEED_CLOUD,
            :cloudfile => savepath, :summary_csv => SUMMARY, :bridge_wall_h => bridge_wall / 3600,
            :log_mdd_incr => (lm = get(full, "log_mdd_estimates", Dict()); lm isa AbstractDict ? get(lm, "smc", NaN) : NaN))
ok, row = run_health(; cloud = fc, seed = seed_cloud(), stage_health = sh, thetatilde = THETA_TILDE, stepdir = STEPDIR, info = info, cs0 = cs0, bridge_labels = BRIDGE_LABELS,
                     npass = parse(Int, get(ENV, "H_PASSES", "10")), ntune = parse(Int, get(ENV, "H_NTUNE", "2")),
                     nchunk = parse(Int, get(ENV, "H_NCHUNK", "5")), polish = parse(Int, get(ENV, "H_POLISH", "0")))
println(@sprintf("step %d %s -> %s: %s  (job total %.2f h on %d processes)", STEPNO, L_OLD, L_NEW, ok ? "PASS" : "FAIL", (time() - T0) / 3600, nprocs()))
exit(ok ? 0 : 2)
