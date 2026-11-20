# RTWALK104 shared setup (master and workers): the ss104 model of the 2019Q4 seed run, the SMC settings, and the
# free/fixed parameter-regime set for a given sample end.
#
# Model: exactly the seed run's (work/rtvint104/scripts/setup.jl = work/est2019blk/scripts/est2019blk.jl): Elena's
# 2026Q3 build of ss104 without SPF short-run inflation expectations and without anticipated mark-up shocks
# (avg = false, antmu = 0), `:likelihood_insample_regimes_only = true`. The SAME build is used at every origin:
# the regime calendar of ss104 is fixed by the subspec (29 regimes, 2020Q1 = regime 2 ...), the likelihood only
# uses the in-sample regimes of the data it gets, and preflight.jl (PART A, "build at origin") checks that a model
# built with the origin's own forecast date gives the same likelihood.
#
# Free set: the seed run fixed 38 parameter-regimes that do not enter the 2019Q4 likelihood (CAND38 below). As the
# sample grows some of them start to enter. `FIXED_AT[label]` (pinned/freeset.jld2, written by preflight.jl PART A
# from a bitwise perturbation test) lists, for each sample end, the members of CAND38 that still do not enter the
# likelihood; `configure_freeset!(m, fixed)` fixes exactly those and frees the rest of CAND38.
#
# Defines: m, CAND38, configure_freeset!, slot_index, nfree, LIK, check_versions, set_smc_settings!
include(joinpath(@__DIR__, "common.jl"))
using SMC, JLD2

# ------------------------------------------------------------------ code versions
# The walk uses the merged mainline APIs. Reject the older pinned clones, which use
# SMC.INITIAL_C_BLOCKS[] and do not accept :smc_initial_c_blocks.
const MINIMUM_PACKAGE_COMMITS = ((DSGE, "27c8b6162"), (SMC, "e5b1ad4"))
function check_versions(; verbose = true)
    for (mod, minimum_commit) in MINIMUM_PACKAGE_COMMITS
        d = pkgdir(mod)
        @assert success(`git -C $d merge-base --is-ancestor $minimum_commit HEAD`) "$mod at $d does not contain required main commit $minimum_commit"
        dirty = readchomp(`git -C $d status --porcelain --untracked-files=no`)
        @assert isempty(dirty) "$mod checkout at $d has uncommitted tracked changes"
        verbose && println(rpad(string(mod), 20), d, "  HEAD ", readchomp(`git -C $d rev-parse --short HEAD`))
    end
end
check_versions(verbose = myid() == 1)

qlabel(d::Date) = string(Dates.year(d)) * "Q" * string(Dates.quarterofyear(d))
qdate(s::AbstractString) = (y = parse(Int, s[1:4]); q = parse(Int, s[6:6]); Dates.lastdayofquarter(Date(y, 3q, 1)))
nextq(d::Date) = Dates.lastdayofquarter(d + Dates.Day(1))
prevq(d::Date) = Dates.lastdayofquarter(Dates.firstdayofquarter(d) - Dates.Day(1))

# ------------------------------------------------------------------ model (= the seed run)
m = build_ss104(Date(2026, 9, 30); avg = false, antmu = 0)
m <= Setting(:fcast_date, "2019Q4", true, "sampleend", "")   # file tag; reset per step by step.jl
@assert !haskey(m.observables, :obs_avgshortinflation)
@assert !any(p -> p.key in (:σ_λ_f1, :σ_λ_f2), m.parameters)
m <= Setting(:likelihood_insample_regimes_only, true)
m <= Setting(:likelihood_truncated_regimes, get(ENV, "TRUNC_REGIMES", "1") == "1")   # RTWALK104-TRUNC: bitwise-identical, skips out-of-sample measurement eqs

regfixed(p, r) = haskey(p.regimes, :fixed) && haskey(p.regimes[:fixed], r) ? p.regimes[:fixed][r] : p.fixed
nfree(mm = m) = (ModelConstructors.toggle_regime!(mm.parameters, 1);
           length(ModelConstructors.get_free_para_inds(mm.parameters; regime_switching = true)))
ModelConstructors.toggle_regime!(m.parameters, 1)
@assert nfree() == 101 "base build has $(nfree()) free parameter-regimes, expected 101"
const VALS0 = copy(ModelConstructors.get_values(m.parameters))

# the 38 parameter-regimes the seed run fixed (work/rtvint104/scripts/setup.jl EXPECTED_TARGETS)
const CAND38 = sort(collect(Set([(:ρ_meas_π, 2), (:ρ_ait_rm, 1),
    [(k, 2) for k in (:σ_g, :σ_b, :σ_μ, :σ_ztil, :σ_λ_f, :σ_λ_w, :σ_σ_ω, :σ_π_star, :σ_lr, :σ_tfp, :σ_gdpdef, :σ_gdp,
                      :σ_gdi, :σ_meas_π, :σ_ait_rm, :σ_condgdp, :σ_condcorepce, :σ_biidc1)]...,
    [(k, r) for k in (:σ_ziid, :σ_biidc, :σ_φ) for r in (2, 3)]...,
    [(Symbol("σ_ait_r_m$i"), 2) for i in 1:6]..., [(Symbol("σ_exp_rm$i"), 2) for i in 1:6]...])))
@assert length(CAND38) == 38

# column of (key, regime) in the particle matrix / the regime-switching value vector
# (layout of ModelConstructors.get_values(...; regime_switching = true): all regime-1 values, then regimes 2.. of
#  each regime-switching parameter in parameter order)
const SLOTSYMS = let s = Symbol[θ.key for θ in m.parameters]
    for θ in m.parameters
        if !isempty(θ.regimes) && haskey(θ.regimes, :value)
            for i in 2:length(θ.regimes[:value]); push!(s, Symbol(θ.key, "_reg$i")); end
        end
    end
    s
end
@assert length(SLOTSYMS) == length(VALS0) == n_parameters_regime_switching(m)
slot_index(key::Symbol, r::Int) = findfirst(==(r == 1 ? key : Symbol(key, "_reg$r")), SLOTSYMS)
slot_index_sym(s::Symbol) = (k = findfirst(==(s), SLOTSYMS); k === nothing && error("no slot $s"); k)   # RTWALK104-STEP1FIX
@assert all(!isnothing, [slot_index(k, r) for (k, r) in CAND38])

"""Fix exactly the members of `fixed` (⊆ CAND38) at their default values; free the other members of CAND38.
Call once, on a freshly built model (the ρ_ait_rm rebuild is one-way)."""
function configure_freeset!(m, fixed)
    fixed = Set(fixed)
    @assert fixed ⊆ Set(CAND38) "fixed set has members outside CAND38: $(setdiff(fixed, Set(CAND38)))"
    ModelConstructors.toggle_regime!(m.parameters, 1)
    for key in unique(first.(CAND38))
        p = m[key]
        if key == :ρ_ait_rm
            @assert isempty(p.regimes)
            if (key, 1) in fixed
                m <= parameter(p.key, p.value, p.valuebounds, p.transform_parameterization, p.transform, get(p.prior);
                               fixed = true, description = p.description, tex_label = p.tex_label)
            end
        else
            nr = length(p.regimes[:value])
            p.regimes[:fixed] = OrderedDict{Int, Bool}(r => (regfixed(p, r) || (key, r) in fixed) for r in 1:nr)
        end
    end
    ModelConstructors.toggle_regime!(m.parameters, 1)
    @assert nfree(m) == 101 - length(fixed)
    @assert ModelConstructors.get_values(m.parameters) == VALS0
    return m
end

# ------------------------------------------------------------------ SMC settings (= the seed run, incl. the blocks)
function set_smc_settings!(m)
    m <= Setting(:sampling_method, :SMC)
    m <= Setting(:use_parallel_workers, nprocs() > 1)
    m <= Setting(:use_fixed_schedule, false)
    m <= Setting(:n_particles, 15000, true, "nparts", "")
    m <= Setting(:n_mh_steps_smc, 3, true, "mhsteps", "")
    m <= Setting(:step_size_smc, 0.5)
    m <= Setting(:mixture_proportion, 0.975)
    m <= Setting(:adaptive_tempering_target_smc, 0.95, true, "phi", "")
    m <= Setting(:resampling_threshold, 0.5)
    m <= Setting(:smc_seed, 20261031)
    m <= Setting(:smc_param_blocks, [[:ρ_σ_w]])     # the seed run's; step.jl replaces it by walk_param_blocks() (STEP1FIX)
    m <= Setting(:smc_blockwise_c, true)
    @assert get_setting(m, :n_smc_blocks) == 1
end
set_smc_settings!(m)

# ------------------------------------------------------------------ MH block groups of the walk (RTWALK104-STEP1FIX)
# Step 1 failed its gate because the 17 regime-2 COVID σ's freed at 2020Q1 sat in the same random-walk block as the 62
# well-identified parameters. Their regime-2 values govern 2 quarters (σ_biidc/σ_gdpdef/σ_ziid/σ_φ: 3; σ_meas_π: 9),
# so their posteriors stay prior-like and very heavy-tailed in levels (excess kurtosis up to 379; Mahalanobis var
# ratio 48 vs 2.6 for the old block; RTWALK104-STEP1FIX.md §1). A common scale for both forced c_rest to ≈ 0.1, and
# the whole block mixed ≈ 6× slower than in the 2019Q4 seed.
# Groups (in this order; SMC puts the rest group last):
#   "ρ_σ_w"  [ρ_σ_w]                                          (as before)
#   "covid"  every member of CAND38 that is FREE at this step  (the slots freed during the walk, all cohorts together;
#            kept as one group at every later step: they never become well identified, and one group per cohort would
#            add one likelihood evaluation per MH step for each cohort)
#   "rest"   everything else
# WALK_BLOCKS=two restores the seed run's two groups [ρ_σ_w, rest] (the old behaviour).
const WALK_BLOCKS = get(ENV, "WALK_BLOCKS", "covid")
@assert WALK_BLOCKS in ("covid", "two") "WALK_BLOCKS = $WALK_BLOCKS: use covid or two"
"""(label, parameter symbols) of the fixed block groups for the current free set of `mm` (rest group not included)."""
function walk_block_groups(mm = m)
    ModelConstructors.toggle_regime!(mm.parameters, 1)
    free = Set(ModelConstructors.get_free_para_inds(mm.parameters; regime_switching = true))
    g = Tuple{String, Vector{Symbol}}[("ρ_σ_w", [:ρ_σ_w])]
    if WALK_BLOCKS == "covid"
        cov = [SLOTSYMS[slot_index(k, r)] for (k, r) in CAND38 if slot_index(k, r) in free]
        isempty(cov) || push!(g, ("covid", cov))
    end
    return g
end
walk_param_blocks(mm = m) = [syms for (_, syms) in walk_block_groups(mm)]           # -> Setting(:smc_param_blocks)
walk_block_labels(mm = m) = vcat(first.(walk_block_groups(mm)), "rest")              # labels of c_blocks, in SMC's order
"""Label SMC's saved `param_blocks` (Vector{Vector{Symbol}}: the fixed groups, then the rest group) the way the walk
labels its groups: a group equal to one of this run's fixed groups gets that label; [ρ_σ_w] is "ρ_σ_w"; the last is
"rest"; anything else is labelled by its "+"-joined names (the pre-STEP1FIX convention)."""
function label_param_blocks(pb, mm = m)
    known = Dict(Set(s) => l for (l, s) in walk_block_groups(mm))
    return vcat([get(known, Set(Symbol.(b)), length(b) == 1 ? string(b[1]) : join(string.(b), "+")) for b in pb[1:end-1]], "rest")
end
"""Carry block scales BY LABEL from (src_labels, src_c) to `run_labels`. A label present in both keeps its scale; a run
label the source lacks (a group that appears at this step, e.g. "covid" at step 1) starts at `cnew` (default: the
source's rest scale); a source label the run lacks (a group merged into the rest) is dropped. "rest" must be in both."""
function carry_c_blocks(src_labels::AbstractVector, src_c::AbstractVector, run_labels::AbstractVector; cnew = nothing)
    @assert length(src_labels) == length(src_c) "c_blocks $src_c and labels $src_labels differ in length"
    @assert allunique(src_labels) && allunique(run_labels)
    @assert "rest" in src_labels && "rest" in run_labels "no rest group in $src_labels or $run_labels"
    d = Dict(String(l) => Float64(c) for (l, c) in zip(src_labels, src_c))
    cn = cnew === nothing ? d["rest"] : Float64(cnew)
    out = [get(d, String(l), cn) for l in run_labels]
    notes = vcat(["$l: new group, starts at $cn" for l in run_labels if !haskey(d, String(l))],
                 ["$l: not a group of this run (merged into rest), scale $(d[String(l)]) dropped" for l in keys(d) if !(l in String.(run_labels))])
    @assert all(x -> isfinite(x) && x > 0, out) "non-positive block scale in $out"
    return out, notes
end

# The likelihood exactly as DSGE.smc2's _my_likelihood_dsge evaluates it (= rtvint104 setup.jl LIK)
const REGSW    = haskey(DSGE.get_settings(m), :regime_switching) && get_setting(m, :regime_switching)
const USECHAND = get_setting(m, :use_chand_recursion)
function LIK(m, θ::AbstractVector, data::Matrix{Float64})::Float64
    ModelConstructors.update!(m.parameters, Vector{Float64}(θ))
    DSGE.update!(m, m.parameters, regime_switching = REGSW)
    m <= Setting(:preprocessed_transitions, Dict())
    L = likelihood(m, data; sampler = false, catch_errors = true, add_zlb_duration = (false, 1),
                   use_chand_recursion = USECHAND, verbose = :none)
    ModelConstructors.toggle_regime!(m.parameters, 1)
    return L
end
# Free-set pin. "Enters the likelihood" = |ΔL| > ENTER_TOL (or non-finite) in the perturbation test (preflight.jl A3).
#   pinned/freeset.jld2           ENTER_TOL = 1e-6 (the original pin; DEFAULT)
#   pinned/freeset_tol1e-4.jld2   ENTER_TOL = 1e-4 (RTWALK104-REVIEW §1 d; built by repin_freeset.jl). NOT the default:
#       it fixes ρ_ait_rm until 2021Q3 and σ_biidc[3] until 2020Q4, and at realistic particles redrawing those slots moves
#       log p(Y_t) by up to 9.4e-4 (step 8) and 1.4e-4 (step 5) (tests/test_reuse.jl, job 343098), i.e. they DO enter;
#       the 1e-6 pin errs the harmless way (frees slots that do not enter). Select it with env FREESET_PIN=<path>.
# ENTER_TOL is read from the pin ("enter_tol", absent = 1e-6). step.jl's old-data check after the θ̃ redraw uses
# atol THETA_ATOL = max(1e-5, ENTER_TOL) (1e-5 = the original atol).
const FREESET_PIN = get(ENV, "FREESET_PIN", joinpath(W, "pinned", "freeset.jld2"))
const ENTER_TOL = isfile(FREESET_PIN) ? jldopen(fp -> haskey(fp, "enter_tol") ? Float64(fp["enter_tol"]) : 1e-6, FREESET_PIN, "r") : 1e-6
const THETA_ATOL = max(1e-5, ENTER_TOL)
fixed_at(label::AbstractString) = (isfile(FREESET_PIN) || error("no $FREESET_PIN: run preflight.jl PART=A first");
                                   load(FREESET_PIN, "fixed_at")[label])
flush(stdout)
