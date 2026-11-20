# RTWALK114 shared setup (master and workers) = work/rtwalk104/scripts/setup.jl with ss104 -> ss114.
# Differences from rtwalk104 (RTWALK114.md §2):
#   - code: DSGE main 27c8b6162+ and SMC main e5b1ad4+; old pins remain in the original walk directory.
#   - model: build_walk_model(2026Q3, "ss114"; avg = false, antmu = 0) = rtwalk104's build with the ss114 subspec
#     (no pgap/ygap states or observables: 26 observables; 93 free; the 8 AIT slots fixed at the ss104 defaults).
#   - CANDW = rtwalk104's CAND38 minus the 8 AIT slots (ρ_ait_rm, σ_ait_rm[2], σ_ait_r_m1..6[2]), which ss114 fixes for
#     good (30 candidates). configure_freeset! frees / fixes only these; the ρ_ait_rm rebuild branch is gone.
#   - free-set pin: pinned/freeset.jld2 = rtwalk104's pinned/freeset.jld2 (ENTER_TOL 1e-6) with the AIT slots removed
#     from every FIXED_AT[t] (scripts/pin_freeset.jl; checked against an ss114 perturbation test at every sample end by
#     tests/test_walk114.jl).
# Everything else (SMC settings, block groups, c_blocks carry-over, LIK) is unchanged.
#
# Defines: m, CANDW (alias CAND38 for health.jl / step.jl), configure_freeset!, slot_index, nfree, LIK, check_versions, ...
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

# ------------------------------------------------------------------ model
const SUBSPEC = "ss114"
m = build_walk_model(Date(2026, 9, 30), SUBSPEC; avg = false, antmu = 0)
m <= Setting(:fcast_date, "2020Q3", true, "sampleend", "")   # file tag; reset per step by step.jl
@assert !haskey(m.observables, :obs_avgshortinflation)
@assert !haskey(m.observables, :obs_pgap) && !haskey(m.observables, :obs_ygap) "ss114 must not have the gap observables"
@assert n_observables(m) == 26
@assert !any(p -> p.key in (:σ_λ_f1, :σ_λ_f2), m.parameters)
m <= Setting(:likelihood_insample_regimes_only, true)
m <= Setting(:likelihood_truncated_regimes, get(ENV, "TRUNC_REGIMES", "1") == "1")   # RTWALK104-TRUNC: bitwise-identical, skips out-of-sample measurement eqs

regfixed(p, r) = haskey(p.regimes, :fixed) && haskey(p.regimes[:fixed], r) ? p.regimes[:fixed][r] : p.fixed
nfree(mm = m) = (ModelConstructors.toggle_regime!(mm.parameters, 1);
           length(ModelConstructors.get_free_para_inds(mm.parameters; regime_switching = true)))
ModelConstructors.toggle_regime!(m.parameters, 1)
const NFREE_BASE = 93
@assert nfree() == NFREE_BASE "base build has $(nfree()) free parameter-regimes, expected $NFREE_BASE"
const VALS0 = copy(ModelConstructors.get_values(m.parameters))
@assert length(VALS0) == 184

# the 8 AIT slots ss114 fixes for good (work/ss114/scripts/common.jl AIT_SLOTS), and the walk's candidates
const AIT_SLOTS = sort(collect(Set([(:ρ_ait_rm, 1), (:σ_ait_rm, 2), [(Symbol("σ_ait_r_m$i"), 2) for i in 1:6]...])))
const CAND38_104 = sort(collect(Set([(:ρ_meas_π, 2), (:ρ_ait_rm, 1),
    [(k, 2) for k in (:σ_g, :σ_b, :σ_μ, :σ_ztil, :σ_λ_f, :σ_λ_w, :σ_σ_ω, :σ_π_star, :σ_lr, :σ_tfp, :σ_gdpdef, :σ_gdp,
                      :σ_gdi, :σ_meas_π, :σ_ait_rm, :σ_condgdp, :σ_condcorepce, :σ_biidc1)]...,
    [(k, r) for k in (:σ_ziid, :σ_biidc, :σ_φ) for r in (2, 3)]...,
    [(Symbol("σ_ait_r_m$i"), 2) for i in 1:6]..., [(Symbol("σ_exp_rm$i"), 2) for i in 1:6]...])))
const CANDW = sort(setdiff(CAND38_104, AIT_SLOTS))
@assert length(CAND38_104) == 38 && length(CANDW) == 30
const CAND38 = CANDW      # the name step.jl / health.jl / the tests use for "the walk's candidate slots"

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
slot_index_sym(s::Symbol) = (k = findfirst(==(s), SLOTSYMS); k === nothing && error("no slot $s"); k)
@assert all(!isnothing, [slot_index(k, r) for (k, r) in CAND38_104])
# slots fixed in the base build (before configure_freeset!): never free in the walk; their particle columns must sit at VALS0
const BASE_FIXED_SLOTS = let free = Set(ModelConstructors.get_free_para_inds(m.parameters; regime_switching = true))
    ModelConstructors.toggle_regime!(m.parameters, 1); [s for s in eachindex(VALS0) if !(s in free)]
end
@assert length(BASE_FIXED_SLOTS) == 184 - NFREE_BASE
@assert all(s -> s in BASE_FIXED_SLOTS, [slot_index(k, r) for (k, r) in AIT_SLOTS]) "an AIT slot is free in the ss114 build"
@assert !any(s -> s in BASE_FIXED_SLOTS, [slot_index(k, r) for (k, r) in CANDW]) "a walk candidate is fixed in the ss114 build"

"""Fix exactly the members of `fixed` (⊆ CANDW) at their default values; free the other members of CANDW.
Call once, on a freshly built model."""
function configure_freeset!(m, fixed)
    fixed = Set(fixed)
    @assert fixed ⊆ Set(CANDW) "fixed set has members outside CANDW: $(setdiff(fixed, Set(CANDW)))"
    ModelConstructors.toggle_regime!(m.parameters, 1)
    for key in unique(first.(CANDW))
        p = m[key]
        @assert !isempty(p.regimes) "$key: walk candidates are regime-switching parameters in ss114"
        nr = length(p.regimes[:value])
        p.regimes[:fixed] = OrderedDict{Int, Bool}(r => (regfixed(p, r) || (key, r) in fixed) for r in 1:nr)
    end
    ModelConstructors.toggle_regime!(m.parameters, 1)
    @assert nfree(m) == NFREE_BASE - length(fixed)
    @assert ModelConstructors.get_values(m.parameters) == VALS0
    return m
end

# ------------------------------------------------------------------ SMC settings (= rtwalk104 = its seed run, incl. the blocks)
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
    m <= Setting(:smc_param_blocks, [[:ρ_σ_w]])     # step.jl replaces it by walk_param_blocks()
    m <= Setting(:smc_blockwise_c, true)
    @assert get_setting(m, :n_smc_blocks) == 1
end
set_smc_settings!(m)

# ------------------------------------------------------------------ MH block groups (= rtwalk104 RTWALK104-STEP1FIX)
#   "ρ_σ_w"  [ρ_σ_w];  "covid"  every member of CANDW free at this step;  "rest"  everything else
const WALK_BLOCKS = get(ENV, "WALK_BLOCKS", "covid")
@assert WALK_BLOCKS in ("covid", "two") "WALK_BLOCKS = $WALK_BLOCKS: use covid or two"
function walk_block_groups(mm = m)
    ModelConstructors.toggle_regime!(mm.parameters, 1)
    free = Set(ModelConstructors.get_free_para_inds(mm.parameters; regime_switching = true))
    g = Tuple{String, Vector{Symbol}}[("ρ_σ_w", [:ρ_σ_w])]
    if WALK_BLOCKS == "covid"
        cov = [SLOTSYMS[slot_index(k, r)] for (k, r) in CANDW if slot_index(k, r) in free]
        isempty(cov) || push!(g, ("covid", cov))
    end
    return g
end
walk_param_blocks(mm = m) = [syms for (_, syms) in walk_block_groups(mm)]
walk_block_labels(mm = m) = vcat(first.(walk_block_groups(mm)), "rest")
function label_param_blocks(pb, mm = m)
    known = Dict(Set(s) => l for (l, s) in walk_block_groups(mm))
    return vcat([get(known, Set(Symbol.(b)), length(b) == 1 ? string(b[1]) : join(string.(b), "+")) for b in pb[1:end-1]], "rest")
end
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

# The likelihood exactly as DSGE.smc2's _my_likelihood_dsge evaluates it (= rtwalk104 setup.jl LIK)
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
# Free-set pin (pinned/freeset.jld2: "fixed_at" Dict label => Vector{(key, regime)}, "enter_tol"). ENTER_TOL and
# THETA_ATOL as rtwalk104.
const FREESET_PIN = get(ENV, "FREESET_PIN", joinpath(W, "pinned", "freeset.jld2"))
const ENTER_TOL = isfile(FREESET_PIN) ? jldopen(fp -> haskey(fp, "enter_tol") ? Float64(fp["enter_tol"]) : 1e-6, FREESET_PIN, "r") : 1e-6
const THETA_ATOL = max(1e-5, ENTER_TOL)
fixed_at(label::AbstractString) = (isfile(FREESET_PIN) || error("no $FREESET_PIN: run pin_freeset.jl first");
                                   load(FREESET_PIN, "fixed_at")[label])
flush(stdout)
