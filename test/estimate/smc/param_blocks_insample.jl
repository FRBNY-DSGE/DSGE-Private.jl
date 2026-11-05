# A module, so that globals of other test files run in the same session cannot shadow names
# used here, and this file's cannot leak out
module TestParamBlocksInsample

using DSGE, ModelConstructors, SMC, Test, Dates, Random, LinearAlgebra, JLD2, FileIO, OrderedCollections, Distributed

# SMC through DSGE on m1002 ss104 with fixed mutation blocks AND in-sample-regime likelihoods: the
# setup of the blocked 2019Q4 estimation. Data end in 2019Q4, :likelihood_insample_regimes_only is
# on (only regime 1 is solved), the parameter regimes that do not enter the 2019Q4 likelihood are
# fixed, and :smc_param_blocks => [[:ρ_σ_w]] with :smc_blockwise_c => true.
# param_blocks.jl runs blocked SMC on AnSchorfheide (on ss104 only with a zero likelihood), and
# likelihood_insample_regimes.jl never runs SMC. This file runs DSGE.estimate (smc2 -> SMC.smc)
# with the real likelihood and checks that
#   1. every final particle's stored loglh is the in-sample likelihood, bitwise, and that the run
#      really used it: some final particles are draws the full solve rejects (see the witness);
#   2. every fixed slot is the same in all particles and equals its fixed value;
#   3. the mutation blocks are {ρ_σ_w} and all other free slots, and no fixed slot is in a block;
#   4. a run resumed from a DSGE intermediate save in the middle ends bitwise equal to the
#      uninterrupted run (particles, weights, per-block c), and it really resumed.
# Workers are used if the session has them (julia -p N), else the run is serial, as in the suite.

# Model1002 ss104 built for a 2026Q3 forecast (29 regimes, regime 2 starts in 2020Q1), with the spec of
# the blocked 2019Q4 estimation: no SPF short-run inflation expectations (add_avgshortinfl = false)
# and no anticipated mark-up shocks (add_ant_markup_shocks_ind = 0). likelihood_insample_regimes.jl
# builds the older production spec, with both on.
function build_ss104(; extra = Setting[])
    Model1002("ss104", custom_settings = vcat([Setting(:date_forecast_start, Date(2026, 9, 30)),
                                               Setting(:date_conditional_end, Date(2026, 9, 30)),
                                               Setting(:add_ant_markup_shocks_ind, 0),
                                               Setting(:add_avgshortinfl, false)], extra))
end

# Data simulated from regime 1 of `sys`, from date_presample_start(m) to `last_date`. Copied from
# likelihood_insample_regimes.jl (same function and seed; the draws differ, as the spec differs).
function simulated_data(m, sys, last_date, keep; rng = MersenneTwister(20261031))
    n_periods = DSGE.subtract_quarters(last_date, date_presample_start(m)) + 1
    mat(key) = sys isa RegimeSwitchingSystem ? sys[1, key] : sys[key]
    TTT, RRR, CCC, ZZ, DD, QQ = mat(:TTT), mat(:RRR), mat(:CCC), mat(:ZZ), mat(:DD), mat(:QQ)
    s = zeros(size(TTT, 1))
    data = fill(NaN, size(ZZ, 1), n_periods)
    obs_rows = [m.observables[k] for k in keep]
    i_zlb = DSGE.subtract_quarters(date_zlb_start(m), date_presample_start(m)) + 1
    ant_rows = [m.observables[k] for k in keys(m.observables) if occursin(r"^obs_nominalrate\d+$", string(k))]
    for t in 1:n_periods
        s = TTT * s + RRR * (sqrt.(diag(QQ)) .* randn(rng, size(QQ, 1))) + CCC
        data[obs_rows, t] = (ZZ * s + DD)[obs_rows]
        t < i_zlb && (data[intersect(obs_rows, ant_rows), t] .= NaN)
    end
    if haskey(DSGE.get_settings(m), :regime_eqcond_info)
        t_zlb = [DSGE.subtract_quarters(get_setting(m, :regime_dates)[r], date_presample_start(m)) + 1
                 for (r, e) in get_setting(m, :regime_eqcond_info) if e.alternative_policy.key == :zlb_rule]
        rate_rows = [m.observables[k] for k in keys(m.observables) if occursin(r"nominalrate", string(k))]
        data[rate_rows, Base.filter(t -> t <= n_periods, t_zlb)] .= NaN
    end
    return data
end

const LAST_Q   = Date(2019, 12, 31)
# Observations before 2017Q1 are left out (NaN). The sample still ends in 2019Q4, so the regimes
# and the in-sample solve are those of the full 2019Q4 sample, but the loglh is small enough for a
# 10-stage fixed tempering schedule. SMC's incremental weights are exp(Δφ * loglh), so with many
# more observations (1959Q3-2019Q4, or even 2010Q1-2019Q4) every weight of so short a schedule
# underflows to zero (a NaN ESS error).
const OBS_FROM = Date(2017, 3, 31)
const N_PART   = 300
const N_PHI    = 10
const STEP     = 2           # intermediate save every STEP stages
const K_RESUME = 6           # resume from this saved stage

regfixed(p, r) = haskey(p.regimes, :fixed) && haskey(p.regimes[:fixed], r) ? p.regimes[:fixed][r] : p.fixed
nregs(p)       = haskey(p.regimes, :value) ? length(p.regimes[:value]) : 1
# The slots of the regime-switching parameter vector, in its order (ModelConstructors.get_values):
# regime 1 of every parameter, then the extra regimes, parameter by parameter
slots(m) = vcat([(p.key, 1) for p in m.parameters], [(p.key, r) for p in m.parameters for r in 2:nregs(p)])
slotname((k, r)) = r == 1 ? k : Symbol(k, "_reg", r)

m = build_ss104()
for k in [:σ_r_m, :σ_r_m1, :σ_r_m2, :σ_r_m3, :σ_r_m4, :σ_r_m5, :σ_r_m6]   # as in the ss104 estimation scripts
    m[k].regimes[:fixed] = OrderedDict(1 => false, 2 => true)
end
m10  = Model1002("ss10")
keep = [k for k in keys(m10.observables) if haskey(m.observables, k)]
data = simulated_data(m, compute_system(m; tvis = true), LAST_Q, keep)
data[:, 1:DSGE.subtract_quarters(OBS_FROM, date_presample_start(m))] .= NaN

# Witness: ψ1 in model regimes 2-29 (after 2019Q4) fixed at 0.995, just below the Taylor principle.
# Gensys then fails after the sample for part of the prior (about 3 of 4 prior draws), so the full
# solve rejects those draws (-Inf) while their in-sample likelihood is finite. Wherever both are
# finite they agree bitwise (likelihood_insample_regimes.jl), so without the witness a run that
# silently used the full solve could not be told apart. Three parameter regimes, so that eqcond
# toggles ψ1 (as in likelihood_insample_regimes.jl).
ψ1 = m[:ψ1].value
ModelConstructors.set_regime_val!(m[:ψ1], 1, ψ1)
ModelConstructors.set_regime_val!(m[:ψ1], 2, 0.995)
ModelConstructors.set_regime_val!(m[:ψ1], 3, ψ1)
m[:ψ1].regimes[:fixed] = OrderedDict(1 => false, 2 => true, 3 => true)
n_regs = get_setting(m, :n_regimes)
get_setting(m, :model2para_regime)[:ψ1] = Dict(r => (r == 1 ? 1 : 2) for r in 1:n_regs)

# Fix every free parameter regime that model regime 1 does not use (as the 2019Q4 run did)
m2p = get_setting(m, :model2para_regime)
targets = [(p.key, r) for p in m.parameters for r in 2:nregs(p)
           if !regfixed(p, r) && (haskey(m2p, p.key) ? get(m2p[p.key], 1, 1) : 1) != r]
for key in unique(first.(targets))
    p = m[key]
    p.regimes[:fixed] = OrderedDict{Int, Bool}(r => (regfixed(p, r) || (key, r) in targets) for r in 1:nregs(p))
end
ModelConstructors.toggle_regime!(m.parameters, 1)

m <= Setting(:likelihood_insample_regimes_only, true)
m <= Setting(:sampling_method, :SMC)
m <= Setting(:saveroot, mktempdir())
m <= Setting(:use_parallel_workers, nprocs() > 1)
m <= Setting(:n_particles, N_PART)
m <= Setting(:use_fixed_schedule, true)
m <= Setting(:n_Φ, N_PHI)
m <= Setting(:λ, 5.0)
m <= Setting(:n_mh_steps_smc, 1)
m <= Setting(:n_smc_blocks, 1)
m <= Setting(:step_size_smc, 0.5)
m <= Setting(:mixture_proportion, 0.975)
m <= Setting(:resampling_threshold, 0.5)
m <= Setting(:smc_seed, 20261031)
m <= Setting(:smc_param_blocks, [[:ρ_σ_w]])
m <= Setting(:smc_blockwise_c, true)

all_slots  = slots(m)
fixed_slot = [regfixed(m[k], r) for (k, r) in all_slots]
free_names = [slotname(s) for (s, f) in zip(all_slots, fixed_slot) if !f]
v_start    = copy(ModelConstructors.get_values(m.parameters; regime_switching = true))
m_ref      = deepcopy(m)             # for fresh likelihoods: SMC leaves m at some particle

# As DSGE.smc2's likelihood closure, and as SMC sets a particle's parameters
function fresh_lik(mm, v; insample = true)
    ModelConstructors.toggle_regime!(mm.parameters, 1)
    DSGE.update!(mm, v)
    mm <= Setting(:likelihood_insample_regimes_only, insample)
    mm <= Setting(:preprocessed_transitions, Dict())
    L = likelihood(mm, data; sampler = false, catch_errors = true, verbose = :none,
                   use_chand_recursion = get_setting(mm, :use_chand_recursion))
    mm <= Setting(:likelihood_insample_regimes_only, true)
    return L
end

est(; kw...) = DSGE.estimate(m, data; verbose = :none, run_csminwel = false, save_intermediate = true,
                             intermediate_stage_increment = STEP, kw...)

@testset "SMC with param blocks and in-sample regimes (ss104, 2019Q4)" begin

@testset "Setup: 2019Q4 solves regime 1 only; the fixed slots and the witness" begin
    @test DSGE.insample_regime_count(m, LAST_Q; tvis = true, compute_pseudo = false) == 1
    @test length(targets) == 37                       # 35 slots with r = 2, σ_ziid/σ_biidc/σ_φ r = 3 ...
    @test (:ψ1, 2) ∉ targets && fixed_slot[findfirst(==((:ψ1, 2)), all_slots)]   # ... ψ1's were fixed above
    @test length(all_slots) == length(v_start)
    @test findall(fixed_slot) == ModelConstructors.get_fixed_para_inds(m.parameters; regime_switching = true)
    @test length(free_names) == 101 - 37
    L_in, L_full = fresh_lik(m_ref, v_start), fresh_lik(m_ref, v_start; insample = false)
    @test isfinite(L_in) && isequal(L_in, L_full)     # the witness leaves the defaults solvable
end

t_full = @elapsed est()
final  = rawpath(m, "estimate", "smc_cloud.jld2")
full   = load(final)
cloud  = full["cloud"]
n_para = length(v_start)
P      = cloud.particles[:, 1:n_para]
let h = full["stage_health"]
    rows = ["stage $(Int(h["stage"][s])): phi $(round(h["phi"][s], sigdigits = 3)), ESS $(round(h["ess"][s], digits = 1)), " *
            "distinct mutated $(h["distinct_mutated"][s]), accept ρ_σ_w/rest $(round.(h["accept_blocks"][:, s], digits = 3)), " *
            "c $(round.(h["c_blocks"][:, s], digits = 3))" for s in eachindex(h["stage"])]
    @info "param_blocks_insample: uninterrupted run, $(round(t_full, digits = 1)) s, $(nworkers()) worker(s), " *
          "$(length(unique(i -> P[i, :], 1:N_PART))) distinct final particles\n" * join(rows, "\n")
end

@testset "1. Stored loglh = fresh in-sample likelihood, bitwise; the in-sample path was used" begin
    loglh, logprior = SMC.get_loglh(cloud), SMC.get_logprior(cloud)
    @test size(P, 1) == N_PART && all(isfinite, loglh)
    L = [fresh_lik(m_ref, P[i, :]) for i in 1:N_PART]
    @test isequal(L, loglh)                          # bitwise (=== per element)
    @test [(ModelConstructors.toggle_regime!(m_ref.parameters, 1); DSGE.update!(m_ref, P[i, :]);
            ModelConstructors.prior(m_ref.parameters)) for i in 1:N_PART] == logprior
    # Witness: the full solve rejects some particles of the stage-2 save (a run with full-solve
    # likelihoods never holds such a particle: initial draws are redrawn until finite, and MH rejects
    # -Inf). Stage 2 is used, not the final cloud: the final cloud often descends from 1-4 ancestors
    # in the non-ρ_σ_w slots, so its count would be 1-4 draws, not 40. At stage 2 the particles are
    # still distinct prior draws, about 3 in 4 of which the full solve rejects.
    c2 = load(rawpath(m, "estimate", "smc_cloud") * "_stage=2.jld2", "cloud")
    P2, loglh2 = c2.particles[:, 1:n_para], SMC.get_loglh(c2)
    rows = unique(i -> P2[i, :], 1:N_PART)[1:min(20, end)]
    @test isequal([fresh_lik(m_ref, P2[i, :]) for i in rows], loglh2[rows])
    L_full = [fresh_lik(m_ref, P2[i, :]; insample = false) for i in rows]
    @info "param_blocks_insample: witness: the full solve rejects $(count(==(-Inf), L_full)) of $(length(rows)) distinct stage-2 particles"
    @test count(==(-Inf), L_full) > 0
    @test all(i -> L_full[i] == -Inf || L_full[i] === loglh2[rows[i]], eachindex(rows))
end

@testset "2. Every fixed slot is the same in all particles and equals its fixed value" begin
    fixed_inds = findall(fixed_slot)
    @test all(j -> all(==(v_start[j]), P[:, j]), fixed_inds)
    @test all(P[:, findfirst(==((:ψ1, 2)), all_slots)] .== 0.995)
    # the free slots were drawn (not left at their start values) and both blocks were mutated. Not
    # "takes more than one value": with a short schedule the final cloud can collapse onto a single
    # ancestor in the rest block (seen with the 2019Q4 run's spec)
    @test all(j -> any(!=(v_start[j]), P[:, j]), findall(.!fixed_slot))
    @test all(>(0), maximum(full["stage_health"]["accept_blocks"], dims = 2))
end

@testset "3. Blocks: {ρ_σ_w} and every other free slot; no fixed slot in a block" begin
    blocks = full["param_blocks"]
    @test blocks == [[:ρ_σ_w], Base.filter(!=(:ρ_σ_w), free_names)]
    @test full["blockwise_c"] === true
    @test full["stage_health"]["block_labels"] == ["ρ_σ_w", "rest"]
    fixed_names = Set(slotname(s) for (s, f) in zip(all_slots, fixed_slot) if f)
    @test isempty(intersect(fixed_names, Set(vcat(blocks...))))
    @test sort(vcat(blocks...)) == sort(free_names)  # every free slot in exactly one block
    @test size(full["c_blocks"], 1) == 2
end

@testset "4. Resume from the stage-$K_RESUME save = uninterrupted run, bitwise" begin
    stages = Int.(full["stage_health"]["stage"])      # stages 2..N_PHI (stage 1 is the prior draw)
    @test stages == collect(2:N_PHI) && K_RESUME in stages[2:end-1]
    ckpt = rawpath(m, "estimate", "smc_cloud") * "_stage=$(K_RESUME).jld2"
    @test isfile(ckpt)
    @test load(ckpt, "param_blocks") == full["param_blocks"] && load(ckpt, "blockwise_c") === true
    t_res = @elapsed est(continue_intermediate = true, intermediate_stage_start = K_RESUME)
    @info "param_blocks_insample: resumed run" seconds = round(t_res, digits = 1)
    res = load(final)
    @test res["cloud"].particles == full["cloud"].particles
    @test res["w"] == full["w"] && res["W"] == full["W"]
    @test res["c_blocks"] == full["c_blocks"]
    @test res["stage_health"]["c_blocks"] == full["stage_health"]["c_blocks"]
    @test res["stage_health"]["accept_blocks"] == full["stage_health"]["accept_blocks"]
    @test res["cloud"].c == full["cloud"].c
    # It resumed, not restarted (SMC's random streams are keyed by stage, so a silent restart would
    # also end bitwise equal): the stages up to the save keep the first run's times, the later
    # ones were run again
    tf, tr = full["stage_health"]["time"], res["stage_health"]["time"]
    before, after = findall(<=(K_RESUME), stages), findall(>(K_RESUME), stages)
    @test tr[before] == tf[before]
    @test all(tr[after] .!= tf[after])
end

end # outer testset

end # module
