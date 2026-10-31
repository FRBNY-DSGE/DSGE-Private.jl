# A module, so globals of other test files run in the same session can't shadow names used here,
# and this file's can't leak out
module TestLikelihoodInsampleRegimes

using DSGE, ModelConstructors, Test, Dates, Random, LinearAlgebra

# With :likelihood_insample_regimes_only, a likelihood on data that end before some regimes start
# builds only the regimes it uses, when they do not depend on later ones (insample_regime_count).
# The likelihood must then be bitwise identical to the full solve, and the full solve must be
# used whenever an observed regime depends on later ones.

# Model1002 ss104, configured as the production forecast builds it (2026Q3 forecast: 29 regimes,
# regime 2 starts in 2020Q1, temporary ZLB and imperfectly credible policies from regime 5 (2020Q4)).
function build_ss104(; extra = Setting[])
    Model1002("ss104", custom_settings = vcat([Setting(:date_forecast_start, Date(2026, 9, 30)),
                                               Setting(:date_conditional_end, Date(2026, 9, 30)),
                                               Setting(:add_ant_markup_shocks_ind, 2),
                                               Setting(:fix_ant_markup_shocks_0, false),
                                               Setting(:add_avgshortinfl, true)], extra))
end

# Data simulated from regime 1 of `sys`, from date_presample_start(m) to `last_date`. Only the
# observables in `keep` are observed, and anticipated policy rates only from the ZLB on, as in the data.
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
    # Policy rates are pinned during the temporary ZLB (2020Q4-2021Q4 in ss104); these data are from
    # regime 1, so leave them out there
    if haskey(DSGE.get_settings(m), :regime_eqcond_info)
        t_zlb = [DSGE.subtract_quarters(get_setting(m, :regime_dates)[r], date_presample_start(m)) + 1
                 for (r, e) in get_setting(m, :regime_eqcond_info) if e.alternative_policy.key == :zlb_rule]
        rate_rows = [m.observables[k] for k in keys(m.observables) if occursin(r"nominalrate", string(k))]
        data[rate_rows, Base.filter(t -> t <= n_periods, t_zlb)] .= NaN
    end
    return data
end

function lik(m, data, insample)
    if isnothing(insample)
        delete!(DSGE.get_settings(m), :likelihood_insample_regimes_only)
    else
        m <= Setting(:likelihood_insample_regimes_only, insample)
    end
    m <= Setting(:preprocessed_transitions, Dict())
    return likelihood(m, data; catch_errors = true, verbose = :none)
end

# Defaults, and two draws a twentieth of the way from them to prior draws
function parameter_points(m; n = 2)
    RS = haskey(DSGE.get_settings(m), :regime_switching) && get_setting(m, :regime_switching)
    pv0 = deepcopy(m.parameters)
    RS && ModelConstructors.toggle_regime!(pv0, 1)
    v0 = ModelConstructors.get_values(pv0; regime_switching = RS)
    pts = [copy(v0)]
    for i in 1:n
        Random.seed!(20261101 + i)
        d = vec(rand(deepcopy(pv0), 1; regime_switching = RS, toggle = true))
        push!(pts, v0 .+ 0.05 .* (d .- v0))
    end
    return pts, pv0, RS
end
# As SMC sets a particle's parameters
function set_point!(m, pv0, v, RS)
    RS && ModelConstructors.toggle_regime!(m.parameters, 1)
    DSGE.update!(m, v)
end

const MATRICES = (:TTT, :RRR, :CCC, :ZZ, :DD, :QQ, :EE)
const PROP_PARAMS = (:σ_φ_prop, :σ_ziid_prop, :σ_biidc_prop) # two regimes, read by eqcond without toggling

m10  = Model1002("ss10")
m104 = build_ss104()
n_regs = get_setting(m104, :n_regimes)
keep = [k for k in keys(m10.observables) if haskey(m104.observables, k)]
data104 = simulated_data(m104, compute_system(m104; tvis = true), Date(2026, 6, 30), keep)
data10  = simulated_data(m10, compute_system(m10), Date(2019, 12, 31), collect(keys(m10.observables)))
n_cols(d) = DSGE.subtract_quarters(d, date_presample_start(m104)) + 1
pts, pv0, RS = parameter_points(m104)

# One outer testset, so that every inner testset runs and reports even if another fails
@testset "Likelihood with in-sample regimes only" begin

@testset "In-sample regimes: off unless :likelihood_insample_regimes_only is true" begin
    m = deepcopy(m104)
    d = Date(2019, 12, 31)
    @test DSGE.insample_regime_count(m, d; tvis = true, compute_pseudo = false) == 0
    m <= Setting(:likelihood_insample_regimes_only, false)
    @test DSGE.insample_regime_count(m, d; tvis = true, compute_pseudo = false) == 0
    @test length(compute_system(m; tvis = true, sample_end_date = d, compute_pseudo = false).transitions) == n_regs
    m <= Setting(:likelihood_insample_regimes_only, true)
    @test DSGE.insample_regime_count(m, d; tvis = true, compute_pseudo = false) == 1
    @test DSGE.insample_regime_count(m, d; tvis = true, compute_pseudo = true) == 0 # not for forecasting
    @test DSGE.insample_regime_count(m, nothing; tvis = true, compute_pseudo = false) == 0
end

@testset "In-sample regimes: full-sample likelihoods are unchanged (ss104, ss10)" begin
    m = deepcopy(m104)
    m <= Setting(:likelihood_insample_regimes_only, true)
    # Data end in 2026Q2 (regime 27): the observed regimes include the temporary ZLB and the
    # imperfectly credible policies
    @test DSGE.insample_regime_count(m, Date(2026, 6, 30); tvis = true, compute_pseudo = false) == 0
    for v in pts
        set_point!(m, pv0, v, RS)
        L_default = lik(m, data104, nothing)
        @test isfinite(L_default)
        @test isequal(lik(m, data104, true), L_default)
        @test isequal(lik(m, data104, false), L_default)
    end
    # ss10 does not switch regimes, so the setting does nothing
    mm = deepcopy(m10)
    L_default = lik(mm, data10, nothing)
    @test isfinite(L_default)
    @test isequal(lik(mm, data10, true), L_default)
    @test isequal(lik(mm, data10, false), L_default)
end

@testset "In-sample regimes: ss104 on samples ending in $d match the full solve bitwise" for (d, k) in
    ((Date(2019, 12, 31), 1), (Date(2020, 9, 30), 4))
    m = deepcopy(m104)
    m <= Setting(:likelihood_insample_regimes_only, true)
    @test DSGE.insample_regime_count(m, d; tvis = true, compute_pseudo = false) == k
    data = data104[:, 1:n_cols(d)]
    for v in pts
        set_point!(m, pv0, v, RS)
        L_full = lik(m, data, false)
        @test isfinite(L_full)
        @test isequal(lik(m, data, true), L_full)

        # The in-sample regimes' matrices are those of the full system
        m <= Setting(:likelihood_insample_regimes_only, true)
        sys_in = compute_system(m; tvis = true, sample_end_date = d, compute_pseudo = false)
        sys_full = compute_system(m; tvis = true)
        @test length(sys_in.transitions) == k
        @test all(isequal(sys_in[r, mat], sys_full[r, mat]) for r in 1:k for mat in MATRICES)

        # Whatever regime the caller left the parameters in. eqcond does not toggle parameters
        # with two regimes, and the full solve builds these regimes after the perfectly credible
        # systems have put every parameter back in regime 1.
        set_point!(m, pv0, v, RS)
        for key in PROP_PARAMS
            ModelConstructors.toggle_regime!(m[key], 2)
        end
        @test m[:σ_φ_prop].value != m[:σ_φ_prop].regimes[:value][1]
        L_full2 = lik(m, data, false)
        for key in PROP_PARAMS
            ModelConstructors.toggle_regime!(m[key], 2)
        end
        @test isequal(lik(m, data, true), L_full2)
        # σ_φ etc. are 0 in regime 1, so check the matrices too (RRR differs if eqcond reads regime 2)
        toggle_props!(m) = foreach(key -> ModelConstructors.toggle_regime!(m[key], 2), PROP_PARAMS)
        toggle_props!(m)
        sys_full2 = compute_system(m; tvis = true)
        toggle_props!(m)
        sys_in2 = compute_system(m; tvis = true, sample_end_date = d, compute_pseudo = false)
        @test all(isequal(sys_in2[r, mat], sys_full2[r, mat]) for r in 1:k for mat in MATRICES)
        @test all(isequal(sys_in2[r, mat], sys_full[r, mat]) for r in 1:k for mat in MATRICES)
    end
end

@testset "In-sample regimes: the full solve is used when observed regimes depend on later ones" begin
    m = deepcopy(m104)
    m <= Setting(:likelihood_insample_regimes_only, true)

    # Data end in 2021Q2 (regime 7). Regimes 5-10 have a temporary ZLB (gensys2, regimes 4:10) under
    # imperfect awareness, and their information sets run to regime 10
    d = Date(2021, 6, 30)
    @test DSGE.likelihood_regime_count(m, d, n_regs) == 7
    @test DSGE.insample_regime_count(m, d; tvis = true, compute_pseudo = false) == 0
    @test length(compute_system(m; tvis = true, sample_end_date = d, compute_pseudo = false).transitions) == n_regs
    data = data104[:, 1:n_cols(d)]
    for v in pts
        set_point!(m, pv0, v, RS)
        @test isequal(lik(m, data, true), lik(m, data, false))
    end
    m <= Setting(:likelihood_insample_regimes_only, true) # lik changed it

    # Expectations in regime 1 formed with the transition matrices of regimes 1-4: in the full
    # system regime 1's measurement equation then changes with regime 4's solution
    d = Date(2019, 12, 31)
    m_tvis = deepcopy(m)
    infosets = copy(get_setting(m_tvis, :tvis_information_set))
    infosets[1] = 1:4
    m_tvis <= Setting(:tvis_information_set, infosets)
    @test DSGE.insample_regime_count(m_tvis, d; tvis = true, compute_pseudo = false) == 0
    @test DSGE.insample_regime_count(m_tvis, d; tvis = false, compute_pseudo = false) == 1
    sys = compute_system(m_tvis; tvis = true)
    TTTs = [sys[r, :TTT] for r in 1:4]
    CCCs = [sys[r, :CCC] for r in 1:4]
    meas(TTTs) = measurement(m_tvis, sys[1, :TTT], sys[1, :RRR], sys[1, :CCC]; reg = 1, TTTs = TTTs, CCCs = CCCs,
                             information_set = 1:4)
    TTTs2 = copy(TTTs); TTTs2[4] = 0.9 .* TTTs[4]
    @test meas(TTTs)[:ZZ] != meas(TTTs2)[:ZZ]

    # An observed regime with an alternative or temporary policy (the regimes of the temporary ZLB
    # are those in :regime_eqcond_info), or one copied from another regime
    # (a temporary ZLB in regime 1 also moves the gensys2 regimes; an imperfectly credible
    # flexible AIT does not, so only :regime_eqcond_info rules it out)
    for r_from in (5, 10)
        m_pol = deepcopy(m)
        rei = deepcopy(get_setting(m_pol, :regime_eqcond_info))
        rei[1] = rei[r_from]
        m_pol <= Setting(:regime_eqcond_info, rei)
        @test DSGE.insample_regime_count(m_pol, d; tvis = true, compute_pseudo = false) == 0
    end
    m_pol = deepcopy(m)
    rei = deepcopy(get_setting(m_pol, :regime_eqcond_info))
    rei[1] = rei[10]
    m_pol <= Setting(:regime_eqcond_info, rei)
    @test DSGE.compute_gensys_gensys2_regimes(m_pol) == DSGE.compute_gensys_gensys2_regimes(m)
    for key in (:identical_eqcond_regimes, :perfect_credibility_identical_transitions)
        m_copy = deepcopy(m)
        m_copy <= Setting(key, Dict(1 => 1))
        @test DSGE.insample_regime_count(m_copy, d; tvis = true, compute_pseudo = false) == 0
    end
    # gensys2 regimes starting before the end of the sample (here regimes 1:10 after removing
    # the policies of regimes 5 on, so that only the gensys2 ranges decide)
    m_g2 = deepcopy(m)
    delete!(DSGE.get_settings(m_g2), :regime_eqcond_info)
    m_g2 <= Setting(:gensys_regimes, [11:n_regs])
    m_g2 <= Setting(:gensys2_regimes, [1:10])
    @test DSGE.compute_gensys_gensys2_regimes(m_g2) == ([11:n_regs], [1:10])
    @test DSGE.insample_regime_count(m_g2, d; tvis = true, compute_pseudo = false) == 0
    m_g2 <= Setting(:gensys_regimes, [1:4, 11:n_regs])
    m_g2 <= Setting(:gensys2_regimes, [4:10])
    @test DSGE.insample_regime_count(m_g2, Date(2020, 9, 30); tvis = true, compute_pseudo = false) == 4
    @test DSGE.insample_regime_count(m_g2, Date(2020, 12, 31); tvis = true, compute_pseudo = false) == 0
    # Settings that change other settings during the full solve
    for (key, val) in ((:set_pgap1, ([10], 0.0)), (:tvis_regime_eqcond_info, [get_setting(m, :regime_eqcond_info)]))
        m_set = deepcopy(m)
        m_set <= Setting(key, val)
        @test DSGE.insample_regime_count(m_set, d; tvis = true, compute_pseudo = false) == 0
    end

    # Models without a regime_is_self_contained method keep the full solve
    m_as = AnSchorfheide()
    m_as <= Setting(:regime_switching, true)
    m_as <= Setting(:n_regimes, 2)
    m_as <= Setting(:regime_dates, Dict(1 => date_presample_start(m_as),
                                        2 => DSGE.iterate_quarters(date_presample_start(m_as), 40)))
    setup_regime_switching_inds!(m_as)
    m_as <= Setting(:likelihood_insample_regimes_only, true)
    @test !DSGE.regime_is_self_contained(m_as, 1, 1:1)
    @test DSGE.insample_regime_count(m_as, DSGE.iterate_quarters(date_presample_start(m_as), 10);
                                     compute_pseudo = false) == 0
end

@testset "In-sample regimes: a draw that only fails to solve after the sample is no longer rejected" begin
    # ψ1 = 0.1 in regimes 2-29 violates the Taylor principle there, so gensys finds no unique
    # solution after 2019Q4. Three parameter regimes, so that eqcond toggles ψ1.
    m = deepcopy(m104)
    ψ1 = m[:ψ1].value
    ModelConstructors.set_regime_val!(m[:ψ1], 1, ψ1)
    ModelConstructors.set_regime_val!(m[:ψ1], 2, 0.1)
    ModelConstructors.set_regime_val!(m[:ψ1], 3, ψ1)
    get_setting(m, :model2para_regime)[:ψ1] = Dict(r => (r == 1 ? 1 : 2) for r in 1:n_regs)
    data = data104[:, 1:n_cols(Date(2019, 12, 31))]
    @test lik(m, data, false) == -Inf
    @test lik(m, data, nothing) == -Inf
    L_in = lik(m, data, true)
    @test isfinite(L_in)
    @test isequal(L_in, lik(deepcopy(m104), data, false)) # regime 1 is unchanged
end

end # outer testset

end # module
