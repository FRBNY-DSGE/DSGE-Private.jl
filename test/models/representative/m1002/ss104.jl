using DSGE, ModelConstructors, Test, Dates, DataFrames

# Model1002 ss104, configured as the production forecast builds it (2026Q3 forecast,
# two anticipated mark-up shocks switched on). No data or saved output is needed.
function build_ss104(; fcast_date = Date(2026, 9, 30), extra = Setting[])
    Model1002("ss104", custom_settings = vcat([Setting(:date_forecast_start, fcast_date),
                                               Setting(:date_conditional_end, fcast_date),
                                               Setting(:add_ant_markup_shocks_ind, 2),
                                               Setting(:fix_ant_markup_shocks_0, false),
                                               Setting(:add_avgshortinfl, true)], extra))
end

# Value of regime-switching parameter `k` in model regime `r`
function regime_value(m, k::Symbol, r::Int)
    p = m[k]
    haskey(p.regimes, :value) || return p.value
    m2p = get_setting(m, :model2para_regime)
    preg = haskey(m2p, k) ? m2p[k][r] : r
    return p.regimes[:value][preg]
end

m = build_ss104()
sys = compute_system(m)
exo = m.exogenous_shocks
n_regimes = get_setting(m, :n_regimes)

# Variance the anticipated mark-up shock i should have in model regime r: σ_λ_f$i only marks the
# shock as on (> 0) or off (0); when on, the standard deviation is σ_λ_f (bugs doc §1.2)
ant_on(m, i, r) = regime_value(m, Symbol("σ_λ_f$i"), r) > 0
ant_var(m, i, r) = ant_on(m, i, r) ? regime_value(m, :σ_λ_f, r)^2 : 0.0
ant_qq(sys, m, i, r) = sys[r, :QQ][m.exogenous_shocks[Symbol("λ_f_ant_sh$i")], m.exogenous_shocks[Symbol("λ_f_ant_sh$i")]]

@testset "ss104 (bugs doc §1.1): QQ holds variances for the anticipated mark-up shocks" begin
    for r in 1:n_regimes, i in 1:get_setting(m, :add_ant_markup_shocks_ind)
        @test ant_qq(sys, m, i, r) ≈ ant_var(m, i, r) atol = 1e-15
    end
    # Same convention as the contemporaneous mark-up shock
    for r in 1:n_regimes
        @test sys[r, :QQ][exo[:λ_f_sh], exo[:λ_f_sh]] ≈ regime_value(m, :σ_λ_f, r)^2
    end
end

@testset "ss104 (bugs doc §1.2): anticipated mark-up σ follows σ_λ_f" begin
    m2 = deepcopy(m)
    ModelConstructors.set_regime_val!(m2[:σ_λ_f], 1, 0.25)   # e.g. a posterior draw
    sys2 = compute_system(m2)
    on_regs = [r for r in 1:n_regimes if ant_on(m2, 1, r)]
    @test !isempty(on_regs)
    for r in on_regs
        @test regime_value(m2, :σ_λ_f, r) == 0.25
        @test ant_qq(sys2, m2, 1, r) ≈ 0.25^2
    end
    # Zeroing σ_λ_f$i in the model turns the shock off
    for i in 1:2, r in keys(m2[Symbol("σ_λ_f$i")].regimes[:value])
        m2[Symbol("σ_λ_f$i")].regimes[:value][r] = 0.
    end
    sys3 = compute_system(m2)
    for r in 1:n_regimes, i in 1:2
        @test ant_qq(sys3, m2, i, r) == 0.
    end
end

# Dates of the model regimes in which anticipated mark-up shock i is on
ant_on_dates(m, i) = sort([get_setting(m, :regime_dates)[r] for r in 1:get_setting(m, :n_regimes) if ant_on(m, i, r)])

@testset "ss104 (bugs doc §1.3): anticipated mark-up regimes are found by date and switched as a pair" begin
    # Production: 1-quarter-ahead shock on in 2025Q2-Q3, 2-quarter-ahead shock in 2025Q2 only
    @test ant_on_dates(m, 1) == [Date(2025, 6, 30), Date(2025, 9, 30)]
    @test ant_on_dates(m, 2) == [Date(2025, 6, 30)]

    # A different forecast origin gives a different regime layout; the dates must not move
    m_q2 = build_ss104(fcast_date = Date(2026, 6, 30))
    @test ant_on_dates(m_q2, 1) == [Date(2025, 6, 30), Date(2025, 9, 30)]
    @test ant_on_dates(m_q2, 2) == [Date(2025, 6, 30)]

    # fix_ant_markup_shocks_0 = true switches both shocks off (σ_λ_f2 used to stay on in 2025Q2)
    m_off = build_ss104(extra = [Setting(:fix_ant_markup_shocks_0, true)])
    sys_off = compute_system(m_off)
    for i in 1:2
        @test isempty(ant_on_dates(m_off, i))
        @test all(ant_qq(sys_off, m_off, i, r) == 0. for r in 1:get_setting(m_off, :n_regimes))
    end

    # More shocks than ss104 defines on-regimes for is an error, not silently dead shocks
    @test_throws AssertionError build_ss104(extra = [Setting(:add_ant_markup_shocks_ind, 3)])
end

@testset "m1002 (bugs doc §1.6): the 2020:Q3 flexible-AIT regime is found when it is not regime 4" begin
    m6 = deepcopy(m)
    m6 <= Setting(:flexible_ait_policy_change_date, Date(2020, 9, 30))
    rd = get_setting(m6, :regime_dates)
    r_q3 = only([r for (r, d) in rd if d == Date(2020, 9, 30)])
    @test DSGE.flexible_ait_policy_change_regime(m6) == r_q3

    # Shift every regime after the first by one (as an extra early regime would): the lookup must
    # follow the date. It used to assign the result to a misspelled variable and return 0.
    shifted = Dict{Int, Date}(1 => rd[1])
    for r in 2:get_setting(m6, :n_regimes)
        shifted[r + 1] = rd[r]
    end
    shifted[2] = Date(1990, 3, 31)
    m6 <= Setting(:regime_dates, shifted)
    m6 <= Setting(:n_regimes, length(shifted))
    @test DSGE.flexible_ait_policy_change_regime(m6) == r_q3 + 1

    # No regime starts on the date: a clear AssertionError, not an UndefVarError
    m6 <= Setting(:flexible_ait_policy_change_date, Date(2020, 8, 15))
    @test_throws AssertionError DSGE.flexible_ait_policy_change_regime(m6)
end

@testset "ss104 (bugs doc §1.4): filter with add_zlb_duration leaves the model's policies unchanged" begin
    m4 = deepcopy(m)
    m4 <= Setting(:zlb_duration, 4)
    policies(mm) = Dict(r => e.alternative_policy.key for (r, e) in get_setting(mm, :regime_eqcond_info))
    settings_to_keep = [:reg_forecast_start, :reg_post_conditional_end, :n_fcast_regimes, :n_hist_regimes,
                        :temporary_altpolicy_length]
    before_pol = policies(m4)
    before_set = Dict(k => get_setting(m4, k) for k in settings_to_keep)
    @test any(!=(:flexible_ait), values(before_pol))   # otherwise the test could not see the bug

    nT   = DSGE.subtract_quarters(date_forecast_start(m4), date_presample_start(m4))
    data = fill(NaN, n_observables(m4), nT)   # all missing: the filter runs without singular-variance rows
    t_zlb = DSGE.subtract_quarters(Date(2020, 12, 31), date_presample_start(m4)) + 1
    # Whether or not the implied-duration forecast succeeds on this artificial data, the model's
    # settings must be restored afterwards
    # The settings are only changed once the filter has run, so the filter itself must succeed here
    # or the test could not see the bug
    kal = DSGE.filter(m4, data, sys; outputs = [:loglh, :filt])
    @test size(kal[:s_filt], 2) == nT
    try
        DSGE.filter_likelihood(m4, data, sys; add_zlb_duration = (true, t_zlb))
    catch e
        @info "filter_likelihood threw after the filter (settings are still checked)" e
    end
    @test policies(m4) == before_pol
    for k in settings_to_keep
        @test get_setting(m4, k) == before_set[k]
    end
end

@testset "ss104 (bugs doc §1.10): regime-switching smooth with filter_smooth = true" begin
    m10 = deepcopy(m)
    dates = DSGE.quarter_range(date_presample_start(m10), DSGE.iterate_quarters(date_forecast_start(m10), -1))
    df = DataFrame(:date => collect(dates))
    # Only GDP growth observed: avoids the singular pegged-rate rows that all-zero data gives, and
    # keeps a numeric data matrix (an all-missing frame becomes Matrix{Missing})
    for k in keys(m10.observables)
        df[!, k] = k == :obs_gdp ? zeros(length(dates)) : fill(NaN, length(dates))
    end
    kal = DSGE.filter(m10, df, sys)
    for smoother in [:durbin_koopman, :koopman, :hamilton, :carter_kohn]
        m10 <= Setting(:forecast_smoother, smoother)
        base = smooth(m10, df, sys; draw_states = false)
        # Used to throw UndefVarError (stil_pred etc.) for every smoother except Carter-Kohn
        withfilt = smooth(m10, df, sys; draw_states = false, filter_smooth = true,
                          s_pred = kal[:s_pred], P_pred = kal[:P_pred], s_filt = kal[:s_filt], P_filt = kal[:P_filt])
        @test withfilt[1] ≈ base[1]   # smoothed states
        @test withfilt[2] ≈ base[2]   # smoothed shocks
    end
end
