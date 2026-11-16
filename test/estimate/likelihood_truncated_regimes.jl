module TestLikelihoodTruncatedRegimes

using DSGE, ModelConstructors, Dates, Random, LinearAlgebra, Test

const m = Model1002("ss104", custom_settings = [
    Setting(:date_forecast_start, Date(2026, 9, 30)),
    Setting(:date_conditional_end, Date(2026, 9, 30)),
    Setting(:add_ant_markup_shocks_ind, 2),
    Setting(:fix_ant_markup_shocks_0, false),
    Setting(:add_avgshortinfl, true),
])
m <= Setting(:likelihood_insample_regimes_only, false)

function simulated_data(m, sys, last_date)
    n_periods = DSGE.subtract_quarters(last_date, date_presample_start(m)) + 1
    TTT, RRR, CCC = (sys[1, key] for key in (:TTT, :RRR, :CCC))
    ZZ, DD, QQ = (sys[1, key] for key in (:ZZ, :DD, :QQ))
    state = zeros(size(TTT, 1))
    data = fill(NaN, size(ZZ, 1), n_periods)
    keep = keys(Model1002("ss10").observables)
    rows = [m.observables[key] for key in keep if haskey(m.observables, key)]
    rng = MersenneTwister(20261113)
    for t in 1:n_periods
        state = TTT * state + RRR * (sqrt.(diag(QQ)) .* randn(rng, size(QQ, 1))) + CCC
        data[rows, t] = (ZZ * state + DD)[rows]
    end
    # The early data have no anticipated rates; omit the rate during the temporary ZLB.
    ant_rows = [m.observables[key] for key in keys(m.observables)
                if occursin(r"^obs_nominalrate\d+$", string(key))]
    i_zlb = DSGE.subtract_quarters(date_zlb_start(m), date_presample_start(m)) + 1
    data[ant_rows, 1:min(n_periods, i_zlb - 1)] .= NaN
    zlb_dates = [get_setting(m, :regime_dates)[r] for (r, entry) in get_setting(m, :regime_eqcond_info)
                 if entry.alternative_policy.key == :zlb_rule]
    rate_rows = [m.observables[key] for key in keys(m.observables)
                 if occursin("nominalrate", string(key))]
    for date in zlb_dates
        t = DSGE.subtract_quarters(date, date_presample_start(m)) + 1
        t <= n_periods && (data[rate_rows, t] .= NaN)
    end
    data
end

const full_system = compute_system(m; tvis = true)
const data = simulated_data(m, full_system, Date(2021, 6, 30))
const n_regimes = get_setting(m, :n_regimes)
const matrices = (:TTT, :RRR, :CCC, :ZZ, :DD, :QQ, :EE)

@testset "truncated measurement solve is bitwise equal to full solve" begin
    @test get_setting(m, :uncertain_altpolicy)
    for last_date in (Date(2019, 12, 31), Date(2021, 6, 30))
        k = DSGE.likelihood_regime_count(m, last_date, n_regimes)
        @test 1 <= k < n_regimes
        sample = data[:, 1:DSGE.subtract_quarters(last_date, date_presample_start(m)) + 1]

        m <= Setting(:likelihood_truncated_regimes, false)
        @test DSGE.likelihood_measured_regime_count(m, last_date; compute_pseudo = false) == 0
        full = compute_system(m; tvis = true, sample_end_date = last_date, compute_pseudo = false)
        full_likelihood = likelihood(m, sample; catch_errors = true, verbose = :none)
        @test isfinite(full_likelihood)

        m <= Setting(:likelihood_truncated_regimes, true)
        @test DSGE.likelihood_measured_regime_count(m, last_date; compute_pseudo = false) == k
        truncated = compute_system(m; tvis = true, sample_end_date = last_date, compute_pseudo = false)
        @test length(truncated.transitions) == k
        @test all(isequal(truncated[r, key], full[r, key]) for r in 1:k for key in matrices)
        @test isequal(likelihood(m, sample; catch_errors = true, verbose = :none), full_likelihood)
        @test length(compute_system(m; tvis = true, sample_end_date = last_date).transitions) == n_regimes
    end
end

end # module
