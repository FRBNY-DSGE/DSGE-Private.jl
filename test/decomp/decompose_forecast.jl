module TestDecomposeForecast   # isolated from other test files' globals

using DSGE, Test, ModelConstructors, DataFrames, Dates, HDF5, Random

@testset "decompose_forecast (bugs doc §1.5): old data is placed on the new dates by date" begin
    q(y, n) = DSGE.quartertodate("$y-Q$n")
    new_dates = [q(2025, 4), q(2026, 1), q(2026, 2), q(2026, 3)]
    # Old frame shorter (the usual case): pad with missing at the end
    @test isequal(DSGE.align_by_date(new_dates, new_dates[1:3], [1., 2., 3.]), [1., 2., 3., missing])
    # Old frame longer than the new one: used to throw in vcat
    @test DSGE.align_by_date(new_dates[1:2], new_dates, [1., 2., 3., 4.]) == [1., 2.]
    # Different start dates: used to shift every value by the offset
    @test isequal(DSGE.align_by_date(new_dates, [q(2025, 3), q(2025, 4), q(2026, 1)], [9., 1., 2.]),
                  [1., 2., missing, missing])
end

@testset "decomposition_forecast solves at its parameters, not from a stale memo" begin
    # :preprocessed_transitions is keyed without parameter values. A caller's model that carries
    # a memo filled at other parameters (as after forecast_one) used to have it reused here.
    ref  = joinpath(dirname(@__FILE__), "..", "reference")
    data = h5read(joinpath(ref, "sim_data_regswitch_anschorfheide.h5"), "data")
    m = AnSchorfheide()
    m <= Setting(:regime_switching, true, true, "rs2", "")
    m <= Setting(:n_regimes, 3)
    m <= Setting(:regime_dates, Dict(1 => date_presample_start(m),
                                     2 => DSGE.iterate_quarters(Date(1960, 3, 31), 50),
                                     3 => DSGE.iterate_quarters(Date(1960, 3, 31), 100)))
    setup_regime_switching_inds!(m)
    m <= Setting(:model2para_regimes, Dict{Int, Int}(1 => 1, 2 => 1, 3 => 2))
    for i in 1:length(m.parameters), k in 1:2
        ModelConstructors.set_regime_val!(m.parameters[i], k, m.parameters[i].value)
    end
    DSGE.setup_param_regimes!(m, repeat([1 1 2], length(m.parameters)))
    nT = size(data, 2)
    m <= Setting(:date_forecast_start, DSGE.iterate_quarters(date_presample_start(m), nT))
    m <= Setting(:date_conditional_end, DSGE.iterate_quarters(date_presample_start(m), nT))
    m <= Setting(:forecast_horizons, 8)
    df = DataFrame(:date => collect(DSGE.quarter_range(date_presample_start(m),
                                                       DSGE.iterate_quarters(date_presample_start(m), nT - 1))))
    for (k, i) in m.observables
        df[!, k] = data[i, :]
    end

    θ0 = h5read(joinpath(ref, "regime_switching_anschorfheide_paramsmode_output.h5"), "params")
    Random.seed!(1)
    θ1 = θ0 .+ 0.3 .* (vec(rand(m.parameters, 1; regime_switching = true)) .- θ0)
    T, k, H = DSGE.decomposition_periods(m, m, df, df, :none, :none)

    fresh = DSGE.decomposition_forecast(deepcopy(m), df, θ1, :none, T, k, H)
    mm = deepcopy(m)
    mm <= Setting(:preprocessed_transitions, Dict())
    DSGE.update!(mm, θ0)
    compute_system(mm)                    # memo filled at θ0
    out = DSGE.decomposition_forecast(mm, df, θ1, :none, T, k, H)
    @test keys(out) == keys(fresh)
    @test all(isequal(out[key], fresh[key]) for key in keys(fresh))
end

# The full decomposition builds and forecasts ss104 several times (about 10 minutes), so it only
# runs with DSGE_SLOW_TESTS=true.
if get(ENV, "DSGE_SLOW_TESTS", "false") == "true"
@testset "decompose_forecast (bugs doc §1.5): ss104, the caller's models and data are left unchanged" begin
    function ss104_at(fcast_date)
        Model1002("ss104", custom_settings = [Setting(:date_forecast_start, fcast_date),
                                              Setting(:date_conditional_end, fcast_date),
                                              Setting(:add_ant_markup_shocks_ind, 2),
                                              Setting(:fix_ant_markup_shocks_0, false),
                                              Setting(:add_avgshortinfl, true),
                                              Setting(:forecast_horizons, 12)])
    end
    m_new = ss104_at(Date(2026, 9, 30))
    m_old = ss104_at(Date(2026, 6, 30))

    # Artificial data through each model's conditional quarter: GDP growth observed (zeros), everything
    # else missing. All-zero data makes the pegged-rate rows of the Kalman filter singular.
    function fake_df(m)
        dates = DSGE.quarter_range(date_presample_start(m), date_conditional_end(m))
        df = DataFrame(:date => collect(dates))
        for k in keys(m.observables)
            df[!, k] = allowmissing(k == :obs_gdp ? zeros(length(dates)) : fill(NaN, length(dates)))
        end
        df
    end
    df_new, df_old = fake_df(m_new), fake_df(m_old)
    # Model defaults as a flat parameter vector. Some regime-1 defaults (e.g. ρ_meas_π = 0) sit outside
    # their own bounds, which update! rejects, so move those just inside.
    function default_params(mm)
        v = ModelConstructors.get_values(mm.parameters; regime_switching = true)
        for (i, p) in enumerate(mm.parameters)
            p isa ModelConstructors.Parameter || continue
            lb, ub = p.valuebounds
            if !(lb <= v[i] <= ub) && ub - lb > 2e-6
                v[i] = clamp(v[i], lb + 1e-6, ub - 1e-6)
            end
        end
        v
    end
    p_new, p_old = default_params(m_new), default_params(m_old)

    m2p_new, m2p_old = deepcopy(get_setting(m_new, :model2para_regime)), deepcopy(get_setting(m_old, :model2para_regime))
    regime_values(mm) = [haskey(p.regimes, :value) ? copy(p.regimes[:value]) : p.value for p in mm.parameters]
    vals_old = regime_values(m_old)
    df_old_copy = deepcopy(df_old)

    out = DSGE.decompose_forecast(m_new, m_old, df_new, df_old, p_new, p_old, :full, :full, [:obs];
                                  catch_smoother_lapack = true)
    @test haskey(out, :decompdataobs) || !isempty(out)

    @test get_setting(m_new, :model2para_regime) == m2p_new          # was changed for σ_condgdp/σ_condcorepce
    @test get_setting(m_old, :model2para_regime) == m2p_old
    @test get_setting(m_old, :model2para_regime) !== get_setting(m_new, :model2para_regime)   # no shared dict
    @test all(m_old.parameters[i] !== m_new.parameters[i] for i in eachindex(m_old.parameters))
    @test regime_values(m_old) == vals_old
    @test isequal(df_old, df_old_copy)                                # pgap/ygap rows and columns were edited
end
end

end # module TestDecomposeForecast
