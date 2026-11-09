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

@testset "align_rows puts an old model's rows on the new model's variables by name" begin
    x = [1. 2.; 3. 4.; 5. 6.]
    @test DSGE.align_rows(x, [:a, :b, :c], [:a, :b, :c]) === x       # same variables: untouched
    y = DSGE.align_rows(x, [:a, :b, :c], [:c, :a])
    @test y == [5. 6.; 1. 2.]
    z = DSGE.align_rows(x, [:a, :b, :c], [:b, :d])
    @test z[1, :] == [3., 4.] && all(isnan, z[2, :])
    s = reshape(collect(1.:12.), 3, 2, 2)                             # shockdec-shaped
    @test DSGE.align_rows(s, [:a, :b, :c], [:c, :a]) == s[[3, 1], :, :]
    @test_throws DimensionMismatch DSGE.align_rows(x, [:a, :b], [:a])
    @test DSGE.dict_names_by_index(Dict(:b => 2, :a => 1, :c => 3)) == [:a, :b, :c]
end

# Production's November ss104 (m_new) has no anticipated mark-up shocks and no SPF short-run inflation
# observable; the September model it is compared with (m_old) has both: 130 vs 132 parameters,
# 48 vs 50 shocks, 28 vs 29 observables.
ss104_new(fd) = Model1002("ss104", custom_settings = [Setting(:date_forecast_start, fd),
                                                      Setting(:date_conditional_end, fd),
                                                      Setting(:forecast_horizons, 12)])
ss104_old(fd) = Model1002("ss104", custom_settings = [Setting(:date_forecast_start, fd),
                                                      Setting(:date_conditional_end, fd),
                                                      Setting(:add_ant_markup_shocks_ind, 2),
                                                      Setting(:fix_ant_markup_shocks_0, false),
                                                      Setting(:add_avgshortinfl, true),
                                                      Setting(:forecast_horizons, 12)])

# The model's flat parameter vector with each free regime-1 value moved a fraction of the way toward
# its upper bound (or +1). Some regime-1 defaults (e.g. ρ_meas_π = 0) sit outside their own bounds,
# which update! rejects, so every value is also kept just inside its bounds.
function shifted_params(mm, frac)
    v = ModelConstructors.get_values(mm.parameters; regime_switching = true)
    for (i, p) in enumerate(mm.parameters)
        p isa ModelConstructors.Parameter || continue
        lb, ub = p.valuebounds
        ub - lb > 2e-6 || continue
        p.fixed || (v[i] += frac * (min(ub, v[i] + 1) - v[i]))
        v[i] = clamp(v[i], lb + 1e-6, ub - 1e-6)
    end
    v
end

@testset "new_params_on_old_model: new estimates on an old model with other parameters, by name" begin
    m_new, m_old = ss104_new(Date(2026, 9, 30)), ss104_old(Date(2026, 6, 30))
    @test length(m_new.parameters) < length(m_old.parameters)
    @test !DSGE.same_parameter_names(m_new, m_old)
    @test DSGE.same_parameter_names(m_old, deepcopy(m_old))

    p_new, p_old = shifted_params(m_new, 0.10), shifted_params(m_old, 0.05)
    mn = deepcopy(m_new); DSGE.update!(mn, p_new)
    mo = deepcopy(m_old); DSGE.update!(mo, p_old)

    m_old_copy = deepcopy(m_old)
    pars, vals = DSGE.new_params_on_old_model(m_new, m_old, p_new, p_old)
    @test [p.key for p in pars] == [p.key for p in m_old.parameters]       # old model's order
    is_new(k) = get(mn.keys, k, 0) in 1:length(mn.parameters)
    val1(p) = haskey(p.regimes, :value) ? p.regimes[:value][1] : p.value   # regime-1 value
    @test all(val1(p) == val1((is_new(p.key) ? mn : mo)[p.key]) for p in pars)
    @test all(isequal(get(p.regimes, :value, nothing), get((is_new(p.key) ? mn : mo)[p.key].regimes, :value, nothing))
              for p in pars)
    @test val1(pars[m_old.keys[:σ_λ_f1]]) == val1(mo[:σ_λ_f1])              # old-only: old value
    @test val1(pars[m_old.keys[:σ_g]]) == val1(mn[:σ_g])                    # shared: new value
    @test val1(mn[:σ_g]) != val1(mo[:σ_g])
    # Neither input model is changed
    @test all(m_old.parameters[i].value == m_old_copy.parameters[i].value for i in eachindex(m_old.parameters))

    # The forecast step can use them: update! with the flat values reproduces them, and the
    # steady state and the system solve
    mb = deepcopy(m_old)
    mb.parameters = deepcopy(pars)
    DSGE.update!(mb, vals)
    @test ModelConstructors.get_values(mb.parameters; regime_switching = true) == vals
    @test mb[:σ_λ_f1].key == :σ_λ_f1
    @test (compute_system(mb; tvis = true); true)

    # The old approach: m_new's parameters on m_old, with m_old's stale name -> index table
    bad = deepcopy(m_old)
    bad.parameters = deepcopy(mn.parameters)
    @test bad[:σ_λ_f1].key != :σ_λ_f1
    @test_throws BoundsError steadystate!(bad)
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

@testset "decompose_forecast across a model change (fewer parameters, shocks and observables)" begin
    m_new, m_old = ss104_new(Date(2026, 9, 30)), ss104_old(Date(2026, 6, 30))
    function fake_df(m)
        dates = DSGE.quarter_range(date_presample_start(m), date_conditional_end(m))
        df = DataFrame(:date => collect(dates))
        for k in keys(m.observables)
            df[!, k] = allowmissing(k == :obs_gdp ? zeros(length(dates)) : fill(NaN, length(dates)))
        end
        df
    end
    df_new, df_old = fake_df(m_new), fake_df(m_old)
    p_new, p_old = shifted_params(m_new, 0.10), shifted_params(m_old, 0.05)

    # Used to throw a BoundsError in steadystate! (parameters), then in the shockdec mapping (shocks)
    out = DSGE.decompose_forecast(m_new, m_old, df_new, df_old, p_new, p_old, :full, :full,
                                  [:obs, :pseudo]; catch_smoother_lapack = true)
    for (class, vars) in ((:obs, m_new.observables), (:pseudo, m_new.pseudo_observables))
        total = out[Symbol(:decomptotal, class)]
        @test size(total, 1) == length(vars)                              # rows are m_new's variables
        @test size(out[Symbol(:decompshockdec, class)], 3) == length(m_new.exogenous_shocks)
        # The components add up to the total: spd + release + cond + revise + model + param
        parts = sum(out[Symbol(:decomp, c, class)] for c in (:spd, :release, :cond, :revise, :model, :param))
        ok = .!isnan.(total)
        @test any(ok)
        @test parts[ok] ≈ total[ok] atol = 1e-8
    end
end
end

end # module TestDecomposeForecast
