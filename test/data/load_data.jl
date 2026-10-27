using DSGE
using Test, DataFrames, HDF5, JLD2, BenchmarkTools

path = dirname(@__FILE__)

@testset "Test various forms of data loading" begin
        # Specify vintage and dates
        global custom_settings = [Setting(:data_vintage, "160812"),
                                  Setting(:cond_vintage, "160812"),
                                  Setting(:cond_id, 0),
                                  Setting(:use_population_forecast, true),
                                  Setting(:date_forecast_start, DSGE.quartertodate("2016-Q3")),
                                  Setting(:date_conditional_end, DSGE.quartertodate("2016-Q3")),
                                  Setting(:n_mon_anticipated_shocks, 6)]

        global m = Model990(custom_settings = custom_settings, testing = true)
        m <= Setting(:rate_expectations_source, :ois)
        install_fred_test_cache!(m)

        # Read expected results
        exp_data, exp_cond_data, exp_semicond_data =
        JLD2.jldopen("$path/../reference/load_data_out.jld2", "r") do file
            read(file, "data"), read(file, "cond_data"), read(file, "semi_cond_data")
        end

        # Check high summary statistics runs without an error
        @info "The following summary statistics are expected."
        load_data(m; try_disk = false, summary_statistics = :high, verbose=:none,
                  check_empty_columns = false)
        load_data(m; try_disk = false, summary_statistics = :low, verbose=:none,
                  check_empty_columns = false)

        # Unconditional data
        println("The following warnings are expected test behavior:")
        global df = load_data(m; try_disk=false, verbose=:none, summary_statistics=:none,
                              check_empty_columns = false)
        global data = df_to_matrix(m, df)
        @test @test_matrix_approx_eq exp_data data

        # Conditional data
        cond_df = load_data(m; cond_type=:full, try_disk=false, verbose=:none,
                            summary_statistics=:none,
                            check_empty_columns = false)
        cond_data = df_to_matrix(m, cond_df; cond_type=:full)
        @test @test_matrix_approx_eq exp_cond_data cond_data

        # Semiconditional data
        semicond_df = load_data(m; cond_type=:semi, try_disk=false, verbose=:none,
                                summary_statistics=:none, check_empty_columns = false)
        semicond_data = df_to_matrix(m, semicond_df; cond_type=:semi)
        @test @test_matrix_approx_eq exp_semicond_data semicond_data

        # Test errors
        m <= Setting(:data_vintage, "160813")
        m <= Setting(:cond_vintage, "160812")
        @test_throws ErrorException load_cond_data_levels(m)
        m <= Setting(:cond_vintage, "160813")
        @test_throws ErrorException load_cond_data_levels(m)
        @test_throws ErrorException load_data(m; try_disk = false, verbose=:none)
end

@testset "check_data flags series that start late, end early or have gaps" begin
    m = Model1002("ss10", testing = true,
                  custom_settings = [Setting(:date_forecast_start, DSGE.quartertodate("2016-Q4")),
                                     Setting(:date_conditional_end, DSGE.quartertodate("2016-Q4"))])
    dates = DSGE.quarter_range(date_presample_start(m), date_mainsample_end(m))
    obs   = intersect(DSGE.CORE_OBSERVABLES, collect(keys(m.observable_mappings)))
    clean = DataFrame(date = dates)
    for col in obs
        clean[!, col] = fill(1.0, length(dates))
    end
    # Hours and the long rate start late by design
    clean[dates .< Date(1964, 3, 31), :obs_hours]      .= NaN
    :obs_longrate in obs && (clean[dates .< Date(1971, 9, 30), :obs_longrate] .= NaN)
    @test all(c in obs for c in (:obs_gdp, :obs_hours, :obs_gdpdeflator, :obs_spread))
    @test check_data(m, clean)

    late(col, d) = (df = copy(clean); df[dates .< d, col] .= NaN; df)
    # A spread that starts years late (realtime ois_vint=230410 started in 2016Q4) is caught
    @test_throws ErrorException check_data(m, late(:obs_spread, Date(2010, 12, 31)))
    err = try check_data(m, late(:obs_spread, Date(2010, 12, 31))); "" catch e; e.msg end
    @test occursin("obs_spread starts 2010-12-31", err)
    # A series starting in 1981 (realtime bluechip_vint=040110 consumption) is caught
    @test_throws ErrorException check_data(m, late(:obs_investment in obs ? :obs_investment : :obs_gdp, Date(1981, 6, 30)))
    # Within max_lead_quarters is allowed (old vintages start core PCE one quarter late)
    @test check_data(m, late(:obs_gdp, Date(1959, 12, 31)))
    @test_throws ErrorException check_data(m, late(:obs_gdp, Date(1959, 12, 31)); max_lead_quarters = 0)
    # Hours starting after its own expected start is caught
    @test_throws ErrorException check_data(m, late(:obs_hours, Date(1970, 3, 31)))
    # Models without :observable_start_dates (e.g. m990, whose long rate starts in 1990)
    # skip the late-start check
    m_nostart = deepcopy(m)
    delete!(m_nostart.settings, :observable_start_dates); delete!(m_nostart.test_settings, :observable_start_dates)
    @test check_data(m_nostart, late(:obs_spread, Date(2010, 12, 31)))

    # Cached datasets (load_data's try_disk path) go through passes_check_data: a bad
    # file is reported and recreated instead of used, unless :allow_data_gaps is set
    @test DSGE.passes_check_data(m, clean)
    ok = @test_logs (:warn, r"fails the data check") DSGE.passes_check_data(m, late(:obs_spread, Date(2010, 12, 31)))
    @test ok == false
    m_gaps = deepcopy(m)
    m_gaps <= Setting(:allow_data_gaps, true)
    @test DSGE.allow_data_gaps(m_gaps)        # testing models keep it in test_settings
    @test DSGE.passes_check_data(m_gaps, late(:obs_spread, Date(2010, 12, 31)))

    # The existing checks still fire
    gap = copy(clean); gap[dates .== Date(2000, 3, 31), :obs_gdp] .= NaN
    @test_throws ErrorException check_data(m, gap)
    short = copy(clean); short[dates .> Date(2015, 12, 31), :obs_gdpdeflator] .= NaN
    @test_throws ErrorException check_data(m, short)
    empty = copy(clean); empty[!, :obs_spread] .= NaN
    @test_throws ErrorException check_data(m, empty)
end

nothing
