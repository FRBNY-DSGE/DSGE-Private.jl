using DSGE, ModelConstructors, Test, Dates, DataFrames, OrderedCollections, CSV

@testset "Reported dormant crashes" begin
    m = Model1002("ss207"; testing=true,
                  custom_settings=[Setting(:date_forecast_start, Date(2026, 12, 31))])
    @test get_setting(m, :pgap_type) == :flexible_ait
    @test get_setting(m, :ygap_type) == :flexible_ait
    @test !isempty(DSGE.parameter_groupings(m))
    @test size(eqcond(m)[1], 1) > 0

    dates = [Date(2020, 3, 31), Date(2020, 6, 30)]
    function mb(date, value)
        metadata = Dict{Symbol,Any}(:product => :forecast, :class => :obs,
            :cond_type => :none, :para => :mode, :forecast_string => "old",
            :date_inds => OrderedDict(date => 1), :indices => Dict(:date => 1, :x => 2))
        MeansBands(metadata, DataFrame(date=[date], x=[value]),
                   Dict(:x => DataFrame(date=[date], lb=[value], ub=[value])))
    end
    combined = cat(mb(dates[1], 1.0), mb(dates[2], 2.0); forecast_string="new")
    @test collect(keys(combined.metadata[:date_inds])) == dates
    @test collect(values(combined.metadata[:date_inds])) == [1, 2]
    @test combined.metadata[:forecast_string] == "new"

    scen_model = AnSchorfheide(testing=true)
    scen_model <= Setting(:forecast_horizons, 2)
    scen = Scenario(:empty, "Empty", Symbol[], Symbol[], "REF")
    targets = DSGE.load_scenario_targets!(scen_model, scen, 1)
    @test size(targets, 1) == 2
    @test all(all(isnan, targets[!, name]) for name in propertynames(targets))
end

@testset "load_data_levels add_vals" begin
    m = Model990(custom_settings=[Setting(:data_vintage, "160812"),
        Setting(:cond_vintage, "160812"), Setting(:cond_id, 0),
        Setting(:use_population_forecast, true),
        Setting(:date_forecast_start, DSGE.quartertodate("2016-Q3")),
        Setting(:date_conditional_end, DSGE.quartertodate("2016-Q3")),
        Setting(:n_mon_anticipated_shocks, 6)], testing=true)
    m <= Setting(:rate_expectations_source, :ois)
    install_fred_test_cache!(m)
    date = Date(2016, 6, 30)
    cond_file = joinpath(dataroot(m), "additional.csv")
    CSV.write(cond_file, DataFrame(date=[date], GDP=[12345.0]))
    m <= Setting(:cond_filename, cond_file)
    levels = DSGE.load_data_levels(m; verbose=:none, add_vals=(true, date))
    @test levels[findfirst(==(date), levels[!, :date]), :GDP] == 12345.0
end
