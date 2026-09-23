using BenchmarkTools
path = dirname(@__FILE__)
fred = CSV.read("$path/../reference/fred_160812.csv", DataFrame)
custom_settings = [Setting(:data_vintage, "160812"),
                   Setting(:cond_vintage, "160812"),
                   Setting(:cond_id, 0),
                   Setting(:use_population_forecast, true),
                   Setting(:date_forecast_start, DSGE.quartertodate("2016-Q3")),
                   Setting(:date_forecast_start, DSGE.quartertodate("2016-Q3")),
                   Setting(:n_mon_anticipated_shocks, 6)]

m = Model990(custom_settings = custom_settings, testing = true)
m <= Setting(:rate_expectations_source, :ois)
install_fred_test_cache!(m)


@testset "Check FRED data is properly loaded" begin
    loaded = sort(load_fred_data(m; end_date = date_mainsample_end(m), verbose = :none), :date)
    @test @test_matrix_approx_eq Matrix(fred[:,2:end]) Matrix(loaded[:,2:end])
end
