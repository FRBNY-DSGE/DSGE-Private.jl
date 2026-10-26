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

# Fake FRED client: each call pops the next scripted outcome (an error message or :ok)
const FredData = DSGE.FredData   # the module DSGE calls get_data from
mutable struct ScriptedFred
    outcomes::Vector{Any}
    calls::Vector{Any}
end
function FredData.get_data(f::ScriptedFred, s::AbstractString; kwargs...)
    push!(f.calls, (s, haskey(kwargs, :vintage_dates)))
    o = popfirst!(f.outcomes)
    o == :ok ? (s, haskey(kwargs, :vintage_dates)) : error(o)
end

@testset "FRED pulls fail loudly instead of caching wrong-vintage data" begin
    d0, d1, v = Date(2000, 1, 1), Date(2001, 12, 31), Date(2002, 1, 10)
    keyerr = "HTTP 500 /fred/series/observations?api_key=abcdef0123456789&series_id=GDP"

    # A transient failure is retried; the vintaged series is returned
    f = ScriptedFred(Any[keyerr, :ok], [])
    @test DSGE.fetch_fred_vintage(f, :GDP, d0, d1, v; waits = [0, 0], verbose = :none) == ("GDP", true)
    @test length(f.calls) == 2

    # Persistent failure: an error (no unvintaged substitute), with the API key redacted
    f = ScriptedFred(Any[keyerr, keyerr, keyerr], [])
    err = try DSGE.fetch_fred_vintage(f, :GDP, d0, d1, v; waits = [0, 0], verbose = :none); "" catch e; e.msg end
    @test occursin("could not be fetched at vintage", err)
    @test occursin("api_key=<redacted>", err) && !occursin("abcdef0123456789", err)
    @test all(last, f.calls)                      # every call asked for the vintage

    # ALFRED keeps no vintage of the series: today's series, with a warning, no retries
    nov = "Bad Request.  The series does not exist in ALFRED but may exist in FRED."
    f = ScriptedFred(Any[nov, :ok], [])
    r = @test_logs (:warn, r"not real-time") DSGE.fetch_fred_vintage(f, :COMPNFB, d0, d1, v; waits = [0, 0], verbose = :none)
    @test r == ("COMPNFB", false)
    @test length(f.calls) == 2

    # Opt-in fallback for old-vintage reruns
    f = ScriptedFred(Any[keyerr, keyerr, keyerr, :ok], [])
    r = @test_logs (:warn, r"not real-time") DSGE.fetch_fred_vintage(f, :GDP, d0, d1, v; waits = [0, 0],
                                                                    allow_unvintaged = true, verbose = :none)
    @test r == ("GDP", false)

    # The setting is read from a testing model too
    @test DSGE.get_setting_or(m, :fred_allow_unvintaged_fallback, false) == false
    m2 = deepcopy(m); m2 <= Setting(:fred_allow_unvintaged_fallback, true)
    @test DSGE.get_setting_or(m2, :fred_allow_unvintaged_fallback, false) == true
end
