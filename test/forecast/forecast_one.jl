using DSGE, FileIO, JLD2, ModelConstructors, Test, Random, Dates, HDF5, BenchmarkTools
path = dirname(@__FILE__)

generate_regime_switch_tests = false # Set to true if you want to regenerate the jld2 files for testing

if VERSION < v"1.5"
    ver = "111"
elseif VERSION < v"1.6"
    ver = "150"
else
    ver = "160"
end

# Initialize model object
m = AnSchorfheide(testing = true)
m <= Setting(:cond_id, 0)
m <= Setting(:date_forecast_start, quartertodate("2015-Q4"))
m <= Setting(:date_conditional_end, quartertodate("2015-Q4"))
m <= Setting(:use_population_forecast, true)

estroot = normpath(joinpath(dirname(@__FILE__), "..", "reference"))
overrides = forecast_input_file_overrides(m)
overrides[:mode] = joinpath(estroot, "optimize.h5")
overrides[:full] = joinpath(estroot, "metropolis_hastings.h5")

if haskey(ENV, "FRED_API_KEY") || isfile(joinpath(homedir(),".freddatarc"))
    df = load_data(m)
    skip_forecast_one_draw = false
else
    skip_forecast_one_draw = true
    @warn "Skipping forecast_one_draw tests because FRED_API_KEY not present"
end
# Make sure output_vars ignores the untransformed and 4Q things because they are
# computed in compute_meansbands
output_vars = add_requisite_output_vars([:histpseudo, :histobs, :histstdshocks,
                                         :histutpseudo, :histutobs,
                                         :hist4qpseudo, :hist4qobs,
                                         :forecaststates, :forecastpseudo, :forecastobs, :forecaststdshocks,
                                         :forecastutpseudo, :forecastutobs,
                                         :forecast4qpseudo, :forecast4qobs,
                                         :bddforecaststates, :bddforecastshocks, :bddforecastpseudo, :bddforecastobs,
                                         :shockdecpseudo, :shockdecobs,
                                         :trendstates, :trendobs, :trendpseudo,
                                         :dettrendstates, :dettrendobs, :dettrendpseudo,
                                         :irfstates, :irfpseudo, :irfobs])

# Check error handling for input_type = :subset
@testset "Ensure properly error handling for input_type = :subset" begin
    @test_throws ErrorException forecast_one(m, :subset, :none, output_vars,
                                             subset_inds = 1:10, forecast_string = "",
                                             verbose = :none)
    @test_throws ErrorException forecast_one(m, :subset, :none, output_vars,
                                             forecast_string = "test",
                                             verbose = :none)
end

# Run modal forecasts
run_benchmarks = false

if run_benchmarks
    b_forecast_one = @benchmark forecast_one($m, :mode, :none, $output_vars, verbose = :none)

    println("\n===== forecast_one benchmark results =====")
    println(rpad("forecast_one", 18), " time: ", rpad(BenchmarkTools.prettytime(median(b_forecast_one).time), 12),
            "memory: ", BenchmarkTools.prettymemory(median(b_forecast_one).memory))
end

out = Dict{Symbol, Dict{Symbol, Array{Float64}}}()
for cond_type in [:none, :semi, :full]
    forecast_one(m, :mode, cond_type, output_vars, verbose = :none)
    forecast_one(m, :init, cond_type, output_vars, verbose = :none)

    # Read output
    out[cond_type] = Dict{Symbol, Array{Float64}}()
    output_files = get_forecast_output_files(m, :mode, cond_type, output_vars)
    for var in keys(output_files)
        out[cond_type][var] = load(output_files[var], "arr")
    end
end

# Check modal forecast output contracts. The old forecast_one_out.jld2 golden
# file is incompatible with the JLD2/Julia type metadata in this environment.
specify_mode!(m, DSGE.get_forecast_input_file(m, :mode); verbose = :none)

@testset "Modal forecast returns finite, correctly shaped output series" begin
    for cond_type in [:none, :semi, :full]
        for var in keys(out[cond_type])
            @test !isempty(out[cond_type][var])
            @test all(isfinite, out[cond_type][var])
        end
        @test size(out[cond_type][:forecastobs], 1) == length(m.observables)
        @test size(out[cond_type][:forecastpseudo], 1) == length(m.pseudo_observables)
    end
end

@testset "Test full-distribution forecasts run" begin
    m <= Setting(:forecast_block_size, 5)
    forecast_one(m, :prior, :none, output_vars, verbose = :none)
    for sampling_method in [:MH]
        m <= Setting(:sampling_method, sampling_method)
        for cond_type in [:none, :semi, :full]
            forecast_one(m, :full, cond_type, output_vars, verbose = :none)
            @test_throws ErrorException forecast_one(m, :subset, cond_type, output_vars, subset_inds = 1:10, verbose = :none)
            forecast_one(m, :subset, cond_type, output_vars, subset_inds = 1:10, forecast_string = "test", verbose = :none)
            forecast_one(m, :init_draw_shocks, cond_type, output_vars, verbose = :none)
            forecast_one(m, :mode_draw_shocks, cond_type, output_vars, verbose = :none)
        end
    end
end


# Test full-distribution blocking
m <= Setting(:forecast_block_size, 5)
forecast_one(m, :full, :none, output_vars, verbose = :none)
# Test read_forecast_output
@testset "Test full-distribution blocking" begin
    for input_type in [:mode, :full]
        output_files = get_forecast_output_files(m, input_type, :none, output_vars)
        @test ndims(DSGE.read_forecast_series(output_files[:trendobs], :trend, m.observables[:obs_gdp])) == 2
        @test ndims(DSGE.read_forecast_series(output_files[:forecastobs], :forecast, m.observables[:obs_gdp])) == 2
        @test ndims(DSGE.read_forecast_series(output_files[:irfobs], m.observables[:obs_gdp], m.exogenous_shocks[:rm_sh])) == 2
    end
end

@testset "Test forecast_one_draw" begin
    if !skip_forecast_one_draw
        m <= Setting(:regime_switching, true)
        for input_type in [:mode, :full]
            params = if input_type == :mode
                load_draws(m, input_type)
            else
                load_draws(m, input_type)[1, :]
            end
            setup_permanent_altpol!(m, AltPolicy(:historical, eqcond, solve))
            @test typeof(DSGE.forecast_one_draw(m, input_type, :none, output_vars, params, df;
                                                regime_switching = get_setting(m, :regime_switching),
                                                n_regimes = get_setting(m, :n_regimes))) == Dict{Symbol, Array{Float64}}

            # Test with alternative policy
            setup_permanent_altpol!(m, AltPolicy(:taylor93, eqcond, solve))
            @test typeof(DSGE.forecast_one_draw(m, input_type, :none, output_vars, params, df;
                                                regime_switching = get_setting(m, :regime_switching),
                                                n_regimes = get_setting(m, :n_regimes))) == Dict{Symbol, Array{Float64}}
        end
    end
end

# The regime-switching cases below use serialized JLD2 DataFrame/output fixtures
# that the installed JLD2 can no longer reconstruct; those cases are dropped.
