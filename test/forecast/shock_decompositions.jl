using DSGE, FileIO, JLD2, ModelConstructors, Test, Random, Dates, BenchmarkTools
path = dirname(@__FILE__)
isdefined(@__MODULE__, :as_dataframe) || include(joinpath(@__DIR__, "..", "jld2_compat.jl"))

# Set up arguments
m = AnSchorfheide(testing = true)
m <= Setting(:date_forecast_start, quartertodate("2015-Q4"))
m <= Setting(:forecast_horizons, 1)

system = compute_system(m)
# The configured decomposition end date maps to index 225 including its
# forecast period.
histshocks = zeros(size(system[:RRR], 2), 224)

# With shockdec_startdate not null
states, obs, pseudo = shock_decompositions(m, system, histshocks)

run_benchmarks = false

if run_benchmarks
    b_shockdec = @benchmark shock_decompositions($m, $system, $histshocks)

    println("\n===== shock_decompositions benchmark results =====")
    println(rpad("shock_decompositions", 22), " time: ", rpad(BenchmarkTools.prettytime(median(b_shockdec).time), 12),
            "memory: ", BenchmarkTools.prettymemory(median(b_shockdec).memory))
end

@testset "Test shockdec with non-null startdate" begin
    @test size(states, 2) == size(histshocks, 2)
    @test size(obs, 2) == size(histshocks, 2)
    @test size(pseudo, 2) == size(histshocks, 2)
    @test all(isfinite, states)
    @test all(isfinite, obs)
    @test all(isfinite, pseudo)
end

# With shockdec_startdate null
#m <= Setting(:shockdec_startdate, Nullable{Date}())
m <= Setting(:shockdec_startdate, nothing)
states, obs, pseudo = shock_decompositions(m, system, histshocks)

@testset "Test shockdec with null startdate" begin
    @test size(states, 2) == size(histshocks, 2)
    @test size(obs, 2) == size(histshocks, 2)
    @test size(pseudo, 2) == size(histshocks, 2)
    @test all(isfinite, states)
    @test all(isfinite, obs)
    @test all(isfinite, pseudo)
end

@testset "Deterministic trends" begin
    dettrend = DSGE.deterministic_trends(m, system, zeros(8))
    @test @test_matrix_approx_eq dettrend[1] zeros(size(dettrend[1]))
    @test @test_matrix_approx_eq dettrend[2] zeros(size(dettrend[2]))
    @test @test_matrix_approx_eq dettrend[3] zeros(size(dettrend[3]))
end

@testset "Trends" begin
    out = DSGE.trends(system)
    @test @test_matrix_approx_eq out[1] system[:CCC]
    @test @test_matrix_approx_eq out[2] system[:ZZ] * system[:CCC] + system[:DD]
    @test @test_matrix_approx_eq out[3] system[:ZZ_pseudo] * system[:CCC] + system[:DD_pseudo]
end

# Check trends for "fake" regime-switching system
reg_sys = RegimeSwitchingSystem(System{Float64}[system, system])
@testset "Regime-switching trends" begin
    out = DSGE.trends(reg_sys)
    for i in 1:2
        @test @test_matrix_approx_eq out[1][:, i] reg_sys[i, :CCC]
        @test @test_matrix_approx_eq out[2][:, i] reg_sys[i, :ZZ] * reg_sys[i, :CCC] + reg_sys[i, :DD]
        @test @test_matrix_approx_eq out[3][:, i] reg_sys[i, :ZZ_pseudo] * reg_sys[i, :CCC] + reg_sys[i, :DD_pseudo]
    end
end

m.settings[:regime_dates] = Setting(:regime_dates, Dict{Int, Date}(1 => date_presample_start(m), 2 => Date(2010, 3, 31)))
m.test_settings[:regime_dates] = Setting(:regime_dates, Dict{Int, Date}(1 => date_presample_start(m), 2 => Date(2010, 3, 31)))
m.settings[:regime_switching] = Setting(:regime_switching, true)
m.test_settings[:regime_switching] = Setting(:regime_switching, true)
setup_regime_switching_inds!(m)
@testset "Regime-switching deterministic trends" begin
    dettrend = DSGE.deterministic_trends(m, reg_sys, zeros(8))
    @test @test_matrix_approx_eq dettrend[1] zeros(size(dettrend[1]))
    @test @test_matrix_approx_eq dettrend[2] zeros(size(dettrend[2]))
    @test @test_matrix_approx_eq dettrend[3] zeros(size(dettrend[3]))
end

# Check shock decompositions for regime-switching system
out_shockdec1 = shock_decompositions(m, reg_sys, histshocks,
                                    date_presample_start(m), date_forecast_start(m)) # check we only return the shockdecs for requested period
out_shockdec2 = shock_decompositions(m, reg_sys, histshocks[:, 1:end - 1]) # check default

@testset "Shock decompositions with regime-switching" begin
    @test @test_matrix_approx_eq out_shockdec1[1] states
    @test @test_matrix_approx_eq out_shockdec1[2] obs
    @test @test_matrix_approx_eq out_shockdec1[3] pseudo
    @test @test_matrix_approx_eq out_shockdec2[1] states
    @test @test_matrix_approx_eq out_shockdec2[2] obs
    @test @test_matrix_approx_eq out_shockdec2[3] pseudo
end

# The time-varying CCC workflow below depends on the obsolete serialized regime
# DataFrame fixture; its JLD2 type metadata is not readable by supported JLD2.
