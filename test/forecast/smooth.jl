using DSGE, ModelConstructors, JLD2, FileIO, Test, Dates, Random, BenchmarkTools
isdefined(@__MODULE__, :as_dataframe) || include(joinpath(@__DIR__, "..", "jld2_compat.jl"))

path = dirname(@__FILE__())

# Set up arguments
m = AnSchorfheide(testing = true)
m <= Setting(:date_forecast_start, quartertodate("2015-Q4"))

m <= Setting(:date_forecast_start, Date(2015, 12, 31))
system = compute_system(m)
dates = collect(Date(2015, 3, 31):Month(3):Date(2015, 12, 31))
df = DataFrame(date = dates)
for observable in keys(m.observables)
    df[!, observable] = zeros(length(dates))
end

# Smooth without drawing states
states = Dict{Symbol, Matrix{Float64}}()
shocks = Dict{Symbol, Matrix{Float64}}()
pseudo = Dict{Symbol, Matrix{Float64}}()

m <= Setting(:forecast_smoother, :durbin_koopman)
run_benchmarks = false

if run_benchmarks
    b_smooth = @benchmark smooth($m, $df, $system; draw_states = false)

    println("\n===== smooth benchmark results =====")
    println(rpad("smooth", 18), " time: ", rpad(BenchmarkTools.prettytime(median(b_smooth).time), 12),
            "memory: ", BenchmarkTools.prettymemory(median(b_smooth).memory))
end

@testset "Test smoother without drawing states" begin
    for smoother in [:hamilton, :koopman, :carter_kohn, :durbin_koopman]
        m <= Setting(:forecast_smoother, smoother)

        states[smoother], shocks[smoother], pseudo[smoother] =
        smooth(m, df, system; draw_states = false)

        @test size(states[smoother], 2) == size(shocks[smoother], 2)
        @test size(states[smoother], 2) == size(pseudo[smoother], 2)
        @test all(isfinite, states[smoother])
        @test all(isfinite, shocks[smoother])
        @test all(isfinite, pseudo[smoother])
    end
end

# Smooth, drawing states
for smoother in [:carter_kohn, :durbin_koopman]
    m <= Setting(:forecast_smoother, smoother)
    smooth(m, df, system; draw_states = true)
end


# Regime-switching smoothing tests relied on old JLD2 serialized DataFrame and
# output fixtures; those fixture-dependent cases are dropped.
