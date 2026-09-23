using DSGE, DataFrames, Dates, LinearAlgebra, Test, BenchmarkTools

# The old forecast_args.jld2/filter_out.jld2 pair cannot be reconstructed by
# current JLD2 because it contains obsolete serialized Julia type metadata. Use
# a small local observation panel and check the filter's output contract.
m = AnSchorfheide(testing = true)
m <= Setting(:date_forecast_start, Date(2015, 12, 31))
system = compute_system(m)

dates = collect(Date(2015, 3, 31):Month(3):Date(2015, 12, 31))
df = DataFrame(date = dates)
for observable in keys(m.observables)
    df[!, observable] = zeros(length(dates))
end

n_filter_states = size(system[:TTT], 1)
z0 = zeros(n_filter_states)
P0 = Matrix{Float64}(I, n_filter_states, n_filter_states)

@testset "Kalman filter on a local observation panel" begin
    for kal in (DSGE.filter(m, df, system), DSGE.filter(m, df, system, z0, P0))
        for field in fieldnames(typeof(kal))
            value = kal[field]
            value isa Number && @test isfinite(value)
            value isa AbstractArray && @test all(isfinite, value)
        end
    end
end

run_benchmarks = false
if run_benchmarks
    @benchmark DSGE.filter($m, $df, $system)
end
