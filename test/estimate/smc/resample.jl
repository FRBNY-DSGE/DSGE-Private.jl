using BenchmarkTools

# Keep the resampling examples deterministic while checking distributional
# properties rather than runtime-version-specific draw sequences.
Random.seed!(42)

weights = rand(400)
weights = weights ./ sum(weights)

n_resampled = 10_000
test_sys_resample    = SMC.resample(weights, n_parts = n_resampled, method = :systematic)
test_multi_resample  = SMC.resample(weights, n_parts = n_resampled, method = :multinomial)
test_poly_resample   = SMC.resample(weights, n_parts = n_resampled, method = :polyalgo)

####################################################################

# Seeded draw sequences depend on Julia's RNG implementation. Check the
# resamplers' actual contract instead: they return valid particle indices and
# their empirical selection frequencies track the input weights.
@testset "Resampling methods" begin
    for draws in (test_sys_resample, test_multi_resample, test_poly_resample)
        @test length(draws) == n_resampled
        @test all(i -> 1 <= i <= length(weights), draws)

        frequencies = [count(==(i), draws) / n_resampled for i in eachindex(weights)]
        @test sum(abs.(frequencies .- weights)) < 0.2
    end
end

################
# Benchmarking #
################
# Flip to true to run; off by default. Pure numerics, no FRED API.
run_benchmarks = false

if run_benchmarks
    b_sys   = @benchmark SMC.resample($weights, method = :systematic)
    b_multi = @benchmark SMC.resample($weights, method = :multinomial)
    b_poly  = @benchmark SMC.resample($weights, method = :polyalgo)

    println("\n===== estimate/smc/resample benchmark results =====")
    for (name, b) in [("resample systematic  (n=400)", b_sys),
                      ("resample multinomial (n=400)", b_multi),
                      ("resample polyalgo    (n=400)", b_poly)]
        println(name, "  time:   ", BenchmarkTools.prettytime(median(b).time),
                "   memory: ", BenchmarkTools.prettymemory(median(b).memory))
    end
end

nothing
