using DSGE, BenchmarkTools
path = dirname(@__FILE__)

write_test_output = false

mb_full = load("$path/../reference/MeansBands.jld2", "mb")

means, bands = meansbands_to_matrix(mb_full)

if write_test_output
    JLD2.jldopen("$path/../reference/meansbands_to_matrix_out.jld2", "w") do f
        f["means"] = means
        f["bands"] = bands
    end
end

saved_means, saved_bands = JLD2.jldopen("$path/../reference/meansbands_to_matrix_out.jld2", "r") do f
    f["means"], f["bands"]
end

nvars    = DSGE.n_vars_means(mb_full)
nperiods = DSGE.n_periods_means(mb_full)
nbands   = length(which_density_bands(mb_full))

@testset "meansbands_to_matrix" begin
    # Output shapes
    @test size(means) == (nvars, nperiods)
    @test size(bands) == (nbands, nvars, nperiods)

    # Output types
    @test eltype(means) == Float64
    @test eltype(bands) == Float64

    # Matches saved reference
    @test means == saved_means
    @test bands == saved_bands
end

run_benchmarks = false

if run_benchmarks
    b_m2m = @benchmark meansbands_to_matrix($mb_full)

    println("\n===== meansbands_to_matrix benchmark results =====")
    println(rpad("meansbands_to_matrix", 22), " time: ", rpad(BenchmarkTools.prettytime(median(b_m2m).time), 12),
            "memory: ", BenchmarkTools.prettymemory(median(b_m2m).memory))
end
