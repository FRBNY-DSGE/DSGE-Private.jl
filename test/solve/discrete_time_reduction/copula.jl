using Test, DSGE, JLD2, FileIO, Interpolations, BenchmarkTools, Random

do_time = false
generate_output = false
bl_cdf_ss, bl_cdf_m, bl_cdf_k, bl_cdf_y, bl_μ =
    JLD2.jldopen(joinpath(dirname(@__FILE__), "../../reference/bayer_luetticke_copula.jld2"), "r") do file
        read(file, "CDF_SS"), read(file, "CDF_m"), read(file, "CDF_k"), read(file, "CDF_y"), read(file, "distrSS")
    end

mycdf = DSGE.cdf_quadrature(bl_μ)
cdf_m, cdf_k, cdf_y = DSGE.marginal_cdf_quadrature(bl_μ)
weights = fill(.5, size(bl_μ)) # ensure quadrature works
half_mycdf = DSGE.cdf_quadrature(bl_μ, weights)
half_cdf_m, half_cdf_k, half_cdf_y = DSGE.marginal_cdf_quadrature(bl_μ, weights)

@testset "CDF calculations from PDF distribution" begin
    @test mycdf ≈ bl_cdf_ss
    @test cdf_m ≈ bl_cdf_m
    @test cdf_k ≈ bl_cdf_k
    @test cdf_y ≈ bl_cdf_y

    @test half_mycdf ≈ bl_cdf_ss ./ 2.
    @test half_cdf_m ≈ bl_cdf_m ./ 2.
    @test half_cdf_k ≈ bl_cdf_k ./ 2.
    @test half_cdf_y ≈ bl_cdf_y ./ 2.
end

mycopula = DSGE.copula(bl_μ)
if generate_output
    # When re-generating output, you may also want to compare against the Copula constructed from
    # Bayer and Luetticke (2020) in Quantitative Economics
    Random.seed!(1793)
    xin = rand(10, 3)
    exp_output = mycopula(xin[:, 1], xin[:, 2], xin[:, 3])
    JLD2.jldopen(joinpath(dirname(@__FILE__), "../../reference/copula_output.jld2"), true, true, true, IOStream) do file
        write(file, "xin", xin)
        write(file, "exp_output", exp_output)
    end
else
    xin, exp_output = JLD2.jldopen(joinpath(dirname(@__FILE__), "../../reference/copula_output.jld2"), "r") do file
        read(file, "xin"), read(file, "exp_output")
    end

    @testset "Linear interpolation to construct Copula" begin
        @test mycopula(xin[:, 1], xin[:, 2], xin[:, 3]) ≈ exp_output
        @test mycopula(cdf_m, cdf_k, cdf_y) ≈ mycdf # Ensure interpolant returns interpolation points
    end

end

if do_time
    Random.seed!(1793)
    xin = rand(100, 3)

    # Implementation by Bayer and Luetticke (2020) in Quantitative Economics is roughly twice as fast but
    # our implementation allows μ to take any dimensionality. The difference in speed results from Interpolations.jl
    # using roughly twice as many allocations as the implementation by Bayer and Luetticke.
    @btime mycopula(xin[:, 1], xin[:, 2], xin[:, 3])
end
