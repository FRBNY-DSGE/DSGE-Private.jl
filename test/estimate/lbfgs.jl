using DSGE, Test, BenchmarkTools, Random

rosenbrock(x::Vector) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x0 = [-1.2, 1.0]

Random.seed!(47)
out, iteration_times, posterior_ls, x_trace = lbfgs(rosenbrock, copy(x0); iterations = 1000)

@testset "lbfgs on Rosenbrock" begin
    @test out.minimizer ≈ [1.0, 1.0] atol = 1e-3
    @test out.minimum   <  1e-6
    @test !isempty(iteration_times)
    @test !isempty(posterior_ls)
    @test !isempty(x_trace)
    @test last(x_trace) ≈ out.minimizer atol = 1e-8
end

run_benchmarks = false
if run_benchmarks
    b = @benchmark lbfgs($rosenbrock, $(copy(x0)); iterations = 1000)
    println("\nlbfgs [Rosenbrock]  time: ", BenchmarkTools.prettytime(median(b).time),
            "   memory: ", BenchmarkTools.prettymemory(median(b).memory))
end

nothing
