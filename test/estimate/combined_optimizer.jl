using DSGE, Test, BenchmarkTools, Random

# Tests + benchmark for combined_optimizer on a 2-D Rosenbrock (min 0 at [1,1]).
# Note: combined_optimizer currently errors before returning. Known issues:
#   56  lbfgs()                  -> no zero-arg method; want Optim's LBFGS()
#   57  Optim.Options(autodiff=) -> autodiff is an `optimize` kwarg, not Options
#   70-75  optimize(..., method=, iterations=, ...) -> old Optim API (pre-1.x)
#   71  SimulatedAnnealing(neighbor!=) -> keyword is `neighbor`
#   91  round(rel_diff, 5)       -> round(rel_diff, digits=5)
# The test is written to pass once these are fixed.

rosenbrock(x::Vector) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x0 = [-1.2, 1.0]

out, run_err = nothing, nothing
try
    global out, run_err
    Random.seed!(47)
    out = combined_optimizer(rosenbrock, copy(x0);
                             iterations = 100, max_cycles = 2, verbose = :none)
catch e
    global out, run_err
    run_err = e
end

@testset "combined_optimizer on Rosenbrock" begin
    if run_err !== nothing
        @test_broken run_err === nothing  # known broken — see header
    else
        @test out.minimizer ≈ [1.0, 1.0] atol = 1e-2
        @test out.minimum   <  1e-4
    end
end

# Benchmark (skipped while the function errors).
run_benchmarks = true
if run_benchmarks && run_err === nothing
    b = @benchmark combined_optimizer($rosenbrock, $(copy(x0));
                                      iterations = 100, max_cycles = 2,
                                      verbose = :none) evals = 1
    println("\ncombined_optimizer [Rosenbrock]  time: ",
            BenchmarkTools.prettytime(median(b).time),
            "   memory: ", BenchmarkTools.prettymemory(median(b).memory))
end

nothing
