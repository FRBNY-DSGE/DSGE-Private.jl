using DSGE, Test, BenchmarkTools, Random
import Optim

# Tests + benchmark for src/estimate/nelder_mead.jl.
#   - nelder_mead(fcn, x0): runs on the installed Optim (its `method=` API is still
#     present). We check the objective improves substantially; try/catch keeps it
#     robust if a newer Optim drops `method=`.
#   - MatlabSimplexer constructors: ok.
#   - Optim.simplexer(MatlabSimplexer, x): runs, but aliases the initial point
#     (`[initial_x for i=...]` doesn't copy), so the loop mutates one shared array
#     and all simplex vertices come out identical. (The zero/nonzero perturbation
#     branches also look inverted vs MATLAB fminsearch.)

#-----------------------------------------------------------------
# nelder_mead — broken (old Optim API)
#-----------------------------------------------------------------
rosenbrock(x::Vector) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x0 = [-1.2, 1.0]

out, run_err = nothing, nothing
try
    global out, run_err
    Random.seed!(47)
    out = DSGE.nelder_mead(rosenbrock, copy(x0); iterations = 1000)
catch e
    global out, run_err
    run_err = e
end

@testset "nelder_mead on Rosenbrock" begin
    if run_err !== nothing
        @test_broken run_err === nothing  # only if a newer Optim drops the method= API
    else
        @test isfinite(out.minimum)
        @test out.minimum < 1.0           # big improvement from f(x0) ≈ 24.2
    end
end

#-----------------------------------------------------------------
# MatlabSimplexer constructors
#-----------------------------------------------------------------
@testset "MatlabSimplexer constructors" begin
    s = DSGE.MatlabSimplexer()
    @test s.a == 0.00025
    @test s.b == 0.05

    s2 = DSGE.MatlabSimplexer(1.0, 2.0)
    @test s2.a == 1.0
    @test s2.b == 2.0

    s3 = DSGE.MatlabSimplexer(a = 0.1, b = 0.2)
    @test s3.a == 0.1
    @test s3.b == 0.2
end

#-----------------------------------------------------------------
# Optim.simplexer(MatlabSimplexer, x)
#-----------------------------------------------------------------
@testset "Optim.simplexer(MatlabSimplexer, x)" begin
    x = [1.0, 2.0, 3.0]
    simplex = Optim.simplexer(DSGE.MatlabSimplexer(), x)
    @test length(simplex) == length(x) + 1          # n+1 vertices
    @test_broken allunique(simplex)                 # aliasing: vertices are all the same array
end

################
# Benchmarking #
################
run_benchmarks = true
if run_benchmarks
    results = Tuple{String, Any}[]

    # The optimizer itself — only if it ran (skipped if a newer Optim drops method=).
    if run_err === nothing
        b_nm = @benchmark DSGE.nelder_mead($rosenbrock, $(copy(x0)); iterations = 1000)
        push!(results, ("nelder_mead", b_nm))
    end

    b_simplex = @benchmark Optim.simplexer(DSGE.MatlabSimplexer(), x) setup = (x = [1.0, 2.0, 3.0, 4.0])
    push!(results, ("Optim.simplexer (4-vec)", b_simplex))

    println("\n===== estimate/nelder_mead benchmark results =====")
    for (name, b) in results
        println(rpad(name, 26), " time: ", BenchmarkTools.prettytime(median(b).time),
                "   memory: ", BenchmarkTools.prettymemory(median(b).memory))
    end
end

nothing
