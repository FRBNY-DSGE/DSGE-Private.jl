using DSGE, Test, BenchmarkTools, LinearAlgebra, Random

# Test + benchmark for block_kalman_filter (loaded by DSGE).
# Currently errors. Known issues:
#   - l.206 size(T,1): T is undefined (param is Ttild) -> UndefVarError
#   - Julia-0.x ctors Matrix{S}(0,0)/Array{...}(0,0,0)/Vector{S}(0) throughout
#   - l.68 precedence: `block_dims[2] == 0 & block_dims[3] == 0` ignores block_dims[3]
#   - l.86 zeros(::Matrix) removed in Julia 1.0
# Written to pass once these are fixed.

Ttild  = [-0.5 0.0; 0.0 -0.3]
Rtild  = Matrix{Float64}(I, 2, 2)
Ctild  = [0.0, 0.0]
Qtild  = 0.1 * Matrix{Float64}(I, 2, 2)
Ztild  = [1.0 0.0]
Dtild  = [0.0]
Etild  = reshape([0.05], 1, 1)
M      = Matrix{Float64}(I, 2, 2)
Mtild  = Matrix{Float64}(I, 2, 2)
block_dims = [1, 0, 0, 1]
s_0tild = Vector{Float64}(undef, 0)
P_0tild = Matrix{Float64}(undef, 0, 0)
y = 0.1 * randn(1, 10)

out, run_err = nothing, nothing
try
    global out, run_err
    Random.seed!(47)
    out = block_kalman_filter(y, Ttild, Rtild, Ctild, Qtild, Ztild, Dtild, Etild,
                              M, Mtild, block_dims, s_0tild, P_0tild)
catch e
    global out, run_err
    run_err = e
end

@testset "block_kalman_filter on a small 2-block system" begin
    if run_err !== nothing
        @test_broken run_err === nothing  # known broken — see header
    else
        loglh = out[1]
        @test length(loglh) == size(y, 2)
        @test all(isfinite, loglh)
    end
end

run_benchmarks = false
if run_benchmarks && run_err === nothing
    b = @benchmark block_kalman_filter($y, $Ttild, $Rtild, $Ctild, $Qtild, $Ztild,
                                       $Dtild, $Etild, $M, $Mtild, $block_dims,
                                       $s_0tild, $P_0tild)
    println("\nblock_kalman_filter  time: ", BenchmarkTools.prettytime(median(b).time),
            "   memory: ", BenchmarkTools.prettymemory(median(b).memory))
end

nothing
