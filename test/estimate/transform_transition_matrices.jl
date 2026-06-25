using DSGE, ModelConstructors, Test, BenchmarkTools

# Tests + benchmark for src/estimate/transform_transition_matrices.jl

freq = 2
m = AnSchorfheide(testing = true)
m <= Setting(:state_simulation_freq, freq)

TT = [0.5 0.0; 0.0 0.4]
R  = reshape([1.0, 1.0], 2, 1)
C  = [0.0, 0.0]
n, ne = size(TT, 1), size(R, 2)

@testset "transform_transition_matrices (track_lag=false)" begin
    TTT, RRR, CCC = transform_transition_matrices(m, TT, R, C; track_lag = false)

    @test size(TTT) == (n * freq, n * freq)
    @test size(RRR) == (size(R, 1) * freq, ne * freq)
    @test length(CCC) == length(C) * freq

    # TTT carries powers of TT in the right-hand column blocks
    @test TTT[1:2, 3:4] == TT
    @test TTT[3:4, 3:4] == TT * TT
    @test all(TTT[:, 1:2] .== 0)

    # RRR carries T^i * R blocks
    @test RRR[1:2, 1:1] == R
    @test RRR[3:4, 1:1] == TT * R
    @test RRR[3:4, 2:2] == R

    @test CCC == zeros(length(C) * freq)
end

#-----------------------------------------------------------------
# track_lag=true (default) — broken
#-----------------------------------------------------------------
out, run_err = nothing, nothing
try
    global out, run_err
    out = transform_transition_matrices(m, TT, R, C; track_lag = true)
catch e
    global out, run_err
    run_err = e
end

@testset "transform_transition_matrices (track_lag=true)" begin
    @test_broken run_err === nothing  # eye(TT) line 46, zeros(C) line 74 — see header
end

################
# Benchmarking #
################
run_benchmarks = false
if run_benchmarks
    b = @benchmark transform_transition_matrices($m, $TT, $R, $C; track_lag = false)
    println("\ntransform_transition_matrices (track_lag=false)  time: ",
            BenchmarkTools.prettytime(median(b).time),
            "   memory: ", BenchmarkTools.prettymemory(median(b).memory))
end

nothing
