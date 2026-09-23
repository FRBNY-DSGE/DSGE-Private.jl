##
## hetfsysblocks_test.jl
##
## Run this after hetfsysblocks.jl (same Julia session, so F, c_a_star, MUnext,
## AProb, etc. are already in scope).
##
## What it checks:
##   1. Every F residual is ≈ 0  (steady-state correctness)
##   2. Intermediate het-block outputs match MATLAB's fsys.mat expected values
##      (only valid when Xt=Xt1=Yt=Yt1=0, i.e. the baseline Jacobian evaluation)
##

using Test
using MAT
using HDF5
using Printf

# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------
function open_matfile(filepath)
    try; return matopen(filepath)
    catch e
        occursin("zlib", string(e)) || occursin("incorrect header", string(e)) ?
            h5open(filepath, "r") : rethrow(e)
    end
end

function to_matrix(x)
    isa(x, Vector{<:AbstractVector}) ? reduce(hcat, x) : x
end

function to_cell_of_matrices(x)
    isa(x, Vector) ? [to_matrix(e) for e in x] : x
end

function clean_arrays!(d::Dict)
    for (k, v) in d
        if isa(v, Vector{<:AbstractVector}); d[k] = to_matrix(v)
        elseif isa(v, AbstractMatrix) && size(v,1)==1; d[k] = vec(v)
        elseif isa(v, Dict); clean_arrays!(v)
        end
    end
    d
end

tol = 1e-8

# -----------------------------------------------------------------------
# 1. Steady-state residuals must all be zero
# -----------------------------------------------------------------------
println("=" ^ 60)
println("TEST 1: SS residuals (all F entries should be ≈ 0)")
println("=" ^ 60)

@testset "Steady-state F residuals" begin
    for (key, val) in F
        mx = maximum(abs.(vec(val)))
        pass = mx < tol
        @printf("  %s %-25s: max|F| = %10.3e  %s\n",
                pass ? "✓" : "✗", string(key), mx, pass ? "" : "  ← FAIL")
        @test pass
    end
end

# -----------------------------------------------------------------------
# 2. Compare het-block intermediates against MATLAB fsys.mat expected outputs
#    (only meaningful if Xt=Xt1=Yt=Yt1=0, i.e. baseline Jacobian evaluation)
# -----------------------------------------------------------------------
output_path = joinpath(@__DIR__, "../../../original/outputs/fsys.mat")

if !isfile(output_path)
    @warn "Skipping MATLAB comparison: output file not found at $output_path"
else
    println("\n", "=" ^ 60)
    println("TEST 2: Intermediates vs MATLAB (fsys.mat baseline evaluation)")
    println("=" ^ 60)

    out = open_matfile(output_path)
    exp_c_a_star         = read(out, "c_a_star")
    exp_b_a_star         = read(out, "b_a_star")
    exp_a_a_star         = read(out, "a_a_star")
    exp_c_n_star         = read(out, "c_n_star")
    exp_b_n_star         = read(out, "b_n_star")
    exp_AProb            = read(out, "AProb")
    exp_AC               = vec(read(out, "AC"))
    exp_MUnext           = read(out, "MUnext")
    exp_P_transition_exp = read(out, "P_transition_exp")
    exp_P_dist           = read(out, "P_transition_dist")
    close(out)

    function cmp(name, julia_val, matlab_val)
        jv = vec(Float64.(julia_val))
        mv = vec(Float64.(matlab_val))
        mx_abs = maximum(abs.(jv .- mv))
        mx_rel = maximum(abs.(jv .- mv) ./ max.(abs.(mv), 1e-15))
        pass = mx_abs < tol
        @printf("  %s %-25s: max|diff| = %10.3e   max rel = %10.3e\n",
                pass ? "✓" : "✗", name, mx_abs, mx_rel)
        return pass
    end

    @testset "Het-block outputs vs MATLAB" begin
        @testset "Policy functions" begin
            @test cmp("c_a_star",  c_a_star, exp_c_a_star)
            @test cmp("b_a_star",  b_a_star, exp_b_a_star)
            @test cmp("a_a_star",  a_a_star, exp_a_a_star)
            @test cmp("c_n_star",  c_n_star, exp_c_n_star)
            @test cmp("b_n_star",  b_n_star, exp_b_n_star)
        end
        @testset "Adjustment decision" begin
            @test cmp("AProb", AProb, exp_AProb)
            @test cmp("AC",    AC,    exp_AC)
        end
        @testset "Transition matrices" begin
            @test cmp("P_transition_exp",  P_transition_exp,  exp_P_transition_exp)
            @test cmp("P_transition_dist", P_transition_dist, exp_P_dist)
        end
        @testset "Distribution" begin
            @test cmp("MUnext", MUnext, exp_MUnext)
        end
    end
end

println("\n", "=" ^ 60)
println("Done.")
println("=" ^ 60)
