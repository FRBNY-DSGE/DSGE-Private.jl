using Test
using JLD2
using DSGE

# This intentionally ignores any param/grid saved in the reference file.
# The solve always runs with the current model-derived objects.
const DEFAULT_REFERENCE = joinpath(@__DIR__, "..", "data", "steadystate_reference.jld2")
const DEFAULT_KEYS = [
    "Output", "R_fc", "H_fc", "W_fc", "r_k", "mc", "v", "u", "V", "M", "f",
    "Profits_fc", "THETA", "NWb", "Profit_FI", "Ab", "Bb", "Cb", "n",
    "c_n_guess", "b_n_star", "c_a_guess", "b_a_star", "a_a_star", "psi_guess",
    "mu_dist", "AProb", "Value", "mutil_c", "Vb", "Va",
]
const DEFAULT_GRID_OUTPUT_KEYS = ["K", "L", "N"]

_stringkeydict(d::AbstractDict) = Dict(string(k) => v for (k, v) in pairs(d))

function build_test_model(subspec::AbstractString)
    try
        return DSGE.mBBQ(subspec; load_steadystate = true)
    catch
        @warn "Could not load a saved steady state. Falling back to a fresh solve for test setup."
        return DSGE.mBBQ(subspec; load_steadystate = false)
    end
end

function load_reference_results(path::AbstractString)
    ref = Dict{String, Any}()
    JLD2.jldopen(path, "r") do file
        if haskey(file, "results")
            results = file["results"]
            results isa AbstractDict || error("Expected `results` in $path to be a Dict-like object.")
            merge!(ref, _stringkeydict(results))
        else
            for key in keys(file)
                ref[string(key)] = file[key]
            end
        end
    end
    return ref
end

function compare_value(name::AbstractString, actual, expected; atol::Real = 1e-8, rtol::Real = 1e-6)
    if actual isa AbstractArray || expected isa AbstractArray
        actual_arr = Array(actual)
        expected_arr = Array(expected)
        @test size(actual_arr) == size(expected_arr)
        @test isapprox(actual_arr, expected_arr; atol = atol, rtol = rtol)
    else
        @test isapprox(Float64(actual), Float64(expected); atol = atol, rtol = rtol)
    end
    return nothing
end

function run_steadystate_reference_test(;
    ref_path::AbstractString = get(ENV, "MBBQ_SS_REFERENCE", DEFAULT_REFERENCE),
    subspec::AbstractString = "ss1",
    compare_keys::Vector{String} = copy(DEFAULT_KEYS),
    grid_output_keys::Vector{String} = copy(DEFAULT_GRID_OUTPUT_KEYS),
    atol::Real = 1e-8,
    rtol::Real = 1e-6,
    m::Union{Nothing, DSGE.mBBQ} = nothing,
)
    isfile(ref_path) || error("Reference JLD2 not found at $ref_path")

    model = isnothing(m) ? build_test_model(subspec) : m

    current_param = DSGE._mbbq_ss_param_dict(model)
    current_grid = DSGE._mbbq_ss_grid_dict(model)
    current = DSGE.ss_ump_new(current_param, current_grid)
    reference = load_reference_results(ref_path)

    @testset "mBBQ steady state vs reference" begin
        @testset "Top-level outputs" begin
            for key in compare_keys
                @test haskey(reference, key)
                @test haskey(current, key)
                compare_value(key, current[key], reference[key]; atol = atol, rtol = rtol)
            end
        end

        @testset "Grid outputs" begin
            ref_grid = haskey(reference, "grid") && reference["grid"] isa AbstractDict ?
                _stringkeydict(reference["grid"]) : Dict{String, Any}()
            for key in grid_output_keys
                @test haskey(ref_grid, key)
                @test haskey(current["grid"], key)
                compare_value("grid.$key", current["grid"][key], ref_grid[key]; atol = atol, rtol = rtol)
            end
        end
    end

    return current, reference, model
end

if abspath(PROGRAM_FILE) == @__FILE__
    ref_path = isempty(ARGS) ? get(ENV, "MBBQ_SS_REFERENCE", DEFAULT_REFERENCE) : ARGS[1]
    run_steadystate_reference_test(ref_path = ref_path)
end
