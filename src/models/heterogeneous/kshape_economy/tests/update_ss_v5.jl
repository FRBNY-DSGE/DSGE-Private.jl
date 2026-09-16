using Revise
using DSGE

using Test
using MAT
using JLD2
include("../dynamics/update_ss_v5.jl")

println("Testing update_ss_v5 function...")
println("=" ^ 60)

# Load MATLAB input fixtures (workspace before running update_ss_v5.m)
println("\nLoading input fixtures from ../data/inputs/update_ss_v5.mat")

# Use matread() for simpler loading (handles compression better than matopen in some cases)
# input_data = matread("../data/inputs/update_ss_v5.mat")

# SS_stats = input_data["SS_stats"]
# param = input_data["param"]
# # param_update = input_data["param_update"]
# grid = input_data["grid"]

# Load MATLAB output fixtures (workspace after running update_ss_v5.m)
println("\nLoading output fixtures from ../../../original/outputs/update_ss_v5.mat")

# output_data = matread("../data/outputs/update_ss_v5.mat")
output_data = matread("../data/update_ss.mat")


SS_stats_expected = output_data["SS_stats"]
param_expected = output_data["param"]
# SS_stats_base_expected = output_data["SS_stats_base"]
# param_base_expected = output_data["param_base"]

jld2file = "../data/XssYss.jld2"
@load jld2file StateSS ControlSS SS_stats grid param
m = mBBQ()
m.dicts[:grid] = grid
m.dicts[:SS_stats] = SS_stats
m.dicts[:param] = param
# m.grids[:StateSS] = StateSS
# m.grids[:ControlSS] = ControlSS

# Run Julia replication
println("\nRunning Julia update_ss_v5 function...")
DSGE.update_ss_v5!(m)
println("Julia computation complete.")



updated_param_keys_SS = ["fix_ratio"]
updated_ss_keys = ["J", "J_bar", "Profit_L", "Profit", "Profit_int", "vv", "ee", "Profit_FI", "AvgC"]
updated_model_keys = Dict(
    "fix_L" => :fix_L,
    "iota" => :ι,
    "theta_b" => :θ_b,
    "fix" => :fix,
    "omega" => :ω,
    "DELTA" => :δ,
    "fix2" => :fix2
)

atol = 1e-3

@testset "update_ss_v5 updated param keys" begin
    for param in updated_param_keys_SS
        @test isapprox(m.dicts[:param][param], param_expected[param], atol=atol)
    end
end

for param in updated_ss_keys
    @testset "$param" begin
        val = m.dicts[:SS_stats][param]
        ref = SS_stats_expected[param]
        if isa(val, Number) && isa(ref, Number)
            @test isapprox(val, ref, atol=atol)
        elseif isa(val, AbstractArray) && isa(ref, AbstractArray)
            # @test size(val) == size(ref)
            @test isapprox(val, ref, atol=atol)
        else
            @test val == ref  # fallback to equality for unhandled types
        end
    end
end

for param in keys(updated_model_keys)
    @testset "model param $param" begin
        model_key = updated_model_keys[param]
        val = m[model_key].value
        ref = param_expected[param]
        if isa(val, Number) && isa(ref, Number)
            @test isapprox(val, ref, atol=atol)
        elseif isa(val, AbstractArray) && isa(ref, AbstractArray)
            @test isapprox(val, ref, atol=atol)
        else
            @test val == ref  # fallback to equality for unhandled types
        end
    end
end









# Run tests
@testset "update_ss_v5 function tests" begin
    rtol = 1e-8   # relative tolerance
    atol = 1e-10  # absolute tolerance

    # # Test that base copies were created correctly
    # @testset "Base copies" begin
    #     @testset "SS_stats_base" begin
    #         for (key, ref) in SS_stats_base_expected
    #             if haskey(SS_stats_base, key)
    #                 val = SS_stats_base[key]

    #                 # Skip nested dictionaries and non-numeric types
    #                 if isa(val, Dict) || isa(ref, Dict) || isa(val, AbstractString) || isa(ref, AbstractString)
    #                     continue
    #                 end

    #                 @testset "$key" begin
    #                     if isa(val, Number) && isa(ref, Number)
    #                         @test isapprox(val, ref, rtol=rtol, atol=atol)
    #                     elseif isa(val, AbstractArray) && isa(ref, AbstractArray)
    #                         @test size(val) == size(ref)
    #                         @test isapprox(val, ref, rtol=rtol, atol=atol)
    #                     end
    #                 end
    #             end
    #         end
    #     end

    #     @testset "param_base" begin
    #         for (key, ref) in param_base_expected
    #             if haskey(param_base, key)
    #                 val = param_base[key]

    #                 # Skip nested dictionaries and non-numeric types
    #                 if isa(val, Dict) || isa(ref, Dict) || isa(val, AbstractString) || isa(ref, AbstractString)
    #                     continue
    #                 end

    #                 @testset "$key" begin
    #                     if isa(val, Number) && isa(ref, Number)
    #                         @test isapprox(val, ref, rtol=rtol, atol=atol)
    #                     elseif isa(val, AbstractArray) && isa(ref, AbstractArray)
    #                         @test size(val) == size(ref)
    #                         @test isapprox(val, ref, rtol=rtol, atol=atol)
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end

    # Test updated param fields
    # @testset "Updated param fields" begin
 
    #     for key in updated_param_keys
    #         if haskey(param, key) && haskey(param_expected, key)
    #             @testset "$key" begin
    #                 @test isapprox(m.dicts[:param][key], param_expected[key], rtol=rtol, atol=atol)
    #             end
    #         end
    #     end
    # end

    # # Test updated SS_stats fields
    # @testset "Updated SS_stats fields" begin

    #     for key in updated_ss_keys
    #         if haskey(SS_stats, key) && haskey(SS_stats_expected, key)
    #             val = SS_stats[key]
    #             ref = SS_stats_expected[key]

    #             @testset "$key" begin
    #                 if isa(val, Number) && isa(ref, Number)
    #                     @test isapprox(val, ref, rtol=rtol, atol=atol)
    #                 elseif isa(val, AbstractArray) && isa(ref, AbstractArray)
    #                     @test size(val) == size(ref)
    #                     @test isapprox(val, ref, rtol=rtol, atol=atol)
    #                 end
    #             end
    #         end
    #     end
    # end

#     # Comprehensive test - check all param fields
#     @testset "All param fields" begin
#         for (key, ref) in param_expected
#             if haskey(param, key)
#                 val = param[key]

#                 # Skip nested dictionaries and non-numeric types
#                 if isa(val, Dict) || isa(ref, Dict) || isa(val, AbstractString) || isa(ref, AbstractString)
#                     continue
#                 end

#                 @testset "$key" begin
#                     if isa(val, Number) && isa(ref, Number)
#                         @test isapprox(val, ref, rtol=rtol, atol=atol)
#                     elseif isa(val, AbstractArray) && isa(ref, AbstractArray)
#                         @test size(val) == size(ref)
#                         @test isapprox(val, ref, rtol=rtol, atol=atol)
#                     end
#                 end
#             end
#         end
#     end

#     # Comprehensive test - check all SS_stats fields
#     @testset "All SS_stats fields" begin
#         for (key, ref) in SS_stats_expected
#             if haskey(SS_stats, key)
#                 val = SS_stats[key]

#                 # Skip nested dictionaries and non-numeric types
#                 if isa(val, Dict) || isa(ref, Dict) || isa(val, AbstractString) || isa(ref, AbstractString)
#                     continue
#                 end

#                 @testset "$key" begin
#                     if isa(val, Number) && isa(ref, Number)
#                         @test isapprox(val, ref, rtol=rtol, atol=atol)
#                     elseif isa(val, AbstractArray) && isa(ref, AbstractArray)
#                         @test size(val) == size(ref)
#                         @test isapprox(val, ref, rtol=rtol, atol=atol)
#                     end
#                 end
#             end
#         end
#     end
end

println("\n" * "=" ^ 60)
println("All update_ss_v5 tests completed!")
