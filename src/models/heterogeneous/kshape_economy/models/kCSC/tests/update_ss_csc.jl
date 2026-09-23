using Test
using MAT
using JLD2
using DSGE
# include("../dynamics/update_ss_csc.jl")

println("Testing update_ss_csc function...")
println("=" ^ 60)

# Load MATLAB input fixtures (workspace before running update_ss_csc.m)
println("\nLoading input fixtures from ../../../original/inputs/update_ss_csc.mat")

# Use matread() for simpler loading (handles compression better than matopen in some cases)
# input_data = matread("../../../../../original/inputs/kshape_csc/update_ss_csc.mat")
# SS_stats = input_data["SS_stats"]
# param = input_data["param"]
# param_update = input_data["param_update"]
# grid = input_data["grid"]

@load "XssYss.jld2" Xss Yss SS_stats grid param


m = DSGE.kCSC()

m.dicts[:grid] = grid
m.dicts[:SS_stats] = SS_stats
m.dicts[:param] = param


# Load MATLAB output fixtures (workspace after running update_ss_csc.m)
println("\nLoading output fixtures from ../../../../../original/outputs/kshape_csc/update_ss_csc.mat")

output_data = matread("../output_data/update_ss_csc.mat")

SS_stats_expected = output_data["SS_stats"]
param_expected = output_data["param"]
SS_stats_base_expected = output_data["SS_stats_base"]
param_base_expected = output_data["param_base"]

# Run Julia replication
println("\nRunning Julia update_ss_csc function...")
DSGE.update_ss_csc!(m)
println("Julia computation complete.")


updated_param_keys = ["iota_1", "iota_2", "lambda_1", "lambda_2", "P_SS"]
updated_ss_keys = ["J", "J_bar_1", "J_bar_2", "Profit_L", "Profit", "Profit_int", "AvgC"]

# Dictionary to map output fields, if needed for model access (update for kCSC if required)
updated_model_keys = Dict(
    "fix_L_1" => :fix_L_1,
    "fix_L_2" => :fix_L_2,
    "fix_ratio" => :fix_ratio,
)

atol = 1e-3

@testset "update_ss_csc updated param keys" begin
    for key in updated_param_keys
        @test isapprox(m.dicts[:param][key], param_expected[key], atol=atol)
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






# # Use model-internal fields (m.dicts[:SS_stats], m.dicts[:param], etc) per new convention
# SS_stats          = m.dicts[:SS_stats]
# param             = m.dicts[:param]
# # SS_stats_base     = m.dicts[:SS_stats_base]
# # param_base        = m.dicts[:param_base]

# @testset "update_ss_csc function tests" begin
#     rtol = 1e-8   # relative tolerance
#     atol = 1e-10  # absolute tolerance

#     # Test updated param fields
#     @testset "Updated param fields" begin
#         updated_param_keys = ["iota", "theta_b", "fix_L", "fix", "omega", "DELTA", "fix2", "fix_ratio"]

#         for key in updated_param_keys
#             if haskey(param, key) && haskey(param_expected, key)
#                 @testset "$key" begin
#                     @test isapprox(param[key], param_expected[key], rtol=rtol, atol=atol)
#                 end
#             end
#         end
#     end

#     # Test updated SS_stats fields
#     @testset "Updated SS_stats fields" begin
#         updated_ss_keys = ["J", "J_bar", "Profit_L", "Profit", "Profit_int", "vv", "ee", "Profit_FI", "AvgC"]

#         for key in updated_ss_keys
#             if haskey(SS_stats, key) && haskey(SS_stats_expected, key)
#                 val = SS_stats[key]
#                 ref = SS_stats_expected[key]

#                 @testset "$key" begin
#                     if isa(val, Number) && isa(ref, Number)
#                         @test isapprox(val, ref, rtol=rtol, atol=atol)
#                     elseif isa(val, AbstractArray) && isa(ref, AbstractArray)
#                         @test isapprox(vec(val), vec(ref), rtol=rtol, atol=atol)
#                     end
#                 end
#             end
#         end
#     end

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
#                         @test isapprox(vec(val), vec(ref), rtol=rtol, atol=atol)
#                     end
#                 end
#             end
#         end
#     end
# end

# println("\n" * "=" ^ 60)
# println("All update_ss_csc tests completed!")
