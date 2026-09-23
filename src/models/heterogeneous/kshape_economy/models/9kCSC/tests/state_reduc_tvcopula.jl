using Revise
using DSGE

using Test
using MAT
using SparseArrays
using JLD2
# xx = Ref{Any}(Dict())
# include(joinpath(@__DIR__, "../dynamics/state_reduc_tvcopula.jl"))
# include(joinpath(@__DIR__, "../../../test_utils.jl"))

println("Testing state_reduc_tvcopula function...")
println("=" ^ 60)

# Load MATLAB input fixtures
input_path = joinpath(@__DIR__, "../input_data/state_reduc_tvcopula.mat")
println("\nLoading input fixtures from $input_path")
input_data = matread(input_path)

# param = input_data["param"]
# grid = input_data["grid"]
# SS_stats = input_data["SS_stats"]
mu_dist = input_data["mu_dist"]
Value = input_data["Value"]
mutil_c = input_data["mutil_c"]
Va = input_data["Va"]

@load "XssYss.jld2" Xss Yss SS_stats grid param

m = DSGE.kCSC()
m.dicts[:grid] = grid
m.dicts[:SS_stats] = SS_stats
m.dicts[:param] = param

m.grids[:mu_dist] = mu_dist
m.grids[:Value] = Value
m.grids[:mutil_c] = mutil_c
m.grids[:Va] = Va



# Load MATLAB output fixtures
output_path = joinpath(@__DIR__, "../output_data/state_reduc_tvcopula.mat")
println("\nLoading output fixtures from $output_path")
output_data = matread(output_path)

# Run Julia replication
println("\nRunning Julia state_reduc_tvcopula! function...")
DSGE.state_reduc_csc(m)

println("Julia computation complete.")

# Run tests
rtol = 1e-8
atol = 1e-10

@testset "Xss matches StateSS" begin
    expected = vec(m.grids[:StateSS])
    actual = vec(Xss)
    @test isapprox(actual, expected; rtol=rtol, atol=atol)
end

@testset "Yss matches ControlSS" begin
    expected = vec(m.grids[:ControlSS])
    actual = vec(Yss)
    @test isapprox(actual, expected; rtol=rtol, atol=atol)
end


# @testset "state_reduc_tvcopula" begin
#     @testset "Xss" begin
#         expected = vec(output_data["Xss"])
#         actual = vec(results.Xss)
#         @test compare_arrays(actual, expected, "Xss"; rtol=rtol, atol=atol)
#     end

#     @testset "Yss" begin
#         expected = vec(output_data["Yss"])
#         actual = vec(results.Yss)
#         @test compare_arrays(actual, expected, "Yss"; rtol=rtol, atol=atol)
#     end

#     @testset "Gamma_state" begin
#         expected = output_data["Gamma_state"]
#         actual = Matrix(results.Gamma_state)
#         @test compare_arrays(actual, expected, "Gamma_state"; rtol=rtol, atol=atol)
#     end

#     @testset "Gamma_control" begin
#         expected = output_data["Gamma_control"]
#         actual = Matrix(results.Gamma_control)
#         @test compare_arrays(actual, expected, "Gamma_control"; rtol=rtol, atol=atol)
#     end

#     @testset "InvGamma" begin
#         expected = output_data["InvGamma"]
#         actual = Matrix(results.InvGamma)
#         @test compare_arrays(actual, expected, "InvGamma"; rtol=rtol, atol=atol)
#     end

#     @testset "Grid parameters" begin
#         # The output fixture saves nRedStates/nRedCtrl/nRedMarg/nPoly/compressionIndexesCOP
#         # as top-level variables, not under "grid". Derive the expected grid fields from those.
#         _cop_expected = vec(Int.(output_data["compressionIndexesCOP"]))
#         _nCOP_expected = length(_cop_expected)
#         _nRedMarg_expected = Int(output_data["nRedMarg"])
#         _nRedStates_expected = Int(output_data["nRedStates"])
#         _nRedCtrl_expected = Int(output_data["nRedCtrl"])
#         _nPoly_expected = Int(output_data["nPoly"])
#         @testset "numstates" begin
#             @test grid["numstates"] == _nRedStates_expected
#         end
#         @testset "numcontrols" begin
#             @test grid["numcontrols"] == _nRedCtrl_expected
#         end
#         @testset "nCOP" begin
#             @test grid["nCOP"] == _nCOP_expected
#         end
#         @testset "os" begin
#             @test grid["os"] == _nRedStates_expected - _nRedMarg_expected - _nCOP_expected
#         end
#         @testset "oc" begin
#             @test grid["oc"] == _nRedCtrl_expected - 3 * _nPoly_expected
#         end
#     end

#     @testset "compressionIndexesCOP" begin
#         expected = vec(Int.(output_data["compressionIndexesCOP"]))
#         actual = results.compressionIndexesCOP
#         @test actual == expected
#     end

#     @testset "State vectors" begin
#         @testset "State" begin
#             expected = vec(output_data["State"])
#             actual = results.State
#             @test compare_arrays(actual, expected, "State"; rtol=rtol, atol=atol)
#         end
#         @testset "Contr" begin
#             expected = vec(output_data["Contr"])
#             actual = results.Contr
#             @test compare_arrays(actual, expected, "Contr"; rtol=rtol, atol=atol)
#         end
#     end
# end

# # Summary
# println("\n" * "=" ^ 60)
# println("COMPARISON SUMMARY")
# println("=" ^ 60)

# outputs_to_check = ["Xss", "Yss", "Gamma_state", "Gamma_control", "InvGamma", "State", "Contr"]
# for name in outputs_to_check
#     if haskey(output_data, name)
#         expected = output_data[name]
#         if name == "Xss"
#             actual = results.Xss
#         elseif name == "Yss"
#             actual = results.Yss
#         elseif name == "Gamma_state"
#             actual = Matrix(results.Gamma_state)
#         elseif name == "Gamma_control"
#             actual = Matrix(results.Gamma_control)
#         elseif name == "InvGamma"
#             actual = Matrix(results.InvGamma)
#         elseif name == "State"
#             actual = results.State
#         elseif name == "Contr"
#             actual = results.Contr
#         end

#         max_diff = maximum(abs.(vec(actual) .- vec(expected)))
#         status = isapprox(actual, expected, rtol=rtol, atol=atol) ? "✓" : "✗"
#         println("  $status $name: max diff = $max_diff")
#     end
# end
# println("=" ^ 60)
