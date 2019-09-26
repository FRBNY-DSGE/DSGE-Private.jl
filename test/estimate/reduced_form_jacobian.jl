# Set up model and arguments
path = dirname(@__FILE__)
m = AnSchorfheide()
params = load("$(path)/../reference/reduced_form_jacobian_input.jld2")["theta"]
DSGE.update!(m, params)
Γ0, Γ1, Γ2, Γ3 = eqcond(m, method = :klein)
T, R, C = solve(m, method = :klein)

# Test differentiation
struct_mat_diff = load("$(path)/../reference/structural_matrices_jacobian_output.jld2")
@testset "Test structural matrices are correctly differentiated" begin
    ∂Γ0∂θ, ∂Γ1∂θ, ∂Γ2∂θ, ∂Γ3∂θ = structural_matrices_jacobian(m, m.parameters)
    @test_matrix_approx_eq ∂Γ0∂θ struct_mat_diff["dGamma0dtheta"]
    @test_matrix_approx_eq ∂Γ1∂θ struct_mat_diff["dGamma1dtheta"]
    @test_matrix_approx_eq ∂Γ2∂θ struct_mat_diff["dGamma2dtheta"]
    @test_matrix_approx_eq ∂Γ3∂θ struct_mat_diff["dGamma3dtheta"]
end

trans_mat_diff = load("$(path)/../reference/transition_matrices_jacobian_output.jld2")
@testset "Test transition matrices are correctly differentiated" begin
    ∂T∂θ, ∂R∂θ = transition_matrices_jacobian(m, m.parameters, T,
                                              Γ0, Γ1, Γ2, Γ3)
    @test_matrix_approx_eq ∂T∂θ trans_mat_diff["dTdtheta"]
    @test_matrix_approx_eq ∂R∂θ trans_mat_diff["dRdtheta"]
end

meas_mat_diff = load("$(path)/../reference/meas_matrices_jacobian_output.jld2")
@testset "Test measurement matrices are correctly differentiated" begin
    ∂T∂θ, ∂R∂θ = transition_matrices_jacobian(m, m.parameters, T, Γ0, Γ1, Γ2, Γ3)
    ∂Z∂θ, ∂D∂θ = measurement_matricess_jacobian(m, m.parameters, T, R, ∂T∂θ, ∂R∂θ)
    @test_matrix_approx_eq ∂Z∂θ meas_mat_diff["dZdtheta"]
    @test_matrix_approx_eq ∂D∂θ meas_mat_diff["dDdtheta"]
end

@testset "Test the reduced form matrices are correctly differentiated" begin
    ∂T∂θ, ∂R∂θ, ∂Z∂θ, ∂D∂θ = reduced_form_jacobian(m, m.parameters, T, R, Γ0, Γ1, Γ2, Γ3)
    @test_matrix_approx_eq ∂T∂θ trans_mat_diff["dTdtheta"]
    @test_matrix_approx_eq ∂R∂θ trans_mat_diff["dRdtheta"]
    @test_matrix_approx_eq ∂Z∂θ meas_mat_diff["dZdtheta"]
    @test_matrix_approx_eq ∂D∂θ meas_mat_diff["dDdtheta"]
end
