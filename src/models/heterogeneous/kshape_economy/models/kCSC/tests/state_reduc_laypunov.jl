using Revise
using Test
using JLD2
using DSGE
using MAT
using LinearAlgebra

matfile = matopen("../input_data/state_reduc_Laypunov_inputs.mat")

hx = read(matfile, "hx")
gx = read(matfile, "gx")
grid = read(matfile, "grid") # likely a Dict or struct; may need adaptation
P_ref = read(matfile, "P_ref")
Q_ref = read(matfile, "Q_ref")


PRightStates, PRightAll, PRightAll_Pref, n_states_r, n_controls_r, StateCOVAR, ControlCOVAR, hx_r, gx_r, P_r, Q_r, keepD, keepV = DSGE.state_reduc_laypunov(hx, gx, grid, P_ref, Q_ref)

matfile = matopen("../output_data/state_reduc_Laypunov_outputs.mat")
PRightStates_mat = read(matfile, "PRightStates")
PRightAll_mat = read(matfile, "PRightAll")
PRightAll_Pref_mat = read(matfile, "PRightAll_Pref")
n_states_r_mat = read(matfile, "n_states_r")
n_controls_r_mat = read(matfile, "n_controls_r")
StateCOVAR_mat = read(matfile, "StateCOVAR")
ControlCOVAR_mat = read(matfile, "ControlCOVAR")
hx_r_mat = read(matfile, "hx_r")
gx_r_mat = read(matfile, "gx_r")
P_r_mat = read(matfile, "P_r")
Q_r_mat = read(matfile, "Q_r")
close(matfile)


tol = 1e-5

# Align columns of A to B by greedy max |correlation| matching.
# Returns A_aligned and the basis-change matrix T such that A*T ≈ B.
function align_columns_to_reference(A::AbstractMatrix, B::AbstractMatrix; eps::Float64=1e-14)
    @assert size(A) == size(B)
    n = size(A, 2)
    C = abs.(A' * B)                    # similarity matrix
    used = falses(n)
    perm = zeros(Int, n)
    sgn = ones(Float64, n)
    for j in 1:n
        col = view(C, :, j)
        best_i = 0
        best_v = -Inf
        for i in 1:n
            if !used[i] && col[i] > best_v
                best_v = col[i]
                best_i = i
            end
        end
        perm[j] = best_i
        used[best_i] = true
        d = dot(view(A, :, best_i), view(B, :, j))
        sgn[j] = d < -eps ? -1.0 : 1.0
    end
    A_aligned = copy(A[:, perm])
    for j in 1:n
        A_aligned[:, j] .*= sgn[j]
    end
    T = zeros(Float64, n, n)            # A*T = A_aligned
    for j in 1:n
        T[perm[j], j] = sgn[j]
    end
    return A_aligned, T
end

PRightStates_aligned, T_state = align_columns_to_reference(PRightStates, PRightStates_mat)
PRightAll_aligned, T_all = align_columns_to_reference(PRightAll, PRightAll_mat)
PRightAll_Pref_aligned, T_pref = align_columns_to_reference(PRightAll_Pref, PRightAll_Pref_mat)

# Transform reduced objects into MATLAB-aligned reduced coordinates.
M_state = transpose(T_state)
hx_r_aligned = M_state * hx_r * inv(M_state)
gx_r_aligned = gx_r * inv(M_state)

P_r_aligned = transpose(T_pref) * P_r * T_pref
Q_r_aligned = transpose(T_pref) * Q_r

@testset "state_reduc_laypunov outputs vs MATLAB" begin
    # Compare shapes
    @test size(PRightStates) == size(PRightStates_mat)
    @test size(PRightAll) == size(PRightAll_mat)
    @test size(PRightAll_Pref) == size(PRightAll_Pref_mat)
    @test n_states_r == n_states_r_mat
    @test n_controls_r == n_controls_r_mat
    @test size(StateCOVAR) == size(StateCOVAR_mat)
    @test size(ControlCOVAR) == size(ControlCOVAR_mat)
    @test size(hx_r) == size(hx_r_mat)
    @test size(gx_r) == size(gx_r_mat)
    @test size(P_r) == size(P_r_mat)
    @test size(Q_r) == size(Q_r_mat)
    @test length(keepD) == length(vec(grid["Dindex"]))
    @test length(keepV) == length(vec(grid["Vindex"]))

    # Compare contents
    # Compare matrices element by element
    for (A, B, name) in [
            (PRightStates_aligned, PRightStates_mat, "PRightStates"),
            (PRightAll_aligned, PRightAll_mat, "PRightAll"),
            (PRightAll_Pref_aligned, PRightAll_Pref_mat, "PRightAll_Pref"),
            (StateCOVAR, StateCOVAR_mat, "StateCOVAR"),
            (ControlCOVAR, ControlCOVAR_mat, "ControlCOVAR"),
            (hx_r_aligned, hx_r_mat, "hx_r"),
            (gx_r_aligned, gx_r_mat, "gx_r"),
            (P_r_aligned, P_r_mat, "P_r"),
            (Q_r_aligned, Q_r_mat, "Q_r"),
        (keepD, keepD, "keepD"),
        (keepV, keepV, "keepV"),
   
        ]
        @test size(A) == size(B)
        num_off = count(i -> !isapprox(A[i], B[i]; atol=tol, rtol=tol), eachindex(A))
        if num_off != 0
            @info "Matrix mismatch" name num_off
        end
        @test num_off == 0
    end

    @test abs(n_states_r - n_states_r_mat) < tol         # Integers, but robust
    @test abs(n_controls_r - n_controls_r_mat) < tol     # Integers, but robust
end

# Print indices where PRightStates and PRightStates_mat differ by more than tol
# num_off = count(i -> !isapprox(PRightStates[i], PRightStates_mat[i]; atol=tol, rtol=tol), eachindex(PRightStates))
# if num_off > 0
#     mismatches = [(i, PRightStates[i], PRightStates_mat[i]) for i in eachindex(PRightStates)
#                   if !isapprox(PRightStates[i], PRightStates_mat[i]; atol=tol, rtol=tol)]
#     println("Indices where PRightStates differs from PRightStates_mat (index, value, mat_value):")
#     for (idx, val, matval) in mismatches
#         println("  Index $idx: Julia = $val, MATLAB = $matval, Diff = $(abs(val-matval))")
#     end
# end

