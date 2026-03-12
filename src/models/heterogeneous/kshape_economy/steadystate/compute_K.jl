using LinearAlgebra
using SparseArrays

include("compute_agg.jl")
include("policyguess.jl")
include("policies_SS.jl")
include("find_dist.jl")
include("CalValueSS.jl")

"""
    compute_K(K, c_a_guess, c_n_guess, psi_guess, AProb, mu_dist, grid, param, meshes)

Compute market clearing for capital given an aggregate capital level K.

This function iterates on the adjustment probability and distribution to find
consistent household policies and aggregate capital.

# Arguments
- `K`: Aggregate capital level (scalar)
- `c_a_guess`: Initial guess for consumption policy (adjustment case) (nb × na × nse)
- `c_n_guess`: Initial guess for consumption policy (no adjustment case) (nb × na × nse)
- `psi_guess`: Initial guess for Lagrange multiplier (nb × na × nse)
- `AProb`: Adjustment probability (nb × na × nse)
- `mu_dist`: Distribution of households (nb × na × nse)
- `grid`: Grid structure
- `param`: Parameter structure
- `meshes`: Mesh structure

# Returns
A tuple with:
- `excess`: Excess demand for capital (grid.K - K_agg)
- `c_n_guess`: Updated consumption policy (no adjustment)
- `b_n_star`: Bond policy (no adjustment)
- `c_a_guess`: Updated consumption policy (adjustment)
- `b_a_star`: Bond policy (adjustment)
- `a_a_star`: Equity policy (adjustment)
- `psi_guess`: Updated Lagrange multiplier
- `mu_dist`: Updated distribution
- `AProb_new`: Updated adjustment probability
- `Value`: Value function
- `mc`: Marginal cost
- `L`: Labor
- `n`: Employment
- `h`: Hours
- `r_k`: Rental rate of capital
- `Profit`: Profits
- `v`: Vacancies
- `J`: Job value
- `u`: Unemployment rate
- `V`: Value of unemployment
- `M`: Matches
- `f`: Job finding rate
- `w`: Wage
- `WW`: Wage income
- `RR`: Return on equity
- `RBRB`: Return on bonds
- `Y`: Output
- `param`: Updated parameter structure
- `Lambda`: Stochastic discount factor
- `TT`: Transition matrix
- `grid`: Updated grid structure

# Notes
- Iterates on adjustment probability until convergence
- Computes aggregate variables, solves household problem, and finds distribution
"""
function compute_K(K, c_a_guess, c_n_guess, psi_guess, AProb, mu_dist, grid, param, meshes)

    grid["K"] = K

    dist_PFI = 1000.0
    dist_AP = 1000.0
    iter_AP = 0
    tol_AP = 1e-10
    max_iter_AP = 5

    # Compute aggregate variables
    mc, L, n, h, r_k, Profit, v, J, u, V, M, f, w, WW, RR, RBRB, Y, P_SE, param, Lambda, grid, THETA, NWb, Profit_FI, K_b_agg, Bb, Cb =
        compute_agg(meshes, grid, param)

    # Generate income
    _, _, _, inc = policyguess(meshes, WW, RR, RBRB, param, grid)

    # Initialize output variables
    AProb_new = similar(AProb)
    Value = zeros(size(AProb))
    H = spzeros(length(AProb[:]), length(AProb[:]))
    
    # Initialize policy variables (needed outside loop)
    b_n_star = zeros(size(c_n_guess))
    b_a_star = zeros(size(c_a_guess))
    a_a_star = zeros(size(c_a_guess))

    # Iterate on adjustment probability
    while dist_AP > tol_AP && iter_AP < max_iter_AP
        iter_AP += 1

        # Solve household problem
        c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, dist_PFI =
            policies_SS(c_a_guess, c_n_guess, psi_guess, AProb, grid, inc, RR, RBRB, P_SE, param, meshes)

        # Calculate value function and adjustment probability
        AProb_new, Value, H = CalValueSS(b_n_star, b_a_star, a_a_star, c_a_guess, c_n_guess, AProb, P_SE, param, grid)

        dist_AP = maximum(abs.(AProb_new[:] .- AProb[:]))
        AProb = AProb_new
    end

    # Find invariant distribution
    # Convert grid dimensions to Int
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    # Create sparse identity matrix
    n_states = nb_val * na_val
    println("Creating sparse identity matrix of size $n_states × $n_states")
    # Use speye equivalent: sparse identity matrix
    # More efficient than sparse(1.0I, n, n) for large matrices
    I_sparse = sparse(1.0I, n_states, n_states)
    println("I_sparse created, nnz: $(nnz(I_sparse))")
    
    # Get P_SS3 and check its size
    P_SS3 = param["P_SS3"]
    P_SS3_T = P_SS3'  # Transpose
    println("P_SS3 size: $(size(P_SS3)), P_SS3' size: $(size(P_SS3_T))")
    
    # Convert to sparse to ensure kron result is sparse (MATLAB's speye ensures sparse result)
    P_SS3_T_sparse = sparse(P_SS3_T)
    println("Computing Kronecker product for H_tilde...")
    println("  Expected H_tilde size: ($(size(P_SS3_T, 1) * n_states), $(size(P_SS3_T, 2) * n_states))")
    
    # kron(A, B) in both MATLAB and Julia: if A is m×n and B is p×q, result is (m*p)×(n*q)
    # MATLAB: kron(param.P_SS3', speye(grid.nb*grid.na))
    # Julia: kron(sparse(param["P_SS3"]'), I_sparse) - convert to sparse to match MATLAB behavior
    H_tilde = kron(P_SS3_T_sparse, I_sparse)
    println("H_tilde created, size: $(size(H_tilde)), nnz: $(nnz(H_tilde))")
    P_SE_dist = param["P_SS2"]

    println("Calling find_dist...")
    mu_dist, dist_mu, TT = find_dist(mu_dist, H, AProb, b_n_star, b_a_star, a_a_star, H_tilde, P_SE_dist, param, grid)
    println("find_dist completed. Final dist_mu: $dist_mu")

    mu_dist = reshape(mu_dist, (nb_val, na_val, nse_val))
    mu_dist_tilde = H_tilde * mu_dist[:]
    mu_dist_tilde = reshape(mu_dist_tilde, (nb_val, na_val, nse_val))

    # Compute aggregate capital components
    K_NI_agg = sum(mu_dist_tilde[:] .* AProb[:] .* a_a_star[:] .+
                   mu_dist_tilde[:] .* (1 .- AProb[:]) .* meshes["a"][:])
    K_NI_agg = (1 - param["death_rate"]) * K_NI_agg

    K_F_agg = param["A_F_ratio"] * grid["K"]
    K_g_agg = param["A_g_ratio"] * Y
    K_I_agg = grid["K"] - K_NI_agg - K_F_agg - K_g_agg

    if K_I_agg < 0
        @warn "A negative equity holding by FIs was encountered."
    end

    # Compute financial intermediary variables
    R = param["R_cb"] / param["pi_cb"]  # ss real rate
    R_a = (param["q"] + RR) / param["q"]  # growth rate of return on equity
    Ab = K_I_agg

    param["Ab_ratio"] = Ab / grid["K"]

    THETA = param["THETA"]

    param["omega"] = (1 - param["theta_b"] * ((R_a - R) * THETA + R)) / THETA

    zz = (R_a - R) * THETA + R
    xx = zz

    ee = (1 - param["theta_b"]) * Lambda * R / (1 - param["theta_b"] * zz * Lambda)
    vv = ((1 - param["theta_b"]) * (R_a - R) * Lambda) / (1 - param["theta_b"] * xx * Lambda)

    param["DELTA"] = vv + ee / THETA

    NWb = Ab / THETA  # aggregate net worth of financial intermediaries
    Profit_FI = (1 - param["theta_b"]) * ((R_a - R) * THETA + R) * NWb - param["omega"] * param["q"] * Ab  # Profits of FI
    B_FI = (THETA - 1) * NWb  # total deposits held at FIs
    Bb = param["Bb_ratio"] * B_FI  # the family's deposits at FIs
    Cb = (Profit_FI + (R - 1) * Bb)  # the family's consumption level

    K_agg = K_NI_agg + K_I_agg + K_F_agg + K_g_agg

    # Compute excess demand
    excess = grid["K"] - K_agg

    return excess, c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist,
           AProb_new, Value, mc, L, n, h, r_k, Profit, v, J, u, V, M, f, w, WW, RR, RBRB,
           Y, param, Lambda, TT, grid
end
