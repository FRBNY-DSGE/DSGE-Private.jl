"""
    find_alpha(in, V, vf, N_tilde, U_tilde, param)

Root-finding objective function for computing the alpha parameter in the steady state.

This function computes the residual (LHS - RHS) where the root (zero) corresponds to
the equilibrium value of alpha. It is typically called via a root-finding solver
like `fsolve` in MATLAB or `nlsolve` in Julia.

# Arguments
- `in`: Log of alpha (i.e., log(alpha)). This is the variable being solved for.
- `V`: Vacancy level (scalar)
- `vf`: Vacancy-filling rate (scalar)
- `N_tilde`: Adjusted employment level (scalar)
- `U_tilde`: Adjusted unemployment level (scalar)
- `param`: Parameter structure containing `lambda` (matching probability parameter)

# Returns
- `out`: Residual value (LHS - RHS). The root (where out = 0) gives the equilibrium alpha.

# Notes
- This is an objective function for root finding, not a direct computation
- In MATLAB: called via `fsolve(@(x) find_alpha(x, V, v_f, N_tilde, U_tilde, param), alpha_init)`
- In Julia: can be called via `nlsolve(x -> find_alpha(x[1], V, v_f, N_tilde, U_tilde, param), [alpha_init])`
- The function computes: LHS = vf*V, RHS = (U_tilde+lambda*N_tilde)*V / ((U_tilde+lambda*N_tilde)^alpha + V^alpha)^(1/alpha)


"""
function find_alpha(in, V, vf, N_tilde, U_tilde, param)
    # Compute alpha from log(alpha)
    # exp() works for both scalars and arrays
    alpha = exp(in)
    
    # Left-hand side: M = vf * V
    LHS = vf * V
    
    # Right-hand side: (U_tilde + lambda*N_tilde) * V / ((U_tilde + lambda*N_tilde)^alpha + V^alpha)^(1/alpha)
    # MATLAB: RHS = (U_tilde+param.lambda*N_tilde)*V/(((U_tilde+param.lambda*N_tilde)^alpha+V^alpha)^(1/alpha));
    
    # Access param.lambda - handle both Dict (from MAT files) and struct-like access
    # MAT.jl loads MATLAB structs as Dict{String, Any}, so use bracket notation
    lambda = if param isa Dict
        # Try string key first (MAT.jl default), then symbol key
        haskey(param, "lambda") ? param["lambda"] : param[:lambda]
    else
        # Assume struct-like access with dot notation
        param.lambda
    end
    
    # Handle both scalar and array inputs for alpha
    # For arrays, use broadcasting (.^) for element-wise operations
    if alpha isa AbstractArray
        numerator = (U_tilde + lambda * N_tilde) * V
        U_lambda_N = U_tilde + lambda * N_tilde
        denominator_base = U_lambda_N .^ alpha .+ V .^ alpha
        denominator = denominator_base .^ (1.0 ./ alpha)
        RHS = numerator ./ denominator
    else
        # Scalar case (typical use in root-finding)
        numerator = (U_tilde + lambda * N_tilde) * V
        U_lambda_N = U_tilde + lambda * N_tilde
        denominator_base = U_lambda_N^alpha + V^alpha
        denominator = denominator_base^(1/alpha)
        RHS = numerator / denominator
    end
    
    # Residual: LHS - RHS (we want this to be zero)
    out = LHS - RHS
    
    return out
end

