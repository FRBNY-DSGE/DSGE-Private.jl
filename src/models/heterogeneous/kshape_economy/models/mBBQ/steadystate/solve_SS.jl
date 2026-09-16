"""
    solve_SS(grid, param)

Solve for the steady state using EGM. This code follows Bayer et al. (2018) closely.

# Dimensions of matrices
- Dimension 1: bond   b
- Dimension 2: equity a
- Dimension 3: idio. prod. s

# Arguments
- `grid`: Grid structure containing grid points and dimensions
- `param`: Parameter structure containing model parameters

# Returns
Tuple of 33 outputs:
- `c_n_guess`: Consumption policy (no adjustment) (nb × na × nse)
- `b_n_star`: Bond policy (no adjustment) (nb × na × nse)
- `c_a_guess`: Consumption policy (adjustment) (nb × na × nse)
- `b_a_star`: Bond policy (adjustment) (nb × na × nse)
- `a_a_star`: Equity policy (adjustment) (nb × na × nse)
- `psi_guess`: Lagrange multiplier (nb × na × nse)
- `mu_dist`: Distribution of households (nb × na × nse)
- `AProb`: Adjustment probability (nb × na × nse)
- `Value`: Value function (nb × na × nse)
- `R_fc`: Return on bonds (scalar)
- `H_fc`: Transition matrix (sparse)
- `W_fc`: Wage (scalar)
- `r_k`: Rental rate of capital (scalar)
- `mc`: Marginal cost (scalar)
- `v`: Utilization rate (scalar)
- `J`: Job value (vector)
- `u`: Unemployment rate (scalar)
- `V`: Value of unemployment (scalar)
- `M`: Matches (scalar)
- `f`: Job finding rate (scalar)
- `Profits_fc`: Profits (scalar)
- `Output`: Output (scalar)
- `n`: Employment (scalar)
- `grid`: Updated grid structure
- `param`: Updated parameter structure
- `P_SE`: Transition matrix for employment states (nse × nse)
- `Lambda`: Discount factor (scalar)
- `TT`: Transition matrix (sparse)
- `WW`: After-tax wage/benefit (3D array)
- `THETA`: Leverage ratio (scalar)
- `NWb`: Net worth of banks (scalar)
- `Profit_FI`: Financial intermediary profit (scalar)
- `Ab`: Bank equity holdings (scalar)
- `Bb`: Family deposits at FIs (scalar)
- `Cb`: Family consumption (scalar)

# Notes
- This function modifies `grid` and `param` in place
- Requires `compute_agg`, `policyguess`, and `compute_K` functions to be available
"""
function solve_SS(grid, param)
    
    # Constructing mesh grids
    
    # 1) Compute relevant agg. variables first
    
    t_start = time()
    
    # [meshes.b, meshes.a, meshes.se] = ndgrid(grid.b,grid.a,grid.se);
    meshes_b, meshes_a, meshes_se = ndgrid(grid["b"], grid["a"], grid["se"])
    meshes = Dict("b" => meshes_b, "a" => meshes_a, "se" => meshes_se)
    
    # AProb   = param.guessadj*ones(grid.nb,grid.na,grid.nse);
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    AProb = param["guessadj"] * ones(nb_val, na_val, nse_val)
    
    # mu_dist = ones(grid.nb,grid.na,grid.nse)/(grid.nb*grid.na*grid.nse);
    mu_dist = ones(nb_val, na_val, nse_val) / (nb_val * na_val * nse_val)
    
    # [ mc, L, n, h, r_k, Profit, v, J, u, V, M, f, w, WW, RR, RBRB, Y, P_SE, param, Lambda, grid, THETA, NW, Profit_FI, Ab, Bb, Cb ] = compute_agg(meshes,grid,param);
    mc, L, n, h, r_k, Profit, v, J, u, V, M, f, w, WW, RR, RBRB, Y, P_SE, param, Lambda, grid, THETA, NW, Profit_FI, Ab, Bb, Cb = 
        compute_agg(meshes, grid, param)
    
    # 2) Solve the terminal period's problem
    
    # [c_a_guess,c_n_guess, psi_guess, inc]= policyguess(meshes,WW,RR,RBRB,param,grid);
    c_a_guess, c_n_guess, psi_guess, inc = policyguess(meshes, WW, RR, RBRB, param, grid)
    
    # update
    
    # grid_K = grid.K;
    grid_K = grid["K"]
    
    # [excess, c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist, AProb, Value, mc, L, n, H_fc, r_k, Profits_fc, v, J, u, V, M, f, W_fc, WW, R_fc, RBRB, Output, param, Lambda, TT, grid ] = compute_K(grid.K,c_a_guess,c_n_guess,psi_guess,AProb,mu_dist, grid, param, meshes);
    # Note: MATLAB compute_K returns h (which is r_l from compute_agg), w, RR
    # solve_SS expects H_fc (which is r_l, the labor rental rate), W_fc (wage), R_fc (return on equity)
    # In MATLAB, h from compute_K is actually r_l (labor rental rate) from compute_agg
    excess, c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist, AProb, Value, mc, L, n, h, r_k, Profits_fc, v, J, u, V, M, f, w, WW, RR, RBRB, Output, param, Lambda, TT, grid = 
        compute_K(grid_K, c_a_guess, c_n_guess, psi_guess, AProb, mu_dist, grid, param, meshes)
    
    # Extract H_fc, W_fc, R_fc from the outputs
    # H_fc is r_l (labor rental rate, scalar) - in MATLAB compute_K, h is actually r_l from compute_agg
    # W_fc is w (wage, scalar)
    # R_fc is RR (return on equity, scalar)
    H_fc = h   # h from compute_K is actually r_l (labor rental rate) from compute_agg
    W_fc = w   # Wage
    R_fc = RR  # Return on equity (after-tax)
    
    # grid.L = L;                                % Aggregate labor input
    grid["L"] = L
    
    # aux    = squeeze(sum(sum(mu_dist,1),2));
    # In Julia, sum over dimensions 1 and 2, then squeeze
    aux = dropdims(sum(sum(mu_dist, dims=1), dims=2), dims=(1,2))
    
    # grid.N = sum(aux(1:grid.ns));              % Aggregate employment ( between 0 and 1)
    ns_val = Int(grid["ns"])
    grid["N"] = sum(aux[1:ns_val])
    
    # [ mc, L, n, h, r_k, Profit, v, J, u, V, M, f, w, WW, RR, RBRB, Y, P_SE, param, Lambda, grid, THETA, NWb, Profit_FI, Ab, Bb, Cb ] = compute_agg(meshes,grid,param);
    mc, L, n, h, r_k, Profit, v, J, u, V, M, f, w, WW, RR, RBRB, Y, P_SE, param, Lambda, grid, THETA, NWb, Profit_FI, Ab, Bb, Cb = 
        compute_agg(meshes, grid, param)
    
    tot_esp_time = time() - t_start
    
    # fprintf('  The steady state is found. Time elapsed is %4.2f \n',tot_esp_time);
    # fprintf('  Differnce is %1.10f \n', excess)
    # println("  The steady state is found. Time elapsed is $(round(tot_esp_time, digits=2))")
    # println("  Difference is $(excess)")
    
    return (c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist, AProb, Value, 
            R_fc, H_fc, W_fc, r_k, mc, v, J, u, V, M, f, Profits_fc, Output, n, grid, param, 
            P_SE, Lambda, TT, WW, THETA, NWb, Profit_FI, Ab, Bb, Cb)
end


function solve_SS(m::mBBQ)
    t_start = time()

    meshes = _mbbq_meshes(m)
    grid = _mbbq_ss_grid_dict(m)
    param = _mbbq_ss_param_dict(m)

    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    AProb = param["guessadj"] * ones(nb_val, na_val, nse_val)
    mu_dist = ones(nb_val, na_val, nse_val) / (nb_val * na_val * nse_val)

    mc, L, n, h, r_k, Profit, v, J, u, V, M, f, w, WW, RR, RBRB, Y, P_SE, param, Lambda, grid, THETA, NW, Profit_FI, Ab, Bb, Cb =
        compute_agg(meshes, grid, param)

    c_a_guess, c_n_guess, psi_guess, inc = policyguess(meshes, WW, RR, RBRB, param, grid)
    grid_K = grid["K"]

    excess, c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist, AProb, Value, mc, L, n, h, r_k, Profits_fc, v, J, u, V, M, f, w, WW, RR, RBRB, Output, param, Lambda, TT, grid =
        compute_K(grid_K, c_a_guess, c_n_guess, psi_guess, AProb, mu_dist, grid, param, meshes)

    H_fc = h
    W_fc = w
    R_fc = RR

    grid["L"] = L
    aux = dropdims(sum(sum(mu_dist, dims=1), dims=2), dims=(1, 2))
    ns_val = Int(grid["ns"])
    grid["N"] = sum(aux[1:ns_val])

    mc, L, n, h, r_k, Profit, v, J, u, V, M, f, w, WW, RR, RBRB, Y, P_SE, param, Lambda, grid, THETA, NWb, Profit_FI, Ab, Bb, Cb =
        compute_agg(meshes, grid, param)

    tot_esp_time = time() - t_start

    return (c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist, AProb, Value,
            R_fc, H_fc, W_fc, r_k, mc, v, J, u, V, M, f, Profits_fc, Output, n, grid, param,
            P_SE, Lambda, TT, WW, THETA, NWb, Profit_FI, Ab, Bb, Cb)
end