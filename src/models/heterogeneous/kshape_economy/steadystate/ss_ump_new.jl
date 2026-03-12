using MAT
using Interpolations

include("compute_agg.jl")
include("policyguess.jl")
include("compute_K.jl")
include("helpers/Cal_SS_stats.jl")
include("helpers/griddedInterpolant.jl")
include("helpers/ndgrid.jl")
include("solve_SS.jl")
include("helpers/steady_anal.jl")

"""
    ss_ump(param_file, grid_file)

Main function to compute the steady state for the HANK model.

This function:
1. Loads parameters and grid
2. Solves for steady state capital using solve_SS
3. Prepares meshes and calculates marginal values
4. Produces a non-parametric copula
5. Computes steady state statistics

# Arguments
- `param_file`: Path to parameter file (.mat file)
- `grid_file`: Path to grid file (.mat file)

# Returns
A dictionary containing:
- All steady state variables
- SS_stats: Dictionary of steady state statistics
- Copula: Interpolation object for the cumulative distribution
"""
function ss_ump_new(param, grid)

    # println("Loading parameters and grid...")

    # # 1) Load parameters
    # mat_param = matopen(param_file)
    # param_ss = read(mat_param, "param_ss_new")
    # close(mat_param)
    # param = param_ss

    # # 2) Load grid
    # mat_grid = matopen(grid_file)
    # grid_ss = read(mat_grid, "grid_ss_new")
    # close(mat_grid)
    # grid = grid_ss

    # Ensure grid vectors are proper 1D arrays
    grid["b"] = vec(grid["b"])
    grid["a"] = vec(grid["a"])
    grid["se"] = vec(grid["se"])
    # Don't vectorize grid["s"] - it should remain as a row vector (1×ns) for compute_agg
    grid["nb"] = Int(grid["nb"])
    grid["na"] = Int(grid["na"])
    grid["nse"] = Int(grid["nse"])
    grid["ns"] = Int(grid["ns"])

    println("Solving for steady state...")

    # 3) Solve for steady state capital (K) using solve_SS
    c_n_guess, b_n_star, c_a_guess, b_a_star, a_a_star, psi_guess, mu_dist,
        AProb, Value, R_fc, H_fc, W_fc, r_k, mc, v, J, u, V, M, f, Profits_fc,
        Output, n, grid, param, P_SE, Lambda, TT, WW, THETA, NWb, Profit_FI, Ab, Bb, Cb =
        solve_SS(grid, param)

    println("Steady state solution complete.")

    # 4) Prepare meshes for state space
    meshes = Dict{String, Any}()
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])

    meshes["b"] = zeros(nb_val, na_val, nse_val)
    meshes["a"] = zeros(nb_val, na_val, nse_val)
    meshes["se"] = zeros(nb_val, na_val, nse_val)

    for k in 1:nse_val
        for j in 1:na_val
            for i in 1:nb_val
                meshes["b"][i, j, k] = grid["b"][i]
                meshes["a"][i, j, k] = grid["a"][j]
                meshes["se"][i, j, k] = grid["se"][k]
            end
        end
    end

    # Create se_aux grid and mesh (needed for Cal_SS_stats)
    grid_se_aux = hcat(grid["s"],
                      min.(param["b_ratio"] .* grid["s"], grid["s_bar"]),
                      [1])
    meshes["se_aux"] = zeros(nb_val, na_val, size(grid_se_aux, 2))
    for k in 1:size(grid_se_aux, 2)
        for j in 1:na_val
            for i in 1:nb_val
                meshes["se_aux"][i, j, k] = grid_se_aux[k]
            end
        end
    end

    # Calculate marginal values of capital (k) and liquid assets (m)
    println("Computing marginal values...")

    RBRB = param["R_cb"] / param["pi_cb"] .+ (meshes["b"] .< 0) .* (param["Rprem"] / param["pi_cb"])

    # Liquid Asset marginal value
    mutil_c_n = 1 ./ (c_n_guess .^ param["sigma"])  # marginal utility at consumption policy no adjustment
    mutil_c_a = 1 ./ (c_a_guess .^ param["sigma"])  # marginal utility at consumption policy adjustment
    mutil_c = AProb .* mutil_c_a .+ (1 .- AProb) .* mutil_c_n  # Expected marginal utility

    # Vb = RBRB .* mutil_c  # take return on money into account
    Vb = RBRB / (1 - param["death_rate"]) * param["b_a_aux"] .* mutil_c
    Vb = reshape(reshape(Vb, (nb_val * na_val, nse_val)), (nb_val, na_val, nse_val))

    # Capital marginal value
    r_a_aux = R_fc + param["death_rate"] / (1 - param["death_rate"]) * param["q"]

    Va = AProb .* (r_a_aux + param["q"]) .* mutil_c_a .+
         (1 .- AProb) .* r_a_aux .* mutil_c_n .+
         (1 .- AProb) .* psi_guess
    Va = reshape(reshape(Va, (nb_val * na_val, nse_val)), (nb_val, na_val, nse_val))

    # 5) Produce non-parametric Copula
    println("Creating non-parametric copula...")

    cum_dist = cumsum(cumsum(cumsum(mu_dist, dims=1), dims=2), dims=3)
    marginal_b = cumsum(dropdims(sum(sum(mu_dist, dims=2), dims=3), dims=(2,3)))
    marginal_a = cumsum(dropdims(sum(sum(mu_dist, dims=1), dims=3), dims=(1,3)))
    marginal_se = cumsum(dropdims(sum(sum(mu_dist, dims=2), dims=1), dims=(1,2)))

    # Create interpolation object (equivalent to MATLAB's griddedInterpolant)
    # Follow same pattern as EGM_Step1.jl: interpolate + extrapolate
    marginal_a_t = vec(marginal_a')
    Copula_itp = interpolate((marginal_b, marginal_a_t, marginal_se), cum_dist, Gridded(Linear()))
    Copula_itp_ext = extrapolate(Copula_itp, Line())
    
    # Evaluate on the grid points to get 40x40x11 array (matching MATLAB output)
    # The interpolant is defined on (marginal_b, marginal_a_t, marginal_se) grid
    # Evaluate at those same points to get the original cum_dist values
    meshes_b, meshes_a, meshes_se = ndgrid(marginal_b, marginal_a_t, marginal_se)
    Copula = Copula_itp_ext.(meshes_b, meshes_a, meshes_se)

    # 6) Compute steady state statistics
    println("Computing steady state statistics...")

    SS_stats, grid, param = Cal_SS_stats(
        mu_dist, c_a_guess, c_n_guess, AProb, b_n_star, b_a_star, a_a_star,
        grid, param, meshes, TT, Output, R_fc, W_fc, H_fc, Profits_fc,
        n, u, v, J, V, M, f, mc, Lambda, NWb, THETA, Profit_FI, Ab, Bb, Cb,
        mutil_c, mutil_c_a, mutil_c_n, Value
    )

    println("Steady state computation complete.")
    anal_stats, SS_stats_steady, Q_dists = steady_anal(grid, param, meshes, SS_stats, AProb, mu_dist, n, SS_stats["H_tilde"],
                    b_n_star, b_a_star, a_a_star, c_n_guess, c_a_guess)

    merge!(SS_stats, SS_stats_steady)

    # Collect all results in a dictionary
    intermediate_results = Dict{String, Any}(
        "c_n_guess" => c_n_guess,
        "b_n_star" => b_n_star,
        "c_a_guess" => c_a_guess,
        "b_a_star" => b_a_star,
        "a_a_star" => a_a_star,
        "psi_guess" => psi_guess,
        "mu_dist" => mu_dist,
        "AProb" => AProb,
        "Value" => Value,
        "R_fc" => R_fc,
        "H_fc" => H_fc,
        "W_fc" => W_fc,
        "r_k" => r_k,
        "mc" => mc,
        "v" => v,
        "J" => J,
        "u" => u,
        "V" => V,
        "M" => M,
        "f" => f,
        "Profits_fc" => Profits_fc,
        "Output" => Output,
        "n" => n,
        "grid" => grid,
        "param" => param,
        "P_SE" => P_SE,
        "Lambda" => Lambda,
        "TT" => TT,
        "WW" => WW,
        "THETA" => THETA,
        "NWb" => NWb,
        "Profit_FI" => Profit_FI,
        "Ab" => Ab,
        "Bb" => Bb,
        "Cb" => Cb,
        "mutil_c" => mutil_c,
        "mutil_c_a" => mutil_c_a,
        "mutil_c_n" => mutil_c_n,
        "Vb" => Vb,
        "Va" => Va,
        "meshes" => meshes,
        "SS_stats" => SS_stats,
        "Copula" => Copula,
        "anal_stats" => anal_stats,
        "Q_dists" => Q_dists,
    )

    return intermediate_results
end


"""
    save_ss_results(results, filename)

Save steady state results to a MAT file.

# Arguments
- `results`: Dictionary of steady state results (output from ss_ump)
- `filename`: Output filename (e.g., "ss_results_new.mat")
"""
function save_ss_results(results, filename)
    println("Saving results to $filename...")
    matwrite(filename, results)
    println("Results saved successfully.")
end
