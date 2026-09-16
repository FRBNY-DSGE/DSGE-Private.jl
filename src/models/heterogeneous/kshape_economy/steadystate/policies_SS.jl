"""
    policies_SS(c_a_guess, c_n_guess, psi_guess, AProb, grid, inc, RR, RBRB, P_SE, param, meshes)

Solve for household policies for consumption, bond and equity holdings in the
adjustment and non-adjustment case given returns and transition matrix P_SE.

This code follows Bayer et al. (2018) very closely and uses a variant of EGM to
iterate on the first order conditions until convergence.

# Arguments
- `c_a_guess`: Initial guess for consumption policy (adjustment case) (nb × na × nse)
- `c_n_guess`: Initial guess for consumption policy (no adjustment case) (nb × na × nse)
- `psi_guess`: Initial guess for Lagrange multiplier (nb × na × nse)
- `AProb`: Adjustment probability (nb × na × nse)
- `grid`: Grid structure with fields `b`, `a`, `nb`, `na`, `nse`
- `inc`: Income structure with fields `labor`, `dividend`, `bond`, `capital`, `transfer`
- `RR`: Return on equity (scalar)
- `RBRB`: Return on bonds (scalar)
- `P_SE`: Transition matrix for productivity states (nse × nse)
- `param`: Parameter structure with fields `sigma`, `beta`, `death_rate`, `q`, `b_a_aux`, etc.
- `meshes`: Mesh structure with field `b` (bond grid meshes)

# Returns
- `c_n_new`: Updated consumption policy (no adjustment) (nb × na × nse)
- `b_n_star`: Updated bond policy (no adjustment) (nb × na × nse)
- `c_a_new`: Updated consumption policy (adjustment) (nb × na × nse)
- `b_a_star`: Updated bond policy (adjustment) (nb × na × nse)
- `a_a_star`: Updated equity policy (adjustment) (nb × na × nse)
- `psi_new`: Updated Lagrange multiplier (nb × na × nse)
- `dist_PFI`: Final convergence distance

# Notes
- Iterates until convergence of consumption and psi policies
- Uses EGM steps 1-4 to update policies

Copyright (c) 2018-05-27
Christian Bayer, Ralph Lutticke, Lien Pham-Dao, and Volker Tjaden

From: 'Precautionary Savings, Illiquid Assets, and the Aggregate Consequences of
Shocks to Household Income Risk', Bonn mimeo
http://wiwi.uni-bonn.de/hump/wp.html

Translated to Julia for the HANK model replication.
"""
function policies_SS(c_a_guess, c_n_guess, psi_guess, AProb, grid, inc, RR, RBRB, P_SE, param, meshes)

    # Initialize distances
    dist_c_n = 1000.0
    dist_PSI = 1000.0
    dist_c_a = 1000.0

    # Compute marginal utilities
    mutil_c_n = c_n_guess.^(-param["sigma"])                    # marginal utility of consumption (no adj)
    mutil_c_a = c_a_guess.^(-param["sigma"])                    # marginal utility of consumption (adj)
    mutil_c = AProb .* mutil_c_a .+ (1 .- AProb) .* mutil_c_n  # expected marginal utility of consumption

    # Iteration parameters
    max_iter_PFI = 10000
    tol_PFI = 1e-10

    iter_PFI = 0
    dist_PFI = 1000.0

    # Adjust returns for death rate
    RR = RR + (param["death_rate"] / (1 - param["death_rate"])) * param["q"]
    RBRB = RBRB / (1 - param["death_rate"]) * param["b_a_aux"]

    # Initialize output variables
    c_n_new = similar(c_n_guess)
    b_n_star = zeros(size(c_n_guess))
    c_a_new = similar(c_a_guess)
    b_a_star = zeros(size(c_a_guess))
    a_a_star = zeros(size(c_a_guess))
    psi_new = similar(psi_guess)

    # Iterate until convergence
    while maximum([dist_c_n, dist_c_a, dist_PSI]) > tol_PFI && iter_PFI < max_iter_PFI
        iter_PFI += 1

        ## Step 1: Update policies for no adjustment case (only adjust bond investment)
        mutil_c = RBRB .* mutil_c  # take (real) bond return into account
        nb_val = Int(grid["nb"])
        na_val = Int(grid["na"])
        nse_val = Int(grid["nse"])
        grid_b = vec(grid["b"])  # Ensure grid["b"] is a 1D vector
        aux = reshape(permutedims(mutil_c, (3, 1, 2)), (nse_val, nb_val * na_val))

        # Take expectations
        EMU_aux = param["beta"] * (1 - param["death_rate"]) *
                  permutedims(reshape(P_SE * aux, (nse_val, nb_val, na_val)), (2, 3, 1))

        # Consumption on an endogenous grid
        c_n_aux = 1 ./ (EMU_aux.^(1 / param["sigma"]))

        # Take the borrowing constraint into account
        c_n_new, b_n_star = EGM_Step1(grid, inc, c_n_aux, param, meshes)
        b_n_star[b_n_star .> grid_b[end]] .= grid_b[end]  # no extrapolation

        ## Step 2: Find optimal portfolio composition for every k on grid, i.e. b*(k')
        b_a_star_aux = EGM_Step2(mutil_c_n, mutil_c_a, psi_guess, grid, P_SE, RBRB, RR, AProb, param)

        ## Step 3: Solve for initial resources / consumption in adjustment case
        _, _, cons_list, res_list, b_list, a_list = EGM_Step3(EMU_aux, grid, inc, b_a_star_aux, c_n_aux, param)

        ## Step 4: Interpolate Consumption policy
        c_a_new, b_a_star, a_a_star = EGM_Step4(cons_list, res_list, b_list, a_list, inc, param, grid)

        a_a_star[a_a_star .> grid["a"][end]] .= grid["a"][end]
        b_a_star[b_a_star .> grid_b[end]] .= grid_b[end]

        ## Step 5: Update psi
        mutil_c_n = 1 ./ (c_n_new.^(param["sigma"]))  # marginal utility of consumption (no adj)
        mutil_c_a = 1 ./ (c_a_new.^(param["sigma"]))  # marginal utility of consumption (adj)
        mutil_c_n[mutil_c_n .== Inf] .= 1e16
        mutil_c_a[mutil_c_a .== Inf] .= 1e16
        mutil_c = AProb .* mutil_c_a .+ (1 .- AProb) .* mutil_c_n  # expected marginal utility of consumption

        ## VFI analogue in updating psi
        term1 = ((AProb .* mutil_c_a .* (param["q"] + RR)) .+
                 ((1 .- AProb) .* mutil_c_n .* RR) .+
                 (1 .- AProb) .* psi_guess)
        aux = reshape(permutedims(term1, (3, 1, 2)), (nse_val, nb_val * na_val))

        E_rhs_psi = param["beta"] * (1 - param["death_rate"]) *
                    permutedims(reshape(P_SE * aux, (nse_val, nb_val, na_val)), (2, 3, 1))
        E_rhs_psi = reshape(E_rhs_psi, (nb_val, na_val * nse_val))

        b_n_star_reshaped = reshape(b_n_star, (nb_val, na_val * nse_val))

        # Interpolation of psi-function at b*_n(b,a)
        # Find indices on grid next smallest to optimal policy
        idx = zeros(Int, size(b_n_star_reshaped))
        for i in eachindex(b_n_star_reshaped)
            idx[i] = searchsortedlast(grid_b, b_n_star_reshaped[i])
        end
        idx[b_n_star_reshaped .<= grid_b[1]] .= 1           # take into account the boundaries
        idx[b_n_star_reshaped .>= grid_b[end]] .= nb_val - 1  # take into account the boundaries

        step = diff(grid_b)
        s = (b_n_star_reshaped .- grid_b[idx]) ./ step[idx]  # slope at the optimal policy to next grid point

        aux_index = ones(nb_val, 1) * (0:(na_val * nse_val - 1))' * nb_val  # aux for linear indices
        aux3 = [E_rhs_psi[idx[i] + Int(aux_index[i])] for i in eachindex(idx)]

        # Linear interpolation
        psi_new = aux3 .+ s[:] .* ([E_rhs_psi[idx[i] + Int(aux_index[i]) + 1] for i in eachindex(idx)] .- aux3)
        psi_new = reshape(psi_new, (nb_val, na_val, nse_val))
        b_n_star = reshape(b_n_star_reshaped, (nb_val, na_val, nse_val))
        dist_PSI = maximum(abs.(psi_guess[:] .- psi_new[:]))

        ## Step 6: Check convergence of policies
        dist_c_n = maximum(abs.(c_n_guess[:] .- c_n_new[:]))
        dist_c_a = maximum(abs.(c_a_guess[:] .- c_a_new[:]))

        dist_PFI = maximum([dist_c_n, dist_c_a, dist_PSI])

        # Update c policy guesses
        c_n_guess = c_n_new
        c_a_guess = c_a_new
        psi_guess = psi_new
    end

    return c_n_new, b_n_star, c_a_new, b_a_star, a_a_star, psi_new, dist_PFI
end
