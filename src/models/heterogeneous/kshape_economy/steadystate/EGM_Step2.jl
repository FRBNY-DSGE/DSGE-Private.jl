include("Fastroot.jl")

"""
    EGM_Step2(mutil_c_n, mutil_c_a, psi_guess, grid, P_SE, RBRB, RR, AProb, param)

Find optimal portfolio composition (bond holdings b*) for a given capital choice k'
in steady state.

# Arguments
- `mutil_c_n`: Marginal utility of consumption (no adjustment case) (nb × na × nse)
- `mutil_c_a`: Marginal utility of consumption (adjustment case) (nb × na × nse)
- `psi_guess`: Lagrange multiplier guess (nb × na × nse)
- `grid`: Grid structure with fields `b`, `a`, `nb`, `na`, `nse`
- `P_SE`: Transition matrix for productivity states (nse × nse)
- `RBRB`: Adjusted bond return (scalar or array)
- `RR`: Return on capital (scalar or array)
- `AProb`: Adjustment probability (nb × na × nse)
- `param`: Parameter structure with fields `q`, `beta`, `death_rate`

# Returns
- `b_star`: Optimal bond holdings for each (a', s) combination (na × nse)

# Notes
- Uses first-order conditions to find optimal portfolio allocation
- Finds root of Euler equation difference between equity and bonds
- Handles corner solutions at grid boundaries
"""
function EGM_Step2(mutil_c_n, mutil_c_a, psi_guess, grid, P_SE, RBRB, RR, AProb, param)

    # Get grid dimensions as Int
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])

    # Compute derivative of expected value w.r.t. a' (equity)
    # term1 = E[β * (1-death_rate) * (λ * u'(c_a) * (q + R) + (1-λ) * u'(c_n) * R + (1-λ) * ψ)]
    term1 = (AProb .* mutil_c_a .* (param["q"] + RR) .+
             (1 .- AProb) .* mutil_c_n .* RR .+
             (1 .- AProb) .* psi_guess)

    # Reshape for matrix multiplication with transition matrix
    # Permute: (nb, na, nse) -> (nse, nb, na)
    aux = reshape(permutedims(term1, (3, 1, 2)), (nse_val, nb_val * na_val))

    # Take expectations over future productivity states
    # P_SE * aux gives (nse × nb*na)
    term1_expected = P_SE * aux  # (nse × nb*na)

    # Reshape back: (nse, nb*na) -> (nse, nb, na) -> (nb, na, nse)
    term1 = param["beta"] * (1 - param["death_rate"]) *
            permutedims(reshape(term1_expected, (nse_val, nb_val, na_val)), (2, 3, 1))

    # Compute derivative of expected value w.r.t. b' (bonds)
    # term2 = E[β * (1-death_rate) * RBRB * (λ * u'(c_a) + (1-λ) * u'(c_n))]
    term2 = RBRB .* (AProb .* mutil_c_a .+ (1 .- AProb) .* mutil_c_n)

    # Reshape for matrix multiplication
    aux = reshape(permutedims(term2, (3, 1, 2)), (nse_val, nb_val * na_val))

    # Take expectations
    term2_expected = P_SE * aux
    term2 = param["beta"] * (1 - param["death_rate"]) *
            permutedims(reshape(term2_expected, (nse_val, nb_val, na_val)), (2, 3, 1))

    # Optimality condition: derivative of EV w.r.t. a' / q = derivative w.r.t. b'
    # Find where this difference equals zero
    E_return_diff = term1 ./ param["q"] .- term2

    # Find b* for each (a', s) that satisfies the optimality condition
    # Use Fastroot to find where E_return_diff crosses zero
    b_star = Fastroot(grid["b"], E_return_diff)

    # Ensure b* stays within grid bounds
    b_star = max.(b_star, grid["b"][1])
    b_star = min.(b_star, grid["b"][end])

    # Reshape to (na × nse)
    b_star = reshape(b_star, (na_val, nse_val))

    return b_star
end
