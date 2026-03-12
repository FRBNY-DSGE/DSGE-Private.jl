"""
EGM (Endogenous Grid Method) functions for solving household optimization problems.

This code follows Bayer et al. (2018) very closely and implements EGM steps
for the household problem with adjustment and non-adjustment cases.

Copyright (c) 2018-05-27
Christian Bayer, Ralph Lutticke, Lien Pham-Dao, and Volker Tjaden

From: 'Precautionary Savings, Illiquid Assets, and the Aggregate Consequences of
Shocks to Household Income Risk', Bonn mimeo
http://wiwi.uni-bonn.de/hump/wp.html

Translated to Julia for the HANK model replication.
"""

using Statistics
using Interpolations

include("Fastroot.jl")


"""
    EGM_Step1_b(grid, inc, c_n_aux, param, meshes)

Compute optimal consumption and corresponding bond holdings when equity holding
cannot be adjusted (no-adjustment case).

# Arguments
- `grid`: Grid structure with fields `b`, `a`, `nb`, `na`, `nse`
- `inc`: Income structure with fields `labor`, `dividend`, `bond`, `transfer`
- `c_n_aux`: Consumption on endogenous grid (nb × na × nse)
- `param`: Parameter structure with fields `R_cb`, `Rprem`, `pi_cb`, `death_rate`, `b_a_aux`
- `meshes`: Mesh structure with field `b` (bond grid meshes)

# Returns
- `c_update`: Updated consumption policy under no-adjustment (nb × na × nse)
- `b_update`: Updated bond investment policy under no-adjustment (nb × na × nse)

# Notes
- Takes budget constraint into account
- Handles borrowing constraint at grid.b[1]
- Uses interpolation to map from endogenous grid to exogenous grid
"""
function EGM_Step1(grid, inc, c_n_aux, param, meshes)

    # Get grid dimensions as Int
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    
    # Ensure grid["b"] is a 1D vector
    grid_b = vec(grid["b"])

    # Compute initial bond holdings from the budget constraint
    # b' = (c + b - y - d - transfer) / (R/π)
    b_star_n = (c_n_aux .+ meshes["b"] .- inc["labor"] .- inc["dividend"] .- inc["transfer"])

    # Adjust for different interest rates on borrowing vs saving
    # If b_star_n < 0 (borrowing), divide by borrowing rate
    # If b_star_n >= 0 (saving), divide by deposit rate
    b_star_n = (b_star_n .< 0) .* b_star_n ./ ((param["R_cb"] + param["Rprem"]) ./ param["pi_cb"]) .+
               (b_star_n .>= 0) .* b_star_n ./ (param["R_cb"] ./ param["pi_cb"])

    # Adjust for death rate
    b_star_n = (1 - param["death_rate"]) * b_star_n / param["b_a_aux"]

    # Identify where borrowing constraint binds
    # Constraint binds when current bond holdings < optimal bond holdings at lowest grid point
    #binding_constraints = meshes["b"] .< repeat(b_star_n[1, :, :], grid["nb"], 1, 1)

    binding_constraints = meshes["b"] .< repeat(reshape(b_star_n[1, :, :], 1, size(b_star_n, 2), size(b_star_n, 3)), nb_val, 1, 1) 

    # Calculate total resources available
    Resource = inc["labor"] .+ inc["dividend"] .+ inc["bond"] .+ inc["transfer"]
    Resource_aux = reshape(Resource, (nb_val, na_val * nse_val))

    # Reshape for interpolation
    b_star_n = reshape(b_star_n, (nb_val, na_val * nse_val))
    c_n_aux = reshape(c_n_aux, (nb_val, na_val * nse_val))

    # Initialize output arrays
    c_update = zeros(nb_val, na_val * nse_val)
    b_update = zeros(nb_val, na_val * nse_val)

    # Interpolate for each (a, se) combination
    for hh in 1:(na_val * nse_val)
        itp = interpolate((b_star_n[:, hh],), c_n_aux[:, hh], Gridded(Linear()))
        itp_ext = extrapolate(itp, Line())

        c_update[:, hh] .= itp_ext.(grid_b)
        b_update[:, hh] .= Resource_aux[:, hh] .- c_update[:, hh]
    end

    # Reshape back to 3D arrays
    c_update = reshape(c_update, (nb_val, na_val, nse_val))
    b_update = reshape(b_update, (nb_val, na_val, nse_val))

    # Handle binding constraints
    # When constraint binds, consume all resources minus minimum bond holdings
    c_update[binding_constraints] .= Resource[binding_constraints] .- grid_b[1]
    b_update[binding_constraints] .= grid_b[1]

    # No extrapolation: cap at maximum grid value
    b_update[b_update .> grid_b[end]] .= grid_b[end]

    return c_update, b_update
end



