using SparseArrays
using LinearAlgebra

include("helpers/genweight.jl")
include("helpers/sub2ind.jl")

"""
    CalValueSS(b_n_star, b_a_star, a_a_star, c_a_star, c_n_star, AProb, P_SE, param, grid)

Compute value functions at steady state for given policies.

# Arguments
- `b_n_star`: Bond policy (no adjustment) (nb × na × nse)
- `b_a_star`: Bond policy (adjustment) (nb × na × nse)
- `a_a_star`: Equity policy (adjustment) (nb × na × nse)
- `c_a_star`: Consumption policy (adjustment) (nb × na × nse)
- `c_n_star`: Consumption policy (no adjustment) (nb × na × nse)
- `AProb`: Adjustment probability (nb × na × nse)
- `P_SE`: Transition matrix for productivity states (nse × nse)
- `param`: Parameter structure
- `grid`: Grid structure

# Returns
- `AProb`: Updated adjustment probability (nb × na × nse)
- `V`: Value function (nb × na × nse)
- `H`: Joint transition matrix (sparse)
- `EU`: Expected utility (optional, if requested)

# Notes
- Uses logit specification for adjustment probability
- Iterates on value function until convergence

Copyright (c) 2017-12-19
Christian Bayer, Ralph Lutticke

From: 'Precautionary Savings, Illiquid Assets, and the Aggregate Consequences of
Shocks to Household Income Risk', Bonn mimeo
http://wiwi.uni-bonn.de/hump/wp.html

Translated to Julia for the HANK model replication.
"""
function CalValueSS(b_n_star, b_a_star, a_a_star, c_a_star, c_n_star, AProb, P_SE, param, grid)

    # Convert grid dimensions to Int
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    
    # Convert grid vectors to 1D vectors
    grid_b = vec(grid["b"])
    grid_a = vec(grid["a"])

    # Initialize matrices
    weight11 = zeros(nb_val * na_val, nse_val, nse_val)
    weight12 = zeros(nb_val * na_val, nse_val, nse_val)
    weight21 = zeros(nb_val * na_val, nse_val, nse_val)
    weight22 = zeros(nb_val * na_val, nse_val, nse_val)

    ## Find next smallest on-grid value for money and capital choices
    Dist_b_a, idb_a = genweight(b_a_star, grid_b)
    Dist_b_n, idb_n = genweight(b_n_star, grid_b)
    Dist_a, ida_a = genweight(a_a_star, grid_a)
    println("After genweight, idb_a shape: ", size(idb_a))
    # Create ida_n: matrix of size (nb, na, nse) where each row is [1, 2, ..., na]
    temp_ida = ones(Int, nb_val) * transpose(1:na_val)  # (nb, na)
    ida_n = repeat(reshape(temp_ida, nb_val, na_val, 1), 1, 1, nse_val)  # (nb, na, nse)

    ## Transition matrix adjustment case
    idb_a = repeat(idb_a[:], 1, nse_val)
    ida_a = repeat(ida_a[:], 1, nse_val)
    println("After repeat, idb_a shape: ", size(idb_a))
    ids = kron(1:nse_val, ones(Int, 1, nb_val * na_val * nse_val))
    ids = sort(vec(ids))

    index11 = sub2ind([nb_val, na_val, nse_val], idb_a[:], ida_a[:], ids[:])
    index12 = sub2ind([nb_val, na_val, nse_val], idb_a[:], ida_a[:] .+ 1, ids[:])
    index21 = sub2ind([nb_val, na_val, nse_val], idb_a[:] .+ 1, ida_a[:], ids[:])
    index22 = sub2ind([nb_val, na_val, nse_val], idb_a[:] .+ 1, ida_a[:] .+ 1, ids[:])

    for hh in 1:nse_val
        # Corresponding weights
        weight11_aux = (1 .- Dist_b_a[:, :, hh]) .* (1 .- Dist_a[:, :, hh])
        weight12_aux = (1 .- Dist_b_a[:, :, hh]) .* Dist_a[:, :, hh]
        weight21_aux = Dist_b_a[:, :, hh] .* (1 .- Dist_a[:, :, hh])
        weight22_aux = Dist_b_a[:, :, hh] .* Dist_a[:, :, hh]

        # Dimensions (nb*na, nse', nse)
        weight11[:, :, hh] = weight11_aux[:] .* P_SE[hh, :]'
        weight12[:, :, hh] = weight12_aux[:] .* P_SE[hh, :]'
        weight21[:, :, hh] = weight21_aux[:] .* P_SE[hh, :]'
        weight22[:, :, hh] = weight22_aux[:] .* P_SE[hh, :]'
    end

    # Dimensions (nb*na, nse, nse')
    weight11 = permutedims(weight11, [1, 3, 2])
    weight22 = permutedims(weight22, [1, 3, 2])
    weight12 = permutedims(weight12, [1, 3, 2])
    weight21 = permutedims(weight21, [1, 3, 2])
    rowindex = repeat(1:(nb_val * na_val * nse_val), 1, 4 * nse_val)

    H_a = sparse(rowindex[:], [index11[:]; index21[:]; index12[:]; index22[:]],
                 [weight11[:]; weight21[:]; weight12[:]; weight22[:]],
                 nb_val * na_val * nse_val, nb_val * na_val * nse_val)

    ## Policy Transition Matrix for no-adjustment case
    weight11 = zeros(nb_val * na_val, nse_val, nse_val)
    weight21 = zeros(nb_val * na_val, nse_val, nse_val)
    idb_n = Int.(repeat(idb_n[:], 1, nse_val))
    ida_n = Int.(repeat(ida_n[:], 1, nse_val))
    index11 = sub2ind([nb_val, na_val, nse_val], Int.(idb_n[:]), Int.(ida_n[:]), Int.(ids[:]))
    index21 = sub2ind([nb_val, na_val, nse_val], Int.(idb_n[:] .+ 1), Int.(ida_n[:]), Int.(ids[:]))

    for hh in 1:nse_val
        # Corresponding weights
        weight21_aux = Dist_b_n[:, :, hh]
        weight11_aux = (1 .- Dist_b_n[:, :, hh])

        weight21[:, :, hh] = weight21_aux[:] * P_SE[hh, :]'
        weight11[:, :, hh] = weight11_aux[:] * P_SE[hh, :]'
    end

    weight11 = permutedims(weight11, [1, 3, 2])
    weight21 = permutedims(weight21, [1, 3, 2])

    rowindex = repeat(1:(nb_val * na_val * nse_val), 1, 2 * nse_val)

    H_n = sparse(rowindex[:], [index11[:]; index21[:]],
                 [weight11[:]; weight21[:]],
                 nb_val * na_val * nse_val, nb_val * na_val * nse_val)

    ## Joint transition matrix and transitions
    APD_a = sparse(1:length(AProb[:]), 1:length(AProb[:]), AProb[:])
    APD_n = sparse(1:length(AProb[:]), 1:length(AProb[:]), 1 .- AProb[:])

    H = APD_a * H_a + APD_n * H_n

    util(c) = (c.^(1 - param["sigma"])) ./ (1 - param["sigma"])

    u = ((1 .- AProb) .* util(c_n_star) .+ AProb .* util(c_a_star))

    distV = 1.0

    # Logit adjustment cost
    # Add small epsilon to avoid log(0)
    AProb_vec = AProb[:]
    p1 = AProb_vec ./ param["nu"]
    p0 = 1 .- p1

    # Handle log(0) cases: set to 0 when probability is 0 (since lim x->0 of x*log(x) = 0)
    term1 = similar(p0)
    term2 = similar(p1)
    for i in eachindex(p0)
        term1[i] = p0[i] > 1e-16 ? p0[i] * log(p0[i]) : 0.0
        term2[i] = p1[i] > 1e-16 ? p1[i] * log(p1[i]) : 0.0
    end

    AC = param["sigma_chi"] .* (term1 .+ term2) .+
         param["mu_chi"] .* p1 .-
         (param["mu_chi"] - param["sigma_chi"] * log(1 + exp(param["mu_chi"] / param["sigma_chi"])))
    AC = AC .* param["nu"]

    V = (sparse(I, length(AC), length(AC)) - param["beta"] * (1 - param["death_rate"]) * H) \ (u[:] - AC)
    iter_V = 1
    while distV > 1e-10 && iter_V < 1000
        iter_V += 1
        Vnew = u[:] + param["beta"] * (1 - param["death_rate"]) .* H * V - AC
        distV = maximum(abs.(Vnew - V))
        V = Vnew
    end

    V_a = util(c_a_star[:]) + param["beta"] * (1 - param["death_rate"]) * H_a * V
    V_n = util(c_n_star[:]) + param["beta"] * (1 - param["death_rate"]) * H_n * V
    V_a = reshape(V_a, (nb_val, na_val, nse_val))
    V_n = reshape(V_n, (nb_val, na_val, nse_val))
    V = reshape(V, (nb_val, na_val, nse_val))

    DV = V_a - V_n

    # Logit adjustment probability
    AProb = 1 ./ (1 .+ exp.(.-((DV) .- param["mu_chi"]) ./ param["sigma_chi"]))

    AProb = param["nu"] .* max.(min.(AProb, 1 - 1e-6), 1e-6)

    AProb = reshape(AProb, (nb_val, na_val, nse_val))

    return AProb, V, H
end
