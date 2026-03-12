"""
    find_dist(mu_dist, H, AProb, b_n_star, b_a_star, a_a_star, H_tilde, P_SE_dist, param, grid)

Compute the stationary distribution mu_dist using transition matrices.

This function constructs transition matrices for three cases:
- Adjustment case (H_a): when households adjust both bonds and equity
- Non-adjustment case (H_n): when households only adjust bonds
- Death case (H_die): when households die and reset to zero holdings

Then combines them and iterates to find the stationary distribution.

# Arguments
- `mu_dist`: Initial distribution guess (will be updated)
- `H`: Transition matrix (input, but will be recomputed)
- `AProb`: Adjustment probability (nb × na × nse)
- `b_n_star`: Optimal bond holdings in non-adjustment case (nb × na × nse)
- `b_a_star`: Optimal bond holdings in adjustment case (nb × na × nse)
- `a_a_star`: Optimal equity holdings in adjustment case (nb × na × nse)
- `H_tilde`: Transition matrix for employment states
- `P_SE_dist`: Transition probabilities for employment states
- `param`: Parameter structure with field `death_rate`
- `grid`: Grid structure with fields `nb`, `na`, `nse`

# Returns
Tuple of 3 outputs:
- `mu_dist`: Stationary distribution (1D vector, length nb*na*nse)
- `dist_mu`: Final distance metric (convergence measure)
- `H`: Combined transition matrix (sparse, nb*na*nse × nb*na*nse)

# Notes
- Assumes all inputs (param, grid) are Dicts with String keys
- Uses sparse matrices for efficiency
- Iterates until convergence (dist_mu < 1e-14) or max iterations (150000)
"""

using SparseArrays
include("helpers/genweight.jl")
include("helpers/sub2ind.jl")

function find_dist(mu_dist, H, AProb, b_n_star, b_a_star, a_a_star, H_tilde, P_SE_dist, param, grid)
    
    # Get grid dimensions - direct Dict access
    nb_val = Int(grid["nb"])
    na_val = Int(grid["na"])
    nse_val = Int(grid["nse"])
    
    # Initialize matrices
    weight11 = zeros(nb_val * na_val, nse_val, nse_val)
    weight12 = zeros(nb_val * na_val, nse_val, nse_val)
    weight21 = zeros(nb_val * na_val, nse_val, nse_val)
    weight22 = zeros(nb_val * na_val, nse_val, nse_val)
    
    # Get grid arrays
    grid_b = vec(grid["b"])
    grid_a = vec(grid["a"])
    
    # Find next smallest on-grid value for money and capital choices
    Dist_b_a, idb_a = genweight(b_a_star, grid_b)
    Dist_b_n, idb_n = genweight(b_n_star, grid_b)
    Dist_a, ida_a = genweight(a_a_star, grid_a)
    
    # ida_n = repmat(ones(grid.nb,1)*(1:grid.na),[1 1 grid.nse])
    ida_n = repeat((1:na_val)', outer=(nb_val, 1, nse_val))
    
    # Death case: zeros for all holdings
    Dist_b_die, idb_die = genweight(zeros(nb_val, na_val, nse_val), grid_b)
    Dist_a_die, ida_die = genweight(zeros(nb_val, na_val, nse_val), grid_a)
    
    # === Transition matrix adjustment case ===
    # idb_a = repmat(idb_a(:),[1 grid.nse])
    idb_a = repeat(vec(idb_a), outer=(1, nse_val))
    # ida_a = repmat(ida_a(:),[1 grid.nse])
    ida_a = repeat(vec(ida_a), outer=(1, nse_val))
    # ids = kron(1:grid.nse,ones(1,grid.nb*grid.na*grid.nse))
    # In MATLAB: kron creates [1,1,...,1, 2,2,...,2, ..., nse,nse,...,nse] with each repeated nb*na*nse times
    # kron needs vectors, so convert ones(1,N) to a row vector properly
    ids = vec(kron(collect(1:nse_val), vec(ones(Int, 1, nb_val * na_val * nse_val))))
    
    # idb_a = idb_a(:)
    idb_a = vec(idb_a)
    # ida_a = ida_a(:)
    ida_a = vec(ida_a)
    # ids = ids(:) - already a vector
    
    # Compute indices using sub2ind
    index11 = sub2ind([nb_val, na_val, nse_val], idb_a, ida_a, ids)
    index12 = sub2ind([nb_val, na_val, nse_val], idb_a, ida_a .+ 1, ids)
    index21 = sub2ind([nb_val, na_val, nse_val], idb_a .+ 1, ida_a, ids)
    index22 = sub2ind([nb_val, na_val, nse_val], idb_a .+ 1, ida_a .+ 1, ids)
    
    for hh = 1:nse_val
        # Corresponding weights
        weight11_aux = (1 .- Dist_b_a[:, :, hh]) .* (1 .- Dist_a[:, :, hh])
        weight12_aux = (1 .- Dist_b_a[:, :, hh]) .* Dist_a[:, :, hh]
        weight21_aux = Dist_b_a[:, :, hh] .* (1 .- Dist_a[:, :, hh])
        weight22_aux = Dist_b_a[:, :, hh] .* Dist_a[:, :, hh]
        
        # Dimensions (mxk,h',h)
        weight11[:, :, hh] = vec(weight11_aux) * P_SE_dist[hh, :]'
        weight12[:, :, hh] = vec(weight12_aux) * P_SE_dist[hh, :]'
        weight21[:, :, hh] = vec(weight21_aux) * P_SE_dist[hh, :]'
        weight22[:, :, hh] = vec(weight22_aux) * P_SE_dist[hh, :]'
    end
    
    # Dimensions (mxk,h,h')
    weight11 = permutedims(weight11, [1, 3, 2])
    weight22 = permutedims(weight22, [1, 3, 2])
    weight12 = permutedims(weight12, [1, 3, 2])
    weight21 = permutedims(weight21, [1, 3, 2])
    
    # rowindex = repmat(1:grid.nb*grid.na*grid.nse,[1 4*grid.nse])
    rowindex = repeat(collect(1:nb_val * na_val * nse_val), outer=(1, 4 * nse_val))
    
    # H_a = sparse(rowindex(:),[index11(:); index21(:); index12(:); index22(:)],...
    #     [weight11(:); weight21(:); weight12(:); weight22(:)],grid.nb*grid.na*grid.nse,grid.nb*grid.na*grid.nse)
    H_a = sparse(
        vec(rowindex),
        [vec(index11); vec(index21); vec(index12); vec(index22)],
        [vec(weight11); vec(weight21); vec(weight12); vec(weight22)],
        nb_val * na_val * nse_val,
        nb_val * na_val * nse_val
    )
    
    # === Policy Transition Matrix for no-adjustment case ===
    weight11 = zeros(nb_val * na_val, nse_val, nse_val)
    weight21 = zeros(nb_val * na_val, nse_val, nse_val)
    
    idb_n = repeat(vec(idb_n), outer=(1, nse_val))
    ida_n = repeat(vec(ida_n), outer=(1, nse_val))
    
    index11 = sub2ind([nb_val, na_val, nse_val], vec(idb_n), vec(ida_n), ids)
    index21 = sub2ind([nb_val, na_val, nse_val], vec(idb_n) .+ 1, vec(ida_n), ids)
    
    for hh = 1:nse_val
        # Corresponding weights
        weight21_aux = Dist_b_n[:, :, hh]
        weight11_aux = 1 .- Dist_b_n[:, :, hh]
        
        weight21[:, :, hh] = vec(weight21_aux) * P_SE_dist[hh, :]'
        weight11[:, :, hh] = vec(weight11_aux) * P_SE_dist[hh, :]'
    end
    
    weight11 = permutedims(weight11, [1, 3, 2])
    weight21 = permutedims(weight21, [1, 3, 2])
    
    # rowindex = repmat(1:grid.nb*grid.na*grid.nse,[1 2*grid.nse])
    rowindex = repeat(collect(1:nb_val * na_val * nse_val), outer=(1, 2 * nse_val))
    
    # H_n = sparse(rowindex,[index11(:); index21(:)],...
    #     [weight11(:); weight21(:)],grid.nb*grid.na*grid.nse,grid.nb*grid.na*grid.nse)
    H_n = sparse(
        vec(rowindex),
        [vec(index11); vec(index21)],
        [vec(weight11); vec(weight21)],
        nb_val * na_val * nse_val,
        nb_val * na_val * nse_val
    )
    
    # === Transition matrix die ===
    idb_die = repeat(vec(idb_die), outer=(1, nse_val))
    ida_die = repeat(vec(ida_die), outer=(1, nse_val))
    # ids_die = kron(1:grid.nse,ones(1,grid.nb*grid.na*grid.nse))
    ids_die = vec(kron(collect(1:nse_val), vec(ones(Int, 1, nb_val * na_val * nse_val))))
    
    index11 = sub2ind([nb_val, na_val, nse_val], vec(idb_die), vec(ida_die), vec(ids_die))
    index12 = sub2ind([nb_val, na_val, nse_val], vec(idb_die), vec(ida_die) .+ 1, vec(ids_die))
    index21 = sub2ind([nb_val, na_val, nse_val], vec(idb_die) .+ 1, vec(ida_die), vec(ids_die))
    index22 = sub2ind([nb_val, na_val, nse_val], vec(idb_die) .+ 1, vec(ida_die) .+ 1, vec(ids_die))
    
    for hh = 1:nse_val
        # Corresponding weights
        weight11_aux = (1 .- Dist_b_die[:, :, hh]) .* (1 .- Dist_a_die[:, :, hh])
        weight12_aux = (1 .- Dist_b_die[:, :, hh]) .* Dist_a_die[:, :, hh]
        weight21_aux = Dist_b_die[:, :, hh] .* (1 .- Dist_a_die[:, :, hh])
        weight22_aux = Dist_b_die[:, :, hh] .* Dist_a_die[:, :, hh]
        
        # Dimensions (mxk,h',h)
        weight11[:, :, hh] = vec(weight11_aux) * P_SE_dist[hh, :]'
        weight12[:, :, hh] = vec(weight12_aux) * P_SE_dist[hh, :]'
        weight21[:, :, hh] = vec(weight21_aux) * P_SE_dist[hh, :]'
        weight22[:, :, hh] = vec(weight22_aux) * P_SE_dist[hh, :]'
    end
    
    # Dimensions (mxk,h,h')
    weight11 = permutedims(weight11, [1, 3, 2])
    weight22 = permutedims(weight22, [1, 3, 2])
    weight12 = permutedims(weight12, [1, 3, 2])
    weight21 = permutedims(weight21, [1, 3, 2])
    
    # rowindex = repmat(1:grid.nb*grid.na*grid.nse,[1 4*grid.nse])
    rowindex = repeat(collect(1:nb_val * na_val * nse_val), outer=(1, 4 * nse_val))
    
    # H_die = sparse(rowindex(:),[index11(:); index21(:); index12(:); index22(:)],...
    #     [weight11(:); weight21(:); weight12(:); weight22(:)],grid.nb*grid.na*grid.nse,grid.nb*grid.na*grid.nse)
    H_die = sparse(
        vec(rowindex),
        [vec(index11); vec(index21); vec(index12); vec(index22)],
        [vec(weight11); vec(weight21); vec(weight12); vec(weight22)],
        nb_val * na_val * nse_val,
        nb_val * na_val * nse_val
    )
    
    # === Joint transition matrix and transitions ===
    AProb_vec = vec(AProb)
    n_elements = length(AProb_vec)
    
    # APD_a = sparse(1:length(AProb(:)),1:length(AProb(:)),AProb(:))
    APD_a = sparse(collect(1:n_elements), collect(1:n_elements), AProb_vec, n_elements, n_elements)
    
    # APD_n = sparse(1:length(AProb(:)),1:length(AProb(:)),1-AProb(:))
    APD_n = sparse(collect(1:n_elements), collect(1:n_elements), 1 .- AProb_vec, n_elements, n_elements)
    
    # H = (1-param.death_rate)*(APD_a*H_a + APD_n*H_n) + param.death_rate*H_die
    death_rate = param["death_rate"]
    H = (1 - death_rate) * (APD_a * H_a + APD_n * H_n) + death_rate * H_die
    
    # H = (H_tilde')*H
    H = H_tilde' * H
    
    # === Iterate to find stationary distribution ===
    dist_mu = 1000.0
    iter_dist = 0
    
    # mu_dist = mu_dist(:)'
    mu_dist = vec(mu_dist)'  # Row vector (1×n)
    
    # mu_dist_next = mu_dist*H
    mu_dist_next = mu_dist * H
    
    # mu_dist_next = full(mu_dist_next)
    # Convert Adjoint (row vector) to regular row vector (keep as row vector!)
    mu_dist_next = collect(mu_dist_next)  # Keep as row vector, just convert sparse to dense
    
    # mu_dist_next = mu_dist_next./sum(mu_dist_next)
    mu_dist_next = mu_dist_next ./ sum(mu_dist_next)
    
    # dist_mu = max((abs(mu_dist_next(:)-mu_dist(:))))
    dist_mu = maximum(abs.(vec(mu_dist_next) .- vec(mu_dist)))
    
    # while dist_mu > 1e-14 && iter_dist < 150000
    while dist_mu > 1e-14 && iter_dist < 150000
        mu_dist_next = mu_dist * H
        # Convert sparse result to dense row vector more efficiently
        if isa(mu_dist_next, SparseVector) || isa(mu_dist_next, Adjoint)
            mu_dist_next = vec(mu_dist_next)'
        else
            mu_dist_next = collect(mu_dist_next)
        end
        mu_dist_next = mu_dist_next ./ sum(mu_dist_next)
        
        dist_mu = maximum(abs.(vec(mu_dist_next) .- vec(mu_dist)))
        
        iter_dist = iter_dist + 1
        
        # Print progress every 1000 iterations
        if iter_dist % 1000 == 0
            println("find_dist: iter=$iter_dist, dist_mu=$dist_mu")
        end
        
        mu_dist = mu_dist_next  # Both are row vectors, direct assignment
    end
    
    if iter_dist >= 150000
        @warn "find_dist: Maximum iterations (150000) reached. dist_mu = $dist_mu"
    end
    
    # Return as column vector (1D) - convert row vector to column vector
    mu_dist = vec(mu_dist)
    
    return (mu_dist, dist_mu, H)
end

