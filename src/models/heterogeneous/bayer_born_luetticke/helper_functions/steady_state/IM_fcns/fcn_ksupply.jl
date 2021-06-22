"""
```
Ksupply(RB_guess, R_guess, m, Vm, Vk,
        distr, inc, eff_int; verbose = :none,
        coarse = false, parallel = false) where {T <: Real}
```
Calculate the aggregate savings when households face idiosyncratic income risk.

Idiosyncratic state is tuple ``(m,k,y)``, where
``m``: liquid assets, ``k``: illiquid assets, ``y``: labor income

### Inputs
- `R_guess::T`: real interest rate illiquid assets
- `RB_guess::T`: nominal rate on liquid assets
- `m::BayerBornLuetticke`: DSGE model object
- `Vm::AbstractArray`: guess for marginal value function to liquid assets
- `Vk::AbstractArray`: guess for marginal value function to illiquid assets
- `distr::AbstractArray`: guess for distribution over idiosyncratic states
- `inc::AbstractArray`: Vector of different types of income in every idiosyncratic state
- `eff_int::AbstractArray`: effective interest rate on liquid assets

where `T <: Real`

### Keyword Arguments
- `verbose::Symbol = :none`: verbosity of print statements with values allowed among `[:none, :low, :high]`.
- `coarse::Bool = false`: use a coarse grid when true.

### Outputs
- `K::T`,`B::T`: aggregate saving in illiquid (`K`) and liquid (`B`) assets
-  `TransitionMat`,`TransitionMat_a`,`TransitionMat_n`: `sparse` transition matrices
    (average, with [`a`] or without [`n`] adjustment of illiquid asset)
- `distr::Array{Float64,3}`: ergodic steady state of `TransitionMat`
- `c_a_star`,`m_a_star`,`k_a_star`,`c_n_star`,`m_n_star`: optimal policies for
    consumption [`c`], liquid [`m`] and illiquid [`k`] asset, with [`a`] or
    without [`n`] adjustment of illiquid asset. All policies have type `Array{Float64,3}
- `V_m::Array{Float64,3}`,`V_kArray{Float64,3}`: marginal value functions
"""

function Ksupply(RB_guess::T, R_guess::T, grids::OrderedDict, θ::NamedTuple, Vm::AbstractArray, Vk::AbstractArray,
                 distr_guess::AbstractArray, inc::AbstractArray, eff_int::AbstractArray,
                 Vm_new::AbstractArray, Vk_new::AbstractArray, m_a_star::AbstractArray, m_n_star::AbstractArray,
                 k_a_star::AbstractArray, c_a_star::AbstractArray, c_n_star::AbstractArray;
                 verbose::Symbol = :none, coarse::Bool = false, parallel::Bool = false,
                 ϵ::Float64 = 1e-5, max_value_function_iters::Int = 1000, n_direct_transition_iters::Int = 10000,
                 kfe_method::Symbol = :krylov) where {T <: Real}

    ## Set up
    # initialize distance variables
    dist                = 9999.0
    dist1               = dist
    dist2               = dist
    Π                   = grids[:Π]::Matrix{T}                   # type declarations necessary b/c grids is an OrderedDict =>
    m_grid              = get_gridpts(grids, :m_grid)::Vector{T} # ensures type stability, or else unnecessary allocations are made
    k_grid              = get_gridpts(grids, :k_grid)::Vector{T}
    y_grid              = get_gridpts(grids, :y_grid)::Vector{T}
    m_ndgrid            = grids[:m_ndgrid]::Array{T, 3}
    k_ndgrid            = grids[:k_ndgrid]::Array{T, 3}
    q                   = 1.0                                    # price of Capital

    #----------------------------------------------------------------------------
    # Iterate over consumption policies
    #----------------------------------------------------------------------------
    count               = 0
    n                   = size(Vm)

    # containers for policies, initialized here
    inv_mutil_old = similar(Vm)
    inv_mutil_new = similar(Vm_new)
    println("EGM Loop")
    @time begin
    while dist > ϵ && count < max_value_function_iters # Iterate consumption policies until convergence
        count          += 1

        # Take expectations for labor income change
        joined_mk_dims  = (n[1] * n[2], n[3])
        EVm             = reshape((reshape(eff_int, joined_mk_dims) .*   # Note that this allocates a new matrix,
                                   reshape(Vm, joined_mk_dims)) * Π', n) # so EVm and EVk are separate from Vm and Vk
        EVk             = reshape(reshape(Vk, joined_mk_dims) * Π', n)   # Also note, Π has dims (n[3], n[3]), hence the reshapes

        # Policy update step: changes EVm, c_a_star, m_a_star, k_a_star, c_n_star, m_n_star
        # Note that EVm is not used in the remainder of the loop, so we overwrite it
        # to stop some calculations from making extra allocations
        EGM_policyupdate!(EVm, EVk, q, θ[:π], RB_guess, 1.0, inc, θ, grids, false,
                          c_a_star, m_a_star, k_a_star, c_n_star, m_n_star; parallel = parallel)

        # marginal value update step: updates Vm_new and Vk_new
        updateV!(Vm_new, Vk_new, EVk, c_a_star, c_n_star, m_n_star,
                 R_guess - 1.0, q, θ, m_grid, Π; parallel = parallel)

        # Calculate distance in updates
        dist1           = maximum(abs, _bbl_invmutil!(inv_mutil_new, Vk_new, θ[:ξ]) - _bbl_invmutil!(inv_mutil_old, Vk, θ[:ξ]))
        dist2           = maximum(abs, _bbl_invmutil!(inv_mutil_new, Vm_new, θ[:ξ]) - _bbl_invmutil!(inv_mutil_old, Vm, θ[:ξ]))
        dist            = max(dist1, dist2) # distance of old and new policy

        # update policy guess/marginal values of liquid/illiquid assets
        # We use .= to overwrite Vm and Vk. Since Vm_new and Vk_new are overwritten by updateV!,
        # we need to copy them, but so Vm = Vm_new won't work. It's also slower to do
        # Vm = copy(Vm_new) since this creates a new allocation. Running Vm .= Vm_new
        # avoids allocating a new array for Vm.
        Vm             .= Vm_new
        Vk             .= Vk_new
    end

    if verbose == :high
        println("Max abs error after completing EGM iterations = $(dist)")
    end
    end
    println("Solving KFE")
    @time begin
    #------------------------------------------------------
    # Find stationary distribution (Is direct transition better for large model?)
    # Expensiveness on coarse grid (ny = 6) => .01 s for making transition matrix,
    #                                       => 0.3 s for calling KrylovKit
    #------------------------------------------------------
        n_total_dims = prod(n) # total number of dimensions, used to construct transition matrix for Krylov methods

        #if kfe_method == :slepc
            #distr = _slepc_solve_kfe(m_a_star, m_n_star, k_a_star, Π, n, n_total_dims, m_grid, k_grid, y_grid, θ, parallel)
        #else
            if kfe_method == :krylov
                # Define transition matrix
                S_a, T_a, W_a, S_n, T_n, W_n = MakeTransition(m_a_star,  m_n_star, k_a_star, Π, n, m_grid, k_grid, y_grid;
                                                              parallel = parallel)
                #=        TransitionMat_a              = sparse(S_a, T_a, W_a, n_total_dims, n_total_dims) # original code
                TransitionMat_n              = sparse(S_n, T_n, W_n, n_total_dims, n_total_dims)=#
                TransitionMat_a              = sparse(T_a, S_a, W_a, n_total_dims, n_total_dims) # but we construct it this way
                TransitionMat_n              = sparse(T_n, S_n, W_n, n_total_dims, n_total_dims) # to avoid applying a transpose
                TransitionMat                = θ[:λ] .* TransitionMat_a + (1.0 - θ[:λ]) .* TransitionMat_n

                # Calculate left-hand unit eigenvector (uses KrylovKit.jl)
                # aux   = real.(eigsolve(TransitionMat', 1)[2][1]) # original code
                distr   = real.(eigsolve(TransitionMat, 1)[2][1]) # but since we construct TransitionMat_a, TransitionMat_n as their transposes,
                distr ./= sum(distr)                              # we don't need to call eigsolve on TransitionMat'
                distr   = reshape(distr, n)
            elseif kfe_method == :direct
                # Direct Transition
                distr_guess .= 1 ./ prod(n) # uniform distribution guess provides most robust convergence rather than using previous distribution
                distr, dist, count = MultipleDirectTransition!(m_a_star, m_n_star, k_a_star, distr_guess, θ[:λ], Π,
                                                               n, m_grid, k_grid, y_grid, ϵ;
                                                               iters = n_direct_transition_iters)
            else
                error("Solution method for Kolmogorov forward equation $(kfe_method) is not recognized. " *
                      "Available methods are [:krylov, :direct]")
            end
        #end
    #end
    #-----------------------------------------------------------------------------
    # Calculate capital stock
    #-----------------------------------------------------------------------------
    K = dot(distr, k_ndgrid) # faster to use dot
    B = dot(distr, m_ndgrid)

    return K, B, c_a_star, m_a_star, k_a_star, c_n_star, m_n_star, Vm, Vk, distr
end
#=
function _slepc_solve_kfe(m_a_star::AbstractArray, m_n_star::AbstractArray, k_a_star::AbstractArray,
                          Π::AbstractMatrix, n::NTuple{3,Int}, n_total_dims::Int,
                          m_grid::AbstractVector, k_grid::AbstractVector, y_grid::AbstractVector,
                          θ::NamedTuple, parallel::Bool)

    # Calculate locations and values for creating the transition matrix
    S_a, T_a, W_a, S_n, T_n, W_n = MakeTransition(m_a_star,  m_n_star, k_a_star, Π, n, m_grid, k_grid, y_grid;
                                                  parallel = parallel)

    #            SlepcInitialize()

    # Update the weights to account for the probability of being able to adjust portfolios
    W_a .*= θ[:λ]
    W_n .*= 1. - θ[:λ]

    # Iterate through the indices to find matches
    ST_a = [(i, j) for (i, j) in zip(S_a, T_a)]
    ST_n = [(i, j) for (i, j) in zip(S_n, T_n)]
    ST_unique = unique(vcat(ST_a, ST_n))
    ST_inter = intersect(ST_a, ST_n)
    W_join = similar(W_a, length(ST_unique))
    ctr_a = 1
    ctr_n = 1
    ctr_inter = 1
    for i in eachindex(W_join)
        if ST_unique[i] == ST_inter[ctr_inter]
            W_join[i] = W_a[ctr_a] + W_n[ctr_n]
            ctr_inter < length(ST_inter) && (ctr_inter += 1)
            ctr_a < length(ST_a) && (ctr_a += 1)
            ctr_n < length(ST_n) && (ctr_n += 1)
        elseif ST_unique[i] == ST_a[ctr_a]
            W_join[i] = W_a[ctr_a]
            ctr_a < length(ST_a) && (ctr_a += 1)
        else
            W_join[i] = W_n[ctr_n]
            ctr_n < length(ST_n) && (ctr_n += 1)
        end
    end

    # Create the problem matrices, set sizes, and apply "command-line" options.
    TransitionMat = MatCreate()
    MatSetSizes(TransitionMat, PETSC_DECIDE, PETSC_DECIDE, n_total_dims, n_total_dims)
    MatSetFromOptions(TransitionMat)
    MatSetUp(TransitionMat)
    #=            TransitionMat_a = MatCreate()
    MatSetSizes(TransitionMat_a, PETSC_DECIDE, PETSC_DECIDE, n_total_dims, n_total_dims)
    MatSetFromOptions(TransitionMat_a)
    MatSetUp(TransitionMat_a)
    TransitionMat_n = MatCreate()
    MatSetSizes(TransitionMat_n, PETSC_DECIDE, PETSC_DECIDE, n_total_dims, n_total_dims)
    MatSetFromOptions(TransitionMat_n)
    MatSetUp(TransitionMat_n)=#

    # Get rows handled by the local processor for the adjust and no-adjust transition matrices
    TransitionMat_rstart, TransitionMat_rend = MatGetOwnershipRange(TransitionMat)
    #=            TransitionMat_a_rstart, TransitionMat_a_rend = MatGetOwnershipRange(TransitionMat_a)
    TransitionMat_n_rstart, TransitionMat_n_rend = MatGetOwnershipRange(TransitionMat_n)=#

    for i in eachindex(W_join)
        MatSetValues(TransitionMat, [ST_unique[i][2] - 1], [ST_unique[i][1] - 1], [W_join[i]], INSERT_VALUES)
    end
    #=            for i in eachindex(W_a)
    # arg 2 = row index (PETSC is in C, so it's zero-based indexing => decrement row by 1)
    # arg 3 = col index (PETSC is in C, so it's zero-based indexing => decrement col by 1)
    # arg 4 = value
    MatSetValues(TransitionMat_a, [T_a[i] - 1], [S_a[i] - 1], [W_a[i]], INSERT_VALUES)
    end
    for i in eachindex(W_n)
    MatSetValues(TransitionMat_n, [T_n[i] - 1], [S_n[i] - 1], [W_n[i]], INSERT_VALUES)
    end=#

    # Add matrices to create TransitionMat
    #=            TransitionMat = MatCreate()
    MatCreateComposite(2, [TransitionMat_a, TransitionMat_n], TransitionMat)=#

    # Assemble matrix
    MatAssemblyBegin(TransitionMat, MAT_FINAL_ASSEMBLY)
    MatAssemblyEnd(TransitionMat, MAT_FINAL_ASSEMBLY)

    # Set up eigenvalue solver. By default, it calculates the largest eigenvalue and
    # the associated eigenvector using a Krylov-Schur algorithm (like KrylovKit.jl)
    eps = EPSCreate()
    EPSSetOperators(eps, TransitionMat) # set B matrix to NULL => solve Ax = λx, not Ax = λBx
    EPSSetFromOptions(eps)
    EPSSetUp(eps)
    EPSSolve(eps)

    # Retrieve largest eigenvector's real part
    vecr, veci = MatCreateVecs(TransitionMat)
    # aux, aux_ref = VecGetArray(EPSGetEigenvector(eps, 0, vecr, veci)[1]) # only want the real part
    aux, aux_ref = VecGetArray(EPSGetEigenpair(eps, 0, vecr, veci)[3]) # only want the real part
    distr = reshape(vec(aux) ./ sum(aux), n)
    VecRestoreArray(vecr, aux_ref)

    # Free Memory
    #=            MatDestroy(TransitionMat_a)
    MatDestroy(TransitionMat_n)=#
    MatDestroy(TransitionMat)
    VecDestroy(vecr)
    VecDestroy(veci)
    EPSDestroy(eps)

    # Stop Slepc
    #            SlepcFinalize()

    return distr
end=#
