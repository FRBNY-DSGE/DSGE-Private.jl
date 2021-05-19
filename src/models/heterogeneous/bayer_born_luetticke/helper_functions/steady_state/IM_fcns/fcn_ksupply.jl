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
    # Define transition matrix
    S_a, T_a, W_a, S_n, T_n, W_n    = MakeTransition(m_a_star,  m_n_star, k_a_star, Π, n, m_grid, k_grid, y_grid;
                                                     parallel = parallel)
    TransitionMat_a                 = sparse(S_a, T_a, W_a, prod(n), prod(n))
    TransitionMat_n                 = sparse(S_n, T_n, W_n, prod(n), prod(n))
    TransitionMat                   = θ[:λ] .* TransitionMat_a + (1.0 - θ[:λ]) .* TransitionMat_n

    if kfe_method == :krylov
        # Calculate left-hand unit eigenvector (uses KrylovKit package)
        aux   = real.(eigsolve(TransitionMat', 1)[2][1])
        distr = reshape(vec(aux) ./ sum(aux), n)
    elseif kfe_method == :direct
        # Direct Transition
        # distr_guess .= 1 ./ prod(n) # uniform distribution guess provides most robust convergence rather than using previous distribution
        distr, dist, count = MultipleDirectTransition!(m_a_star, m_n_star, k_a_star, distr_guess, θ[:λ], Π,
                                                       n, m_grid, k_grid, y_grid, ϵ;
                                                       iters = n_direct_transition_iters)
    else
        error("Solution method for Kolmogorov forward equation $(kfe_method) is not recognized. " *
              "Available methods are [:krylov, :direct]")
    end
    end
    #-----------------------------------------------------------------------------
    # Calculate capital stock
    #-----------------------------------------------------------------------------
    K = dot(distr, k_ndgrid) # faster to use dot
    B = dot(distr, m_ndgrid)

    return K, B, TransitionMat, TransitionMat_a, TransitionMat_n, c_a_star, m_a_star, k_a_star, c_n_star, m_n_star, Vm, Vk, distr
end
