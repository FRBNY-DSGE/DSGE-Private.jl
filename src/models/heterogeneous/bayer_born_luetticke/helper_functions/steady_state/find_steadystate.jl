"""
```
find_steadystate(m::BayerBornLuetticke)
```

Find the stationary equilibrium capital stock.

# Returns
- `KSS`: steady-state capital stock
- `VmSS`, `VkSS`: marginal value functions
- `distrSS::Array{Float64,3}`: steady-state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)
"""
function find_steadystate(m::BayerBornLuetticke{T}; verbose::Symbol = :none,
                          skip_coarse_grid::Bool = false) where {T <: Real}

    # BLAS.set_num_threads(Threads.nthreads()) # this should be set outside of the function

    # TODO: add option to skip the coarse grid b/c already found the steady state and using it as a guess
    # -------------------------------------------------------------------------------
    ## STEP 1: Find the stationary equilibrium for coarse grid
    # -------------------------------------------------------------------------------
    #-------------------------------------------------------
    # Income Process and Income Grids
    #-------------------------------------------------------

    # Construct coarse grid based on information from settings
    DSGE.init_grids!(m; coarse = true)

    if verbose in [:low, :high]
        println("Finding equilibrium capital stock for coarse income grid...")
    end

    # Capital stock guesses
    brent_Kmax = 1.75 * ((m[:δ_0] - 0.0025 + (1.0 - m[:β]) / m[:β]) / m[:α])^(1.0 / (m[:α] - 1.0))
    brent_Kmin = 1.0  * ((m[:δ_0] - 0.0005 + (1.0 - m[:β]) / m[:β]) / m[:α])^(0.5 / (m[:α] - 1.0))

    # a.) Define excess demand function with coarse = true
    init_distr_guess = get_untransformed_values(m[:distr_star])
@inline function d_coarse(  K,
               initial::Bool=true,
               Vm_guess = zeros(1,1,1),
               Vk_guess = zeros(1,1,1),
               distr_guess = init_distr_guess
               )
        out = Kdiff(K, m, initial, Vm_guess, Vk_guess, distr_guess;
                    verbose = verbose, coarse = true)
    return out
end

    # b.) Find equilibrium capital stock (multigrid on y,m,k)
    KSS = CustomBrent(d_coarse, brent_Kmin, brent_Kmax)[1]

    if verbose in [:low, :high]
        println("Capital stock is $(KSS)")
    end
    # -------------------------------------------------------------------------------
    ## STEP 2: Find the stationary equilibrium for final grid
    # -------------------------------------------------------------------------------
    if verbose in [:low, :high]
        println("Finding equilibrium capital stock for refined income grid...")
    end
    DSGE.init_grids!(m)

    # Find stationary equilibrium for refined economy
    # a.) Define excess demand function with coarse = false
@inline function d(  K,
               initial::Bool=true,
               Vm_guess = zeros(1,1,1),
               Vk_guess = zeros(1,1,1),
               distr_guess = init_distr_guess
               )
        out = Kdiff(K, m, initial, Vm_guess, Vk_guess, distr_guess;
                    verbose = verbose, coarse = false)
    return out
end
#=    d(  K,
        initial::Bool=true,
        Vm_guess = zeros(1,1,1),
        Vk_guess = zeros(1,1,1),
        distr_guess = init_distr_guess
        )                              = Kdiff(K, m, initial, Vm_guess, Vk_guess, distr_guess;
                                               verbose = verbose, coarse = false)=#

    # b.) Find equilibrium capital stock (multigrid on y,m,k) # TODO: isn't grid on (m, k, y)?
    BrentOut = CustomBrent(d, KSS*.95, KSS*1.05; tol = get_setting(m, :ϵ))
    KSS      = BrentOut[1]
    VmSS     = BrentOut[3][2]
    VkSS     = BrentOut[3][3]
    distrSS  = BrentOut[3][4]
    if verbose in [:low, :high]
        println("Capital stock is $(KSS)")
    end

    return KSS, VmSS, VkSS, distrSS
end
