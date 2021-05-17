"""
```
find_steadystate(m::BayerBornLuetticke; verbose::Symbol = :none,
                 skip_coarse_grid::Bool = false, parallel::Bool = false)
```
Find the stationary equilibrium capital stock.

### Keyword Arguments
- `verbose`: verbosity of print statements at 3 different levels `[:none, :low, :high]`
- `skip_coarse_grid`: skip using a coarse grid to get close to the steady-state capital stock

### Outputs
- `KSS::Float64`: steady-state capital stock
- `VmSS::Array{Float64,3}`, `VkSS::Array{Float64,3}`: marginal value functions
- `distrSS::Array{Float64,3}`: steady-state distribution of idiosyncratic states, computed by `Ksupply`
"""
function find_steadystate(m::BayerBornLuetticke{T}; verbose::Symbol = :none,
                          skip_coarse_grid::Bool = false, parallel::Bool = false) where {T <: Real}

    # BLAS.set_num_threads(Threads.nthreads()) # this should be set outside of the function

    # -------------------------------------------------------------------------------
    ## STEP 1: Find the stationary equilibrium for coarse grid
    # -------------------------------------------------------------------------------
    #-------------------------------------------------------
    # Income Process and Income Grids
    #-------------------------------------------------------

    if skip_coarse_grid
        KSS = m[:K_star] # use K_star as an initial guess
    else
        # Construct coarse grid based on information from settings
        init_grids!(m; coarse = true)

        if verbose in [:low, :high]
            println("Finding equilibrium capital stock for coarse income grid...")
        end

        # Capital stock guesses
        brent_Kmax = 1.75 * ((m[:δ_0] - 0.0025 + (1.0 - m[:β]) / m[:β]) / m[:α])^(1.0 / (m[:α] - 1.0))
        brent_Kmin = 1.0  * ((m[:δ_0] - 0.0005 + (1.0 - m[:β]) / m[:β]) / m[:α])^(0.5 / (m[:α] - 1.0))

        # a.) Define excess demand function with coarse = true
        init_distr_guess = get_untransformed_values(m[:distr_star])

        θ = parameters2namedtuple(m)
        ϵ = get_setting(m, :coarse_ϵ)
        max_value_function_iters = get_setting(m, :max_value_function_iters)
        n_direct_transition_iters = get_setting(m, :n_direct_transition_iters)
        kfe_method = get_setting(m, :kfe_method)

        ny = get_setting(m, :coarse_ny)
        param_key = [m.parameters[i].key for i in 1:length(m.parameters)]
        param_dict = Dict(param_key .=> m.parameters)

        @inline function d_coarse(  K,
                                    initial::Bool=true,
                                    Vm_guess = zeros(1,1,1),
                                    Vk_guess = zeros(1,1,1),
                                    distr_guess = init_distr_guess
                                    )
            distr = initial ? get_untransformed_values(m[:distr_star])::Array{T,3} : distr_guess
            out = Kdiff(K, m.grids, distr, θ, param_dict,
                        initial, Vm_guess, Vk_guess, distr_guess;
                        verbose = verbose, coarse = true, parallel = parallel,
                        ϵ, max_value_function_iters, n_direct_transition_iters,
                        kfe_method, ny)
            #out = Kdiff(K, m, initial, Vm_guess, Vk_guess, distr_guess;
            #            verbose = verbose, coarse = true, parallel = parallel)
            return out
        end
        # TODO: directly write d_coarse as an inline function e.g. d_coarse(...) = ... instead of @inline function d_coarse(...)
        # This form is meant to be useful for timing how long the Kdiff function takes

        # b.) Find equilibrium capital stock (multigrid on y,m,k)
        KSS = CustomBrent(d_coarse, brent_Kmin, brent_Kmax)[1]

        if verbose in [:low, :high]
            println("Capital stock is $(KSS)")
        end
    end

    # -------------------------------------------------------------------------------
    ## STEP 2: Find the stationary equilibrium for final grid
    # -------------------------------------------------------------------------------
    if verbose in [:low, :high]
        println("Finding equilibrium capital stock for refined income grid...")
    end
    init_grids!(m)

    # Find stationary equilibrium for refined economy
    # a.) Define excess demand function with coarse = false
    ny_fine = get_setting(m, :ny)
    ϵ_fine = get_setting(m, :ϵ)

    θ = parameters2namedtuple(m)
    max_value_function_iters = get_setting(m, :max_value_function_iters)
    n_direct_transition_iters = get_setting(m, :n_direct_transition_iters)
    kfe_method = get_setting(m, :kfe_method)

    param_key = [m.parameters[i].key for i in 1:length(m.parameters)]
    param_dict = Dict(param_key .=> m.parameters)
    @inline function d(  K,
                         initial::Bool=true,
                         Vm_guess = zeros(1,1,1),
                         Vk_guess = zeros(1,1,1),
                         distr_guess = init_distr_guess
                         )
        distr = initial ? get_untransformed_values(m[:distr_star])::Array{T,3} : distr_guess
            out = Kdiff(K, m.grids, distr, θ, param_dict,
                        initial, Vm_guess, Vk_guess, distr_guess;
                        verbose = verbose, coarse = false, parallel = parallel,
                        ϵ = ϵ_fine, max_value_function_iters, n_direct_transition_iters,
                        kfe_method, ny = ny_fine)
        #out = Kdiff(K, m, initial, Vm_guess, Vk_guess, distr_guess;
        #            verbose = verbose, coarse = false, parallel = parallel)
        return out
    end
    # TODO: uncomment code below b/c it directly writes d as an inline function
#=    d(  K,
        initial::Bool=true,
        Vm_guess = zeros(1,1,1),
        Vk_guess = zeros(1,1,1),
        distr_guess = init_distr_guess
        )                              = Kdiff(K, m, initial, Vm_guess, Vk_guess, distr_guess;
                                               verbose = verbose, coarse = false, parallel = parallel)=#

    # b.) Find equilibrium capital stock (multigrid on (m, k, y))
    BrentOut = CustomBrent(d, KSS*.95, KSS*1.05; tol = get_setting(m, :ϵ)) # TODO: if using skip_coarse_grid, add option to permit a wider interval than .95 and 1.05, maybe via a for loop check
    KSS      = BrentOut[1]
    VmSS     = BrentOut[3][2]
    VkSS     = BrentOut[3][3]
    distrSS  = BrentOut[3][4]
    if verbose in [:low, :high]
        println("Capital stock is $(KSS)")
    end

    return KSS, VmSS, VkSS, distrSS
end
