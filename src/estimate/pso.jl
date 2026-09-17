"""
```
pso(fcn, x0, args...; kwargs...)
```

Particle Swarm Optimization using BlackBoxOptim.jl

Minimizes `fcn` starting from initial parameter vector `x0` using particle swarm optimization.
This is a derivative-free global optimization method that is particularly useful for
non-convex objective functions.

### Arguments
- `fcn::Function`: Objective function to minimize
- `x0::Vector`: Initial parameter guess

### Keyword Arguments
- `xtol::Real`: Tolerance for parameter convergence (default: 1e-32)
- `ftol::Float64`: Tolerance for function value convergence (default: 1e-14)
- `grtol::Real`: Gradient tolerance (not used for PSO, included for consistency)
- `iterations::Int`: Maximum number of iterations (default: 1000)
- `store_trace::Bool`: Whether to store optimization trace (default: false)
- `show_trace::Bool`: Whether to display optimization progress (default: false)
- `extended_trace::Bool`: Whether to store extended trace information (default: false)
- `verbose::Symbol`: Verbosity level (:none, :low, :high) (default: :none)
- `rng::AbstractRNG`: Random number generator (default: MersenneTwister())
- `autodiff::Bool`: Not used for PSO (included for consistency with other optimizers)
- `n_particles::Int`: Number of particles in the swarm (default: 50)
- `kwargs...`: Additional keyword arguments

### Returns
Returns an optimization result object compatible with Optim.jl results, containing:
- `minimizer`: The optimal parameter vector found
- `minimum`: The objective function value at the minimizer
- `iterations`: Number of iterations performed
- `f_converged`: Whether convergence in function value was achieved
- `x_converged`: Whether convergence in parameters was achieved
- `g_converged`: Set to false (PSO is gradient-free)

### Notes
- PSO is a derivative-free method, so `autodiff` and `grtol` are not used
- The algorithm maintains a swarm of particles that explore the parameter space
- Good for global optimization and non-convex problems
- May require more function evaluations than gradient-based methods
"""
function pso(fcn::Function,
             x0,
             m;
             xtol::Real           = 1e-32,
             ftol::Float64        = 1e-14,
             grtol::Real          = 1e-8,
             iterations::Int      = 1000,
             store_trace::Bool    = false,
             show_trace::Bool     = false,
             extended_trace::Bool = false,
             verbose::Symbol      = :none,
             rng::AbstractRNG     = MersenneTwister(),
             autodiff::Bool       = false,
             n_particles::Int     = 10000,
             use_parallel::Bool   = false,
             n_workers::Int       = nworkers(),
             kwargs...)





    #default search space: TODO make search space the param bounds
    #=
    search_factor = 0.5
    search_range = [(x0[i] - abs(x0[i]) * search_factor - 0.1,
                     x0[i] + abs(x0[i]) * search_factor + 0.1)
                    for i in 1:length(x0)]
    =#

    #search range from model parameters
    free_params = [p for p in m.parameters if !p.fixed]
    n_free_params = length(free_params)
    x_opt = [p.value for p in free_params]

    #search range that concentrates around starting values but respects bounds
    concentration_factor = 0.5
    search_range = Vector{Tuple{Float64, Float64}}(undef, n_free_params)

    for i = 1:n_free_params
        mparam = free_params[i]        
        starting_val = mparam.value

        
        lower_bound = mparam.valuebounds[1]
        upper_bound = mparam.valuebounds[2]
        #=
        lower_search = max(lower_bound,
                          starting_val - concentration_factor * (starting_val - lower_bound))
        upper_search = min(upper_bound,
                          starting_val + concentration_factor * (upper_bound - starting_val))

        if lower_search >= upper_search
            lower_search = lower_bound
            upper_search = upper_bound
        end
        =#
        search_range[i] = (lower_bound, upper_bound)
    end

#=
result = if autodiff 
         Optim.optimize(fcn, x0, LBFGS(m=10, linesearch = ls), autodiff=:forward, 
                        Optim.Options(g_tol = grtol, f_tol = ftol, x_tol = xtol, 
                                      iterations = iterations, store_trace = store_trace, 
                                      show_trace = show_trace, 
                                      extended_trace = extended_trace, 
                                      callback = callback, 
                                      allow_f_increases = true)) 
     else 
         Optim.optimize(fcn, x0, LBFGS(m=10, linesearch = ls), 
                         Optim.Options(g_tol = grtol, f_tol = ftol, x_tol = xtol, 
                                      iterations = iterations, store_trace = store_trace, 
                                      show_trace = show_trace, 
                                      extended_trace = extended_trace, 
                                      callback = callback, 
                                      allow_f_increases = true)) 
     end 
  
     return result, iteration_times
=#
 

    
    #set starting population, ad some jitter to x0
    #n_pop = 100
    #random_scaling = 0.98 .+ (rand(n_pop, n_free_params) .* 0.02)
    #x0_free = x0[1:n_free_params]
    
    #initial_population = [x0_free]
    #initial_population = [x0_free .* random_scaling[i,:] for i in 1:n_pop]
    #@show initial_population
    max_fevals = n_particles * iterations
    trace_mode = show_trace ? :verbose : :silent
    method = :adaptive_de_rand_1_bin_radiuslimited


    #callback - need implement
    frequency = 2
    x_current_minimizer = Vector{Vector{Float64}}()
    time_seconds = Float64[]

    
    start_time = time()

    callback = function(state)
        if state.iteration % frequency == 0

           elapsed = time()- start_time
           push!(x_current_minimizer, copy(best_candidate(state)))
           push!(time_seconds, elapsed)
        end
        false
    end

#=
   # parallelization if requested
    if use_parallel && nworkers() > 1
        result = BlackBoxOptim.bboptimize(fcn;
                                          SearchRange = search_range,
                                          Method = method,
                                          MaxFuncEvals = max_fevals,
                                          TraceMode = trace_mode,
                                          PopulationSize = n_particles,
                                          FitnessToleranceSince = iterations ÷ 10,
                                          FitnessTolerance = ftol,
                                          #RandomizeRNGseed = true,
                                          #RngSeed = rand(rng, 1:10000),
                                          initial_population = initial_population,
                                          Workers = workers()[1:min(n_workers, nworkers())])
    else
        result = BlackBoxOptim.bboptimize(fcn;
                                          SearchRange = search_range,
                                          Method = method,
                                          MaxFuncEvals = max_fevals,
                                          TraceMode = trace_mode,
                                          #PopulationSize = n_particles,
                                          FitnessToleranceSince = iterations ÷ 10,
                                          FitnessTolerance = ftol,
                                          #RandomizeRNGseed = true,
                                          #RngSeed = rand(rng, 1:10000)
                                          initial_population = initial_population)
    end
    @show initial_population
=#



   # parallelization if requested
    if use_parallel && nworkers() > 1
        result = BlackBoxOptim.bboptimize(fcn;
                                          SearchRange = search_range,
                                          Method = method,
                                          MaxFuncEvals = max_fevals,
                                          TraceMode = trace_mode,
                                          PopulationSize = n_particles,
                                          FitnessToleranceSince = iterations ÷ 10,
                                          FitnessTolerance = ftol,
                                          #RandomizeRNGseed = true,
                                          #RngSeed = rand(rng, 1:10000),
                                          initial_population = initial_population,
                                          Workers = workers()[1:min(n_workers, nworkers())])
    else
        result = BlackBoxOptim.bboptimize(fcn, x_opt;
                                          SearchRange = search_range,
                                          #Method = method,
                                          #MaxFuncEvals = max_fevals,
                                          #TraceMode = trace_mode,
                                          NumDimensions = n_free_params,
                                          PopulationSize = n_particles,
                                          Method = method)
                                          #FitnessToleranceSince = iterations ÷ 10,
                                          #FitnessTolerance = ftol,
                                          #RandomizeRNGseed = true,
                                          #RngSeed = rand(rng, 1:10000)
                                          #initial_population = initial_population
                                          
    end
 





    x_min = BlackBoxOptim.best_candidate(result)
    f_min = BlackBoxOptim.best_fitness(result)

    f_initial = fcn(x0)
    f_improvement = abs(f_initial - f_min)
    f_converged = f_improvement < ftol || isapprox(f_min, f_initial, atol=ftol)

    #check parameter convergence
    #x_change = maximum(abs.(x_min .- x0[1:13]))
    #x_converged = x_change < xtol
    x_converged = false
    n_iterations = BlackBoxOptim.f_calls(result) ÷ n_particles
    
    opt_result = (
        minimizer = x_min,
        minimum = f_min,
        iterations = n_iterations,
        f_converged = f_converged,
        x_converged = x_converged,
        g_converged = false,  # PSO is derivative-free
        iteration_converged = f_converged || x_converged,
        trace = store_trace ? BlackBoxOptim.get_trace(result) : nothing
    )

    return opt_result



end
