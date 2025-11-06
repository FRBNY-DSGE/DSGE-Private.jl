function cmaes(fcn::Function,
               x0::Vector,
               args...;
               #xtol::Real           = 1e-32, # default from Optim.jl
               ftol::Float64        = 1e-14, # Default from csminwel
               grtol::Real          = 1e-8,  # default from Optim.jl
               iterations::Int      = 1000,
               store_trace::Bool    = true,
               show_trace::Bool     = true,
               extended_trace::Bool = true,
               verbose::Symbol      = :none,
               rng::AbstractRNG     = MersenneTwister(),
               autodiff::Bool       = false,
               σ::Float64           = 1.0,
               μ::Union{Int, Nothing} = 10,
               λ::Union{Int, Nothing} = 2*μ,
               nworkers::Int          =10,
               kwargs...)
    
    #callback time and parameter trace
    iteration_times = Float64[]
    posterior_ls = Float64[]
    x_trace = typeof(x0)[]
    start_time = time()

    #=
    callback = function(trace_or_state)
        push!(iteration_times, time() - start_time)
        #when store_trace=true, callback receives the full trace (a vector)
        #get the last state from the trace
        
        #extended callback, but currently saving untransformed        
        if trace_or_state isa AbstractVector
            current_state = trace_or_state[end]
        else
            current_state = trace_or_state
        end
        if haskey(current_state.metadata, "x")
            push!(x_trace, copy(current_state.metadata["x"]))
            push!(posterior_ls, current_state.value)
        end
        false
    end =#

    #wrapper objective fcn replace Inf with large finite value before gradients are computed
    INF_REPLACEMENT = 1e15
    fcn_wrapped = function(x)
        val = fcn(x)
        if isinf(val) || isnan(val) || val > INF_REPLACEMENT
            #push!(posterior_ls)
            return INF_REPLACEMENT
        else
            #push!(posterior_ls)
            return val
        end
    end
    
    n = length(x0)
    if isnothing(λ)
        λ = 4 + floor(Int, 3 * log(n))
    end
    if isnothing(μ)
        μ = floor(Int, λ / 2)
    end
    addprocs(nworkers)
    #line search, high iters search
    #ls = LineSearches.BackTracking(order=2, maxstep=Inf, iterations=50)
    #ls_hager = LineSearches.HagerZhang()
    #ls2 = LineSearches.BackTracking(order=2, maxstep=5.0, iterations=100, c_1 = 1e-4, ρ_lo = 0.4)
#=
    result = if autodiff
        Optim.optimize(fcn_wrapped, x0, LBFGS(m=20, linesearch = ls2),
                       Optim.Options(g_tol = grtol, f_tol = ftol, x_tol = xtol,
                                     iterations = iterations, store_trace = store_trace,
                                     show_trace = show_trace,
                                     extended_trace = extended_trace,
                                     callback = callback,
                                     allow_f_increases = true))
    else
        Optim.optimize(fcn_wrapped, x0, ConjugateGradient(linesearch = ls2),
                       Optim.Options(g_tol = grtol, f_tol = ftol, x_tol = xtol,
                                     iterations = iterations, store_trace = store_trace,
                                     show_trace = show_trace,
                                     extended_trace = extended_trace,
                                     callback = callback,
                                     allow_f_increases = true))
    end=#


    function fcn_batch(X::Matrix{Float64})
        population = [X[:, i] for i ∈ 1:size(X, 2)]

        results = pmap(fcn_wrapped, population)
        return results
    end

    function fcn_batch(x::AbstractVector{<:Real})
        return fcn_wrapped(x)
    end

    Random.seed!(rng)

    if nworkers > 1
        if verbose!= :none
            println("Running CMAES with $(nworkers) workers")
        end
       
        cma_result = CMAEvolutionStrategy.minimize(fcn_batch, x0, σ;
                             maxfevals = iterations * λ,
                             popsize = λ,
                             #μ = μ,
                             #verb_disp = show_trace ? 10 : 0,
                             #parallel_objective = true,  # Tell CMA-ES we're providing batch evaluation
                             seed = rand(rng, UInt32))
        #Main.xx = cma_result
    else
        if verbose != :none
            println("Running CMA-ES in serial mode (no workers detected)")
            println("Add workers with: addprocs(n) before calling this function")
        end
        
        # Fallback to serial evaluation
        cma_result = CMAEvolutionStrategy.minimize(fcn_wrapped, x0, σ;
                             maxfevals = iterations * λ,
                             popsize = λ,
                             #μ = μ,
                             #verb_disp = show_trace ? 10 : 0,
                             seed = rand(rng, UInt32))
    end

    x_best = cma_result.logger.xbest
    f_best = minimum(cma_result.logger.fbest)
    n_evals = cma_result.p.λ * cma_result.stop.it
    stop_flag = cma_result.stop.reason


    println("\n===== CMAEvolutionStrategy Summary =====")

# Best solution and objective value
println("Best x (xbest):")
println(cma_result.logger.xbest)
println("\nBest f (fbest / posterior):")
println(cma_result.logger.fbest)

# Current population mean
println("\nPopulation mean (search distribution center):")
println(cma_result.p.mean)

# Optional: if logger tracks fmedian or frange
if hasproperty(cma_result.logger, :fmedian)
    println("\nMedian objective value:")
    println(cma_result.logger.fmedian)
end

if hasproperty(cma_result.logger, :frange)
    println("\nRange of fitness values (min, max):")
    println(cma_result.logger.frange)
end

# Optional: iteration times if recorded
if hasproperty(cma_result.logger, :times)
    println("\nIteration times (seconds):")
    println(join(round.(cma_result.logger.times; digits=4), ", "))
end

# Stop reason
println("\nStop reason:")
println(cma_result.stop.reason)

println("\n========================================\n")


    
    # Store trace (at least the final result)
    if store_trace
        push!(iteration_times, time() - start_time)
        push!(x_trace, x_best)
        push!(posterior_ls, f_best)
    end
    
    # Determine convergence based on stop flag
    converged = (stop_flag != :maxfevals)
    
    # Create result object similar to Optim.jl output structure
    result = (
        minimizer = x_best,
        minimum = f_best,
        iterations = floor(Int, n_evals / λ),
        iteration_converged = !converged,
        #x_converged = stop_flag == :tolx,
        f_converged = stop_flag == :tolfun,
        g_converged = false,
        converged = converged,
        x_trace = store_trace ? x_trace : nothing,
        f_trace = store_trace ? posterior_ls : nothing,
        time_run = time() - start_time
    )


    return result, iteration_times, posterior_ls, x_trace
end
