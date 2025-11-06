function xnes(fcn::Function,
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
               μ::Union{Int, Nothing} = nothing,
               λ::Union{Int, Nothing} = nothing,
               nworkers::Int        = 1,
               kwargs...)

    # Callback time and parameter trace
    iteration_times = Float64[]
    posterior_ls = Float64[]
    x_trace = typeof(x0)[]
    start_time = time()

    # Wrapper objective fcn - replace Inf with large finite value
    INF_REPLACEMENT = 1e15
    fcn_wrapped = function(x)
        val = fcn(x)
        if isinf(val) || isnan(val) || val > INF_REPLACEMENT
            return INF_REPLACEMENT
        else
            return val
        end
    end

    n = length(x0)

    if isnothing(λ)
        λ = 4 + floor(Int, 3 * log(n))
    end

    Random.seed!(rng)

    if verbose != :none
        println("Running XNES with population size $(λ)")
    end

    # Use BlackBoxOptim's bboptimize with xnes method
    # BlackBoxOptim expects a minimization problem
    xnes_result = BlackBoxOptim.bboptimize(fcn_wrapped;
                            Method = :xnes,
                            SearchRange = [(-Inf, Inf) for _ in 1:n],  # Unbounded search
                            NumDimensions = n,
                            PopulationSize = λ,
                            MaxFuncEvals = iterations * λ,
                            TraceMode = show_trace ? :verbose : :silent,
                            RandomizeRngSeed = false,
                            RngSeed = rand(rng, UInt),
                            x0 = x0)

    x_best = BlackBoxOptim.best_candidate(xnes_result)
    f_best = BlackBoxOptim.best_fitness(xnes_result)
    n_evals = BlackBoxOptim.num_func_evals(xnes_result)
    stop_reason = BlackBoxOptim.stop_reason(xnes_result)

    # Store trace (at least the final result)
    if store_trace
        push!(iteration_times, time() - start_time)
        push!(x_trace, x_best)
        push!(posterior_ls, f_best)
    end

    # Determine convergence based on stop reason
    # BlackBoxOptim stops for various reasons, consider it converged if not max_evals
    converged = !occursin("Max number", stop_reason)

    # Create result object similar to Optim.jl output structure
    result = (
        minimizer = x_best,
        minimum = f_best,
        iterations = floor(Int, n_evals / λ),
        iteration_converged = !converged,
        x_converged = occursin("fitness tolerance", lowercase(stop_reason)),
        f_converged = occursin("fitness tolerance", lowercase(stop_reason)),
        g_converged = false,
        converged = converged,
        x_trace = store_trace ? x_trace : nothing,
        f_trace = store_trace ? posterior_ls : nothing,
        time_run = time() - start_time
    )

    return result, iteration_times, posterior_ls, x_trace
end
