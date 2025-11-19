#= function cmaes(fcn::Function,
               x0::Vector,
               s0::Float64,
               args...;
               lower::Array{Float64}       = nothing, #default from CMAEvolutionStrategy.jl
               upper::Array{Float64}       = nothing, 
               popsize::Int     =4 + floor(Int, 3*log(length(x0))), #default from CMAEvolutionStrategy.jl
               callback:Union{Int, Nothing} = nothing,
               parallel_evaluation::Bool = false,
               maxiter::Union{Int, Nothing}         = nothing,
               maxfevals::Union{Int, Nothing}            = nothing,
               store_trace::Bool = false,
               show_trace::Bool =false,
               extended_trace::Bool = false,
               verbose::Symbol = nothing,
               rng::AbstractRNG = Random.default_rng(),
               kwargs...) =#
   

function cmaes(fcn::Function,
               x0::Vector,
               s0::Float64,
               args...;
               lower = nothing,
               upper = nothing,
               popsize = 3*length(x0),  #4 + floor(Int, 3*log(length(x0))),
               callback = nothing,
               parallel_evaluation = false,
               maxiter = 10000,
               maxfevals = 100000,
               store_trace = false,
               show_trace = false,
               extended_trace = false,
               verbose = :none,
               rng = Random.default_rng(),
               kwargs...)




    #callback time and parameter trace
    iteration_times = Float64[]
    posterior_ls = Float64[]
    x_trace = typeof(x0)[]
    start_time = time()

#=
    callback = (o, y, fvals, perm) -> begin
        it     = o.stop.it
        fbest  = minimum(fvals)
        σ      = o.p.sigma.σ
        elapsed = time() - start_time
        push!(iteration_times, elapsed)

        # store trace if requested
        if store_trace
            push!(x_trace, copy(o.logger.xbest[end]))
            push!(posterior_ls, fbest)
        end

        if show_trace
            @printf("Iter %4d | fbest = %10.4e | σ = %.5f | elapsed %.2fs\n",
                    it, fbest, σ, elapsed)
            flush(stdout)
        end

        if σ > 1.0
            pritnlin("\n WARNING: σ has grwon to $σ - stopping optimization")
            return true
        end
        return false
        

    end
=# ##^CALLBACK FN FOR LIMITING THE STEP SIZE GROWTH

callback = (o, y, fvals, perm) -> begin
    it     = o.stop.it
    fbest  = minimum(fvals)
    σ      = o.p.sigma.σ
    elapsed = time() - start_time
    push!(iteration_times, elapsed)

    # store trace if requested
    if store_trace
        push!(x_trace, copy(o.logger.xbest[end]))
        push!(posterior_ls, fbest)
    end

    if show_trace
        ("Iter %4d | fbest = %10.4e | σ = %.5f | elapsed %.2fs\n",
                it, fbest, σ, elapsed)
        flush(stdout)
    end
    nothing
end


    function f_batch(X::Matrix)
        pop = [X[:, i] for i ∈ 1:size(X, 2)]
        return pmap(fcn, pop)
    end

    f_batch(x::AbstractVector) = fcn(x)

    cma_result = CMAEvolutionStrategy.minimize(
        f_batch, x0, s0;
        lower = lower,
        upper = upper,
        popsize = popsize,
        callback = callback,
        parallel_evaluation = parallel_evaluation,
        seed = 123,
        maxiter = maxiter,
        maxfevals = maxfevals               
    )

    Main.xx[] = cma_result

    x_best = xbest(cma_result)
    f_best = fbest(cma_result)
    n_evals = sum(cma_result.logger.times)
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


    
   # Create result object similar to Optim.jl output structure
    result = (minimizer  = xbest(cma_result),
        minimum    = fbest(cma_result),
        iterations = cma_result.stop.it,
        converged  = cma_result.stop.reason != :maxiter,
        x_trace    = store_trace ? x_trace : nothing,
        f_trace    = store_trace ? posterior_ls : nothing,
        time_run   = time() - start_time
   )


    return result, iteration_times, posterior_ls, x_trace
end
