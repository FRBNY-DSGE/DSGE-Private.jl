function trust_region_newton(fcn::Function,
               x0::Vector,
               args...;
               xtol::Real           = 1e-32, # default from Optim.jl
               ftol::Float64        = 1e-14, # Default from csminwel
               grtol::Real          = 1e-8,  # default from Optim.jl
               iterations::Int      = 1000,
               store_trace::Bool    = false,
               show_trace::Bool     = false,
               extended_trace::Bool = false,
               verbose::Symbol      = :none,
               rng::AbstractRNG     = MersenneTwister(),
               autodiff::Bool       = false,
               kwargs...)

    #callback time and parameter trace
    iteration_times = Float64[]
    posterior_ls = Float64[]
    x_trace = typeof(x0)[]
    start_time = time()
 
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
     end


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


   
    result = if autodiff
        Optim.optimize(fcn_wrapped, x0, NewtonTrustRegion(),
                       Optim.Options(g_tol = grtol, f_tol = ftol, x_tol = xtol,
                                     iterations = iterations, store_trace = store_trace,
                                     show_trace = show_trace,
                                     extended_trace = extended_trace, callback=callback))
    else
        Optim.optimize(fcn_wrapped, x0, NewtonTrustRegion(),
                       Optim.Options(g_tol = grtol, f_tol = ftol, x_tol = xtol,
                                     iterations = iterations, store_trace = store_trace,
                                     show_trace = show_trace,
                                     extended_trace = extended_trace, callback=callback))
    end

    return result, iteration_times, posterior_ls, x_trace
end
