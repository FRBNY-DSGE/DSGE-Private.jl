function lbfgs(fcn::Function,
               x0::Vector,
               args...;
               xtol::Real           = 1e-32, # default from Optim.jl
               ftol::Float64        = 1e-14, # Default from csminwel
               grtol::Real          = 1e-8,  # default from Optim.jl
               iterations::Int      = 1000,
               store_trace::Bool    = true,
               show_trace::Bool     = true,
               extended_trace::Bool = true,
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
        # Optim 2 passes an LBFGSState to the callback; older Optim versions
        # passed trace entries whose iterate lived in `metadata`.
        if hasproperty(current_state, :x) && hasproperty(current_state, :f_x)
            push!(x_trace, copy(current_state.x))
            push!(posterior_ls, current_state.f_x)
        elseif hasproperty(current_state, :metadata) && hasproperty(current_state, :value)
            metadata = current_state.metadata
            if haskey(metadata, "x")
                push!(x_trace, copy(metadata["x"]))
                push!(posterior_ls, current_state.value)
            end
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

    #line search, high iters search
    #ls = LineSearches.BackTracking(order=2, maxstep=Inf, iterations=50)
    #ls2 = LineSearches.HagerZhang(c1=1e-4, c2=0.9, linesearchmax=50)
    ls2 = LineSearches.BackTracking(order=2, maxstep=2.0, iterations=150, c_1 = 1e-4, ρ_lo = 0.4)
    #ls2 = LineSearches.BackTracking(order=2, maxstep=3.0, iterations=150, c_1 = 1e-4, ρ_lo = 0.05, ρ_hi = 0.95)
    #ls2 = LineSearches.BackTracking(order=2, maxstep=3.0, iterations=500, c_1 = 1e-4, ρ_lo = 0.05, ρ_hi = 0.95)

    #=
    ls2 = LineSearches.HagerZhang(;
    c1 = 1e-4,       # sufficient decrease
    c2 = 0.9,        # curvature condition
    αmax = 2.0,      # (optional) max step length
    ρ = 0.5,         # shrink factor for backtracking
    ψ = 0.1          # safeguard parameter
)
=#
    result = if autodiff
        Optim.optimize(fcn_wrapped, x0, Optim.LBFGS(m=20, linesearch = ls2, damping = true),
                       Optim.Options(g_abstol = grtol, f_reltol = ftol, x_abstol = xtol,
                                     iterations = iterations, store_trace = store_trace,
                                     show_trace = show_trace,
                                     extended_trace = extended_trace,
                                     callback = callback,
                                     allow_f_increases = true))
    else
        Optim.optimize(fcn_wrapped, x0, Optim.LBFGS(m=20, linesearch = ls2),
                       Optim.Options(g_abstol = grtol, f_reltol = ftol, x_abstol = xtol,
                                     iterations = iterations, store_trace = store_trace,
                                     show_trace = show_trace,
                                     extended_trace = extended_trace,
                                     callback = callback,
                                     allow_f_increases = true))
    end

    return result, iteration_times, posterior_ls, x_trace
end
