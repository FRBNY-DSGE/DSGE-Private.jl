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
    
    #callback time
    iteration_times = Float64[]
    start_time = time()

    callback = function(state)
        push!(iteration_times, time() - start_time)
        false
    end


    #random_scaling = 0.97
    #x0 = x0 .* random_scaling

    #wrapper objective fcn replace Inf with large finite value before gradients are computed
    INF_REPLACEMENT = 1e15
    fcn_wrapped = function(x)
        val = fcn(x)
        if isinf(val) || isnan(val) || val > INF_REPLACEMENT
            return INF_REPLACEMENT
        else
            return val
        end
    end

    #line search, high iters search
    ls = LineSearches.BackTracking(order=2, maxstep=Inf, iterations=50)
    #ls_hager = LineSearches.HagerZhang()
    ls2 = LineSearches.BackTracking(order=2, maxstep=5.0, iterations=100, c_1 = 1e-4, ρ_lo = 0.4)

    result = if autodiff
        Optim.optimize(fcn_wrapped, x0, LBFGS(m=20, linesearch = ls2),
                       Optim.Options(g_tol = grtol, f_tol = ftol, x_tol = xtol,
                                     iterations = iterations, store_trace = store_trace,
                                     show_trace = show_trace,
                                     extended_trace = extended_trace,
                                     callback = callback,
                                     allow_f_increases = true))
    else
        Optim.optimize(fcn_wrapped, x0, LBFGS(m=20, linesearch = ls2),
                       Optim.Options(g_tol = grtol, f_tol = ftol, x_tol = xtol,
                                     iterations = iterations, store_trace = store_trace,
                                     show_trace = show_trace,
                                     extended_trace = extended_trace,
                                     callback = callback,
                                     allow_f_increases = true))
    end

    return result, iteration_times
end
