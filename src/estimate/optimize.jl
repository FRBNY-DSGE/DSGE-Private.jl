"""
```
mutable struct optimization_result{T}
    minimizer::Vector{T}
    minimum::T
    converged::Bool
    iterations::Int

```
Container type for various optimization outputs
"""
mutable struct optimization_result{T}
    minimizer::Vector{T}
    minimum::T
    converged::Bool
    iterations::Int
end

global f_opt_global

"""
```
optimize!(m::Union{AbstractDSGEModel,AbstractVARModel}, data::Matrix;
          method::Symbol       = :csminwel,
          xtol::Real           = 1e-32,  # default from Optim.jl
          ftol::Float64        = 1e-14,  # Default from csminwel
          grtol::Real          = 1e-8,   # default from Optim.jl
          iterations::Int      = 1000,
          store_trace::Bool    = false,
          show_trace::Bool     = false,
          extended_trace::Bool = false,
          mle::Bool            = false, # default from estimate.jl
          step_size::Float64   = .01,
          toggle::Bool         = true,  # default from estimate.jl
          verbose::Symbol      = :none)
```
Wrapper function to send a model to csminwel (or another optimization routine).
"""
function optimize!(m::Union{AbstractDSGEModel,AbstractVARModel, AbstractDSGEVECMModel},
                   data::AbstractArray;
                   method::Symbol       = :csminwel,
                   xtol::Real           = 1e-32,  # default from Optim.jl
                   ftol::Float64        = 1e-14,  # Default from csminwel
                   grtol::Real          = 1e-8,   # default from Optim.jl
                   iterations::Int      = 1000,
                   store_trace::Bool    = true,
                   show_trace::Bool     = true,
                   extended_trace::Bool = true,
                   mle::Bool            = false, # default from estimate.jl
                   step_size::Float64   = .01,
                   toggle::Bool         = true,  # default from estimate.jl
                   verbose::Symbol      = :none,
                   autodiff::Bool       = false)

    ########################################################################################
    ### Step 1: Setup
    ########################################################################################


    #global f_opt_global
    
    # For now, only csminwel should be used
    optimizer = if method == :csminwel
        csminwel
    elseif method == :simulated_annealing
        simulated_annealing
    elseif method == :nelder_mead
        nelder_mead
    elseif method == :combined_optimizer
        combined_optimizer
    elseif method == :lbfgs
        lbfgs
    elseif method == :trust_region_newton
        trust_region_newton
    elseif method == :pso
        pso
    elseif method == :conjugate_gradient
        conjugate_gradient
    elseif method == :cmaes
        cmaes
    elseif method == :xnes
        xnes
    else
        error("Method ", method, " is not supported.")
    end

    regime_switching = haskey(get_settings(m), :regime_switching) && get_setting(m, :regime_switching)

    # Inputs to optimization
    para_free_inds = ModelConstructors.get_free_para_inds(get_parameters(m);
                                                          regime_switching = regime_switching, toggle = toggle)
    H0             = 1e-4 * eye(length(para_free_inds))
    
    if method == :pso
        x_model = [p for p in m.parameters]
        n_params = length(x_model)
        x_opt = [p.value for p in x_model]
    else

        x_model        = transform_to_real_line(get_parameters(m); regime_switching = regime_switching)
        x_opt          = x_model[para_free_inds]

    end

    
    ########################################################################################
    ### Step 2: Initialize f_opt
    ########################################################################################

    function f_opt(x_opt)
        try
            x_model[para_free_inds] = x_opt

            if typeof(m) <: AbstractDSGEVARModel
                transform_to_model_space!(m, x_model)
            else
                transform_to_model_space!(m, x_model; regime_switching = regime_switching)
            end

        catch
            return Inf
        end

        if mle
            out = -likelihood(m, data; catch_errors = true)
        else
            out = -posterior(m, data; catch_errors = true)
            @show out
        end

        out = !isnan(out) ? out : Inf
        return out
    end



    function f_opt2(x_opt)
        try
           x_model[para_free_inds] = x
           DSGE.update!(m, para_vals)
        catch
            return Inf
        end

        if mle
            out = -likelihood(m, data; catch_errors = true)
        else
            out = -posterior(m, data; catch_errors = true)
        end

        out = !isnan(out) ? out : Inf
        return out
    end



    ########################################################################################
    ### Step 3: Optimizer-specific setup, call optimizer
    ########################################################################################

    # variables used across several optimizers
    rng = get_rng(m)
    temperature = get_setting(m, :simulated_annealing_temperature)
    max_cycles  = get_setting(m, :combined_optimizer_max_cycles)
    block_frac  = get_setting(m, :simulated_annealing_block_proportion)
    H_ = nothing
    callback_data = nothing

    neighbor! = if isa(m, AbstractDSGEModel)
        function _neighbor_dsge!(x, x_proposal)
            # This function computes a proposal "next step" during simulated annealing.
            # Inputs:
            # - `x`: current position (of non-fixed states)
            # - `x_proposal`: proposed next position (of non-fixed states).
            #                 (passed in for pre-allocation purposes)
            # Outputs:
            # - `x_proposal`

            T = eltype(x)
            npara = length(x)
            subset_inds = []
            while length(subset_inds) == 0
                subset_inds = randsubseq(para_free_inds,block_frac)
            end

            # Convert x_proposal to model space and expand to full parameter vector
            x_all = T[ModelConstructors.get_values(get_parameters(m); regime_switching = regime_switching)]  # to get fixed values
            x_all[para_free_inds] = x                 # this is from real line

            x_all_model = transform_to_model_space(get_parameters(m), x_all; regime_switching = regime_switching)
            x_proposal_all = copy(x_all_model)

            success = false
            while !success
                # take a step in model space
                for i in subset_inds
                    # TODO: generalize to regime-switching. Key problem is that we need to construct a dictionary which maps
                    # the indices of `x_all` or at least `para_free_inds` to the location in get_parameters(m)
                    if regime_switching
                        p_i, reg_i = x_ind_2_pvec_ind[i] # returns index within get_parameters(m) and the regime
                    else
                        p_i = i
                    end
                    prior_var = moments(get_parameters(m)[i])[2] # moments(get(get_parameters(m)[i].prior))[2]
                    proposal_in_bounds = false
                    proposal = x_all_model[i]
                    if haskey(get_parameters(m)[i].regimes, :valuebounds)
                        lower = regime_valuebounds(get_parameters(m)[i], 1)[1]
                        upper = regime_valuebounds(get_parameters(m)[i], 1)[2]
                    else
                        lower = get_parameters(m)[i].valuebounds[1]
                        upper = get_parameters(m)[i].valuebounds[2]
                    end

                    # draw a new parameter value, and redraw if out of bounds
                    while !proposal_in_bounds # TODO: generalize to regime-switching
                        r = rand([-1 1]) * rand()
                        proposal = x_all_model[i] + (r * step_size * prior_var)
                        if lower < proposal < upper
                            proposal_in_bounds = true
                        end
                    end
                    @inbounds x_proposal_all[i] = proposal
                end

                # check that model can be solved
                try
                    DSGE.update!(m, x_proposal_all)
                    compute_system(m; tvis = haskey(get_settings(m), :tvis_information_set))
                    x_proposal_all = transform_to_real_line(get_parameters(m), x_proposal_all;
                                                            regime_switching = regime_switching)
                    success = true
                catch ex
                    if !(typeof(ex) in [DomainError, ParamBoundsError, GensysError])
                        rethrow(ex)
                    end
                end
            end

            x_proposal[1:end] = x_proposal_all[para_free_inds]

            return
        end
    elseif isa(m, AbstractDSGEVARModel) || isa(m, AbstractDSGEVECMModel)
        function _neighbor_dsgevar!(x, x_proposal)
            T = eltype(x)
            npara = length(x)
            subset_inds = []
            while length(subset_inds) == 0
                subset_inds = randsubseq(para_free_inds,block_frac)
            end

            # Convert x_proposal to model space and expand to full parameter vector
            x_all = T[ModelConstructors.get_values(get_parameters(m); regime_switching = regime_switching)]  # to get fixed values
            x_all[para_free_inds] = x                 # this is from real line

            x_all_m = transform_to_model_space(get_parameters(m), x_all;
                                               regime_switching = regime_switching)
            x_proposal_all = copy(x_all_m)

            success = false
            while !success
                # take a step in model space
                for i in subset_inds
                    prior_var = moments(get_parameters(m)[i])[2]#moments(get(m.parameters[i].prior))[2] # TODO: generalize to regime-switching
                    proposal_in_bounds = false
                    proposal = x_all_m[i]
                    lower = get_parameters(m)[i].valuebounds[1] # TODO: generalize to regime-switching
                    upper = get_parameters(m)[i].valuebounds[2]
                    # draw a new parameter value, and redraw if out of bounds
                    while !proposal_in_bounds # TODO: generalize to regime-switching
                        r = rand([-1 1]) * rand()
                        proposal = x_all_m[i] + (r * step_size * prior_var)
                        if lower < proposal < upper
                            proposal_in_bounds = true
                        end
                    end
                    @inbounds x_proposal_all[i] = proposal
                end

                # check that model can be solved
                try
                    DSGE.update!(m, x_proposal_all)
                    compute_system(m; tvis = haskey(get_settings(m), :tvis_information_set))
                    x_proposal_all = transform_to_real_line(get_parameters(m), x_proposal_all;
                                                            regime_switching = regime_switching)
                    success = true
                catch ex
                    if !(typeof(ex) in [DomainError, ParamBoundsError, GensysError])
                        rethrow(ex)
                    end
                end
            end

            x_proposal[1:end] = x_proposal_all[para_free_inds]

            return
        end
    else
        error("The simulated annealing function for a model of type $(typeof(m)) has not been implemented yet")
    end

    temperature = get_setting(m, :simulated_annealing_temperature)
    rng = get_rng(m)
    if method == :simulated_annealing
        if regime_switching
            error("Simulated annealing with regime switching currently does not work.")
        end
        opt_result = optimizer(f_opt, x_opt;
                               iterations = iterations, step_size = step_size,
                               store_trace = store_trace, show_trace = show_trace,
                               extended_trace = extended_trace,
                               neighbor! = neighbor!, verbose = verbose, rng = rng,
                               temperature = temperature)
        converged = opt_result.iteration_converged
        out = optimization_result(opt_result.minimizer, opt_result.minimum, converged,
                                  opt_result.iterations)

    elseif method == :nelder_mead
        opt_result = optimizer(f_opt, x_opt;
                               iterations = iterations,
                               store_trace = store_trace, show_trace = show_trace,
                               extended_trace = extended_trace, verbose = verbose, rng = rng)
        converged = opt_result.iteration_converged
        out = optimization_result(opt_result.minimizer, opt_result.minimum, converged,
                                  opt_result.iterations)

    elseif method == :csminwel
        opt_result, H_, iteration_times, posterior_ls = optimizer(f_opt, x_opt, H0;
                                   xtol = xtol, ftol = ftol, grtol = grtol,
                                   iterations = iterations,
                                   store_trace = store_trace, show_trace = show_trace,
                                   extended_trace = extended_trace,
                                   verbose = verbose, rng = rng)
        callback_data = (times = iteration_times, posteriors = posterior_ls)
        converged = opt_result.g_converged || opt_result.f_converged #|| opt_result.x_converged
        out = optimization_result(opt_result.minimizer, opt_result.minimum, converged,
                                  opt_result.iterations)

    elseif method == :lbfgs || method == :conjugate_gradient || method == :cmaes || method == :xnes
        opt_result, iteration_times, posterior_ls, x_trace = optimizer(f_opt, x_opt;
                               xtol = xtol, ftol = ftol, grtol = grtol, iterations = iterations,
                               store_trace = store_trace, show_trace = show_trace,
                               extended_trace = extended_trace,
                               verbose = verbose, rng = rng, autodiff = false)
        callback_data = (trace = store_trace ? opt_result.trace : nothing,
                        times = iteration_times, posteriors = posterior_ls,
                        x_trace = store_trace ? x_trace : nothing)
        converged = opt_result.g_converged || opt_result.f_converged #|| opt_result.x_converged
        out = optimization_result(opt_result.minimizer, opt_result.minimum, converged,
                                  opt_result.iterations)
    elseif method == :trust_region_newton
        opt_result, iteration_times, posterior_ls, x_trace = optimizer(f_opt, x_opt;
                               xtol = xtol, ftol = ftol, grtol = grtol, iterations = iterations,
                               store_trace = store_trace, show_trace = show_trace,
                               extended_trace = extended_trace,
                               verbose = verbose, rng = rng)
        converged = opt_result.g_converged || opt_result.f_converged #|| opt_result.x_converged
        callback_data = (trace = store_trace ? opt_result.trace : nothing,
                        times = iteration_times, posteriors = posterior_ls, x_trace = store_trace ? x_trace : nothing)

        out = optimization_result(opt_result.minimizer, opt_result.minimum, converged,
                                  opt_result.iterations)
    elseif method == :pso
        global f_opt_global = f_opt
        opt_result = optimizer(f_opt2, x_opt, m;
                               xtol = xtol, ftol = ftol, grtol = grtol, iterations = iterations,
                               store_trace = store_trace, show_trace = show_trace,
                               extended_trace = extended_trace,
                               verbose = verbose, rng = rng)
        converged = opt_result.f_converged || opt_result.x_converged || opt_result.iteration_converged
        out = optimization_result(opt_result.minimizer, opt_result.minimum, converged,
                                  opt_result.iterations)
    elseif method == :combined_optimizer
        opt_result = optimizer(f_opt, x_opt;
                               xtol = xtol, ftol = ftol, grtol = grtol, iterations = iterations,
                               step_size = step_size, store_trace = store_trace,
                               show_trace = show_trace, extended_trace = extended_trace,
                               neighbor! = neighbor!, verbose = verbose, rng = rng,
                               temperature = temperature, max_cycles = max_cycles)
        converged = opt_result.g_converged || opt_result.f_converged || opt_result.x_converged
        converged = opt_result.method == "Simulated Annealing" ? opt_result.iteration_converged : converged
        out = optimization_result(opt_result.minimizer, opt_result.minimum, converged,
                                  opt_result.iterations)
    end

    ########################################################################################
    ### Step 4: transform output, populate Hessian
    ########################################################################################

    x_model[para_free_inds] = out.minimizer
    if typeof(m) <: AbstractDSGEVARModel
        transform_to_model_space!(m, x_model)
    else
        transform_to_model_space!(m, x_model; regime_switching = regime_switching)
    end
    # Match original dimensions
    out.minimizer = ModelConstructors.get_values(get_parameters(m); regime_switching = regime_switching)
    npara = regime_switching ? n_parameters_regime_switching(m) : n_parameters(m)
    H = zeros(npara, npara)
    if H_ != nothing

        # Fill in rows/cols of zeros corresponding to location of fixed parameters
        # For each row corresponding to a free parameter, fill in columns corresponding to
        # free parameters. Everything else is 0.
        for (row_free, row_full) in enumerate(para_free_inds)
            H[row_full, para_free_inds] = H_[row_free,:]
        end

    end

    return out, H, callback_data
end
