function pso_meta(f_opt, x0, m; n_iter = n_iter)


    n_particles = 200

    para_free_inds = ModelConstructors.get_free_para_inds(DSGE.get_parameters(m))
    n_free_params = length(para_free_inds)

    lb = zeros(n_free_params)
    ub = zeros(n_free_params)

    for i in 1:n_free_params
        lb[i] = m.parameters[para_free_inds[i]].valuebounds[1]
        ub[i] = m.parameters[para_free_inds[i]].valuebounds[2]
    end

    # Create bounds matrix for Metaheuristics
    bounds = [lb'; ub']

    options = Options(
        iterations = n_iter,              # Maximum iterations (equivalent to MaxSteps)
        parallel_evaluation = false,     # Enable multi-threading
        f_calls_limit = 30000,         # Maximum function evaluations
        f_tol = 1e-6                 # Convergence tolerance                # Set to true for detailed progress
    )

    pso_algorithm = PSO(N=n_particles, C1=2.0, C2=2.0, ω=0.7; options)
    set_user_solutions!(pso_algorithm, x0, f_opt)

    result = Metaheuristics.optimize(f_opt, bounds, pso_algorithm)

    println("\nOptimization complete!")
    println("Best candidate: ", result.best_sol.x)
    println("Fitness: ", result.best_sol.f)

end
