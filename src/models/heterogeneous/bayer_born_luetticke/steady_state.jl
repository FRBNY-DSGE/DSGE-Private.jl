function steadystate!(m::BayerBornLuetticke; verbose::Symbol = :none)
    # See helper_functions/HetAgentDSGE

    # TODO: add ability to pass initial guesses very robustly
    # Compute steady state
    KSS, VmSS, VkSS, distrSS = find_steadystate(m; verbose = verbose,
                                                skip_coarse_grid = haskey(get_settings(m), :skip_coarse_grid) &&
                                                get_setting(m, :skip_coarse_grid))

    # Update steady-state parameters, reduce state space, and update indices
    prepare_linearization(m, KSS, VmSS, VkSS, distrSS; verbose = verbose)

    m
end
