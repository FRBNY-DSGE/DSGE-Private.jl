
function compute_system(m; use_expected_rate_data=false)

    # Solve model
    TTT, RRR, CCC = solve(m)

    # Index out non-anticipated shocks rows and columns of system matrices if desired
    if !use_expected_rate_data
        n_states     = DSGE.n_states(m)
        n_ant        = n_anticipated_shocks(m)
        n_states_aug = DSGE.n_states_augmented(m)
        n_exo        = n_shocks_exogenous(m)
        
        state_inds = [1:(n_states-n_ant); (n_states+1):n_states_aug]
        shock_inds = 1:(n_exo-n_ant)
        
        TTT = TTT[state_inds, state_inds]
        RRR = RRR[state_inds, shock_inds]
        CCC = CCC[state_inds]
    end
    
    transition_equation = Transition(TTT, RRR, CCC)

    # Load data if m is a reduced-form model
    if n_forcing_processes(m) > 0
        #print("right place\n")
        df = load_data(m)
        forcing_ind = get_setting(m, :forcing_index_start)
        data = df_to_matrix(m,df)
        X = data[forcing_ind:end,inds_presample_periods(m)]
        measurement_equation = measurement(m, TTT, RRR, CCC, X; shocks = use_expected_rate_data)
    else
        # Solve measurement equation
        measurement_equation = measurement(m, TTT, RRR, CCC; shocks = use_expected_rate_data)
    end
    
    return System(transition_equation, measurement_equation)
    
end
