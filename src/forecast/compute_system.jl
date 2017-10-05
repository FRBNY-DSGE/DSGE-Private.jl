
function compute_system(m)

    # Solve model
    TTT, RRR, CCC = solve(m)
    transition_equation = Transition(TTT, RRR, CCC)

    # Solve measurement equation
    shocks = n_anticipated_shocks(m) > 0

    # Load data if m has forcing processes
    if n_forcing_processes(m) > 0

        df = load_data(m)
        forcing_ind = get_setting(m, :forcing_index_start)
        data = df_to_matrix(m,df)
        X = data[forcing_ind:end,:]
        measurement_equation = measurement(m, TTT, RRR, CCC, X; shocks = shocks)
    else
        # Solve measurement equation
        measurement_equation = measurement(m, TTT, RRR, CCC; shocks = shocks)
    end

    return System(transition_equation, measurement_equation)

end
