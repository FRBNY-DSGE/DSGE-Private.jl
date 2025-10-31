"""
```
dsge_coint_likelihood(m::AbstractDSGEModel{S}, data::Matrix{S}) where {S<:Real}
```
evaluates the likelihood of a DSGE model with cointegrated data

The matrix `data` is assumed an n_coint+n_var x T+lags matrix,
where the lags indicate we cut off the data for presampling.
"""

function dsge_coint_likelihood(m, data)

    lh = 0
    lags = get_setting(m, :lags)

    if get_setting(m, :n_coint) > 0
        sys = compute_system(m)

        # Data preprocessing
        data_main = data[get_setting(m, :main_data_inds), :]
        data_coint = data[get_setting(m, :coint_data_inds), :]

        lh = dsge_coint_likelihood(data_main', data_coint', sys[:TTT], sys[:RRR], sys[:QQ],
                              sys[:ZZ], sys[:DD], sys[:EE], lags)
    else
        lh = likelihood(m, data)
    end

    return lh

end

function dsge_coint_likelihood(data, coint_data, TTT, RRR, QQQ, ZZ, DD, EE, lags)

    # Initialization
    n_states = size(TTT, 1)
    n_var    = size(ZZ, 1)
    n_coint  = size(coint_data, 2)

    # Obtain presample for initializing likelihood calculation
    data_presample = data[1:lags, :]
    coint_first_row = coint_data[1, :]
    data_init = [data_presample[1, :]; coint_first_row] # Transposed

    # Initial state and state-variance
    S_0 = zeros(n_states, 1)
    P_0 = MatrixEquations.lyapd(TTT, RRR*QQQ*RRR')

    ################### Run Kalman Filter #######################################

    ############## First run of Kalman Filter (on initial observations)
    outputs = DSGE.kalman_filter(data_init, TTT, RRR, vec(zeros(n_states, 1)), QQQ,
                                 ZZ, vec(DD), EE, vec(S_0), P_0, outputs = [:loglh, :pred])
    #loglh = outputs[1]
    #S_T   = outputs[8]
    #P_T   = outputs[9]

    ############## Run Kalman Filter on rest of presample:

    # (1) Remove cointegration component from system
    n_var = n_var - n_coint

    ZZ = ZZ[1:n_var, :]
    DD = DD[1:n_var, :]
    EE = EE[1:n_var, 1:n_var]

    # (2) Run kalman filter on rest of lags of main data
    outputs = DSGE.kalman_filter(data_presample[2:end, :]', TTT, RRR, vec(zeros(n_states, 1)), QQQ,
                                 ZZ, vec(DD), EE, vec(outputs[8]), outputs[9], outputs = [:loglh, :pred])


    ############## Run Kalman Filter on main sample:
    data_mainsample = data[lags+1:end, :]
    outputs = DSGE.kalman_filter(data_mainsample', TTT, RRR, vec(zeros(n_states, 1)), QQQ,
                                 ZZ, vec(DD), EE, vec(outputs[8]), outputs[9], outputs = [:loglh, :pred])

    loglh = sum(outputs[1])

    return loglh
end
