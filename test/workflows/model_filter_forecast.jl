using DSGE, HDF5, ModelConstructors, Random, Test

@testset "Model to simulated data, filter, and forecast workflow" begin
    model = AnSchorfheide(testing = true)
    model <= Setting(:forecast_horizons, 4)

    # Build the state-space system from the model, then simulate a short
    # deterministic-seed sample through the package's public data API.
    system = compute_system(model)
    Random.seed!(1793)
    mktempdir() do temp_dir
        filepath = joinpath(temp_dir, "simulated_data.h5")
        data = DSGE.simulate_data(model; filepath = filepath, burnin = 2,
                                  n_periods = 8, meas_error = 0.)
        persisted_data = h5read(filepath, "data")

        @test size(data) == (length(model.observables), 8)
        @test all(isfinite, data)
        @test persisted_data == data

        kalman = DSGE.filter(model, persisted_data, system)
        @test length(kalman[:loglh]) == size(data, 2)
        @test all(isfinite, kalman[:loglh])
        @test isapprox(kalman[:total_loglh], sum(kalman[:loglh]))
        @test size(kalman[:s_filt], 2) == size(data, 2)
    end

    initial_state = zeros(n_states_augmented(model))
    supplied_shocks = zeros(n_shocks_exogenous(model), forecast_horizons(model))
    states, observables, pseudo_observables, shocks =
        DSGE.forecast(model, system, initial_state; shocks = supplied_shocks)

    @test size(states, 2) == forecast_horizons(model)
    @test size(observables) == (length(model.observables), forecast_horizons(model))
    @test size(pseudo_observables, 2) == forecast_horizons(model)
    @test shocks == supplied_shocks
    @test all(isfinite, states)
    @test all(isfinite, observables)

    expected_states = similar(states)
    previous_state = initial_state
    for t in 1:forecast_horizons(model)
        previous_state = system[:CCC] + system[:TTT] * previous_state +
                         system[:RRR] * supplied_shocks[:, t]
        expected_states[:, t] = previous_state
    end
    @test states ≈ expected_states
    @test observables ≈ system[:DD] .+ system[:ZZ] * states
    @test pseudo_observables ≈ system[:DD_pseudo] .+ system[:ZZ_pseudo] * states
end
