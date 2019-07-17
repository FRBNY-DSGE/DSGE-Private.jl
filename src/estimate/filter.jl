"""
```
filter(m, data, system, s_0 = [], P_0 = []; cond_type = :none,
       include_presample = true, in_sample = true,
       outputs = [:loglh, :pred, :filt])
```

Computes and returns the filtered values of states for the state-space
system corresponding to the current parameter values of model `m`.

### Inputs

- `m::AbstractModel`: model object
- `data`: `DataFrame` or `nobs` x `hist_periods` `Matrix{S}` of data for
  observables. This should include the conditional period if `cond_type in
  [:semi, :full]`
- `system::System`: `System` object specifying state-space system matrices for
  the model
- `s_0::Vector{S}`: optional `Nz` x 1 initial state vector
- `P_0::Matrix{S}`: optional `Nz` x `Nz` initial state covariance matrix

where `S<:AbstractFloat`.

### Keyword Arguments

- `cond_type::Symbol`: conditional case. See `forecast_all` for documentation of
  all `cond_type` options
- `include_presample::Bool`: indicates whether to include presample periods in
  the returned vector of `Kalman` objects
- `in_sample::Bool`: indicates whether or not to discard out of sample rows in
    `df_to_matrix` call
- `outputs::Vector{Symbol}`: which Kalman filter outputs to compute and return.
  See `?kalman_filter`

### Outputs

- `kal::Kalman`: see `?Kalman`
"""
function filter(m::AbstractModel, df::DataFrame, system::System{S},
                s_0::Vector{S} = Vector{S}(undef, 0),
                P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
                cond_type::Symbol = :none, include_presample::Bool = true,
                in_sample::Bool = true,
                outputs::Vector{Symbol} = [:loglh, :pred, :filt],
                tol::Float64 = 0.0) where {S<:AbstractFloat}

    data = df_to_matrix(m, df; cond_type = cond_type, in_sample = in_sample)
    start_date = max(date_presample_start(m), df[1, :date])
    filter(m, data, system, s_0, P_0; start_date = start_date,
           include_presample = include_presample, outputs = outputs, tol = tol)
end

function filter(m::AbstractModel, data::AbstractArray, system::System{S},
                s_0::Vector{S} = Vector{S}(undef, 0),
                P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
                start_date::Date = date_presample_start(m),
                include_presample::Bool = true,
                outputs::Vector{Symbol} = [:loglh, :pred, :filt],
                tol::Float64 = 0.0) where {S<:AbstractFloat}

    T = size(data, 2)

    # Partition sample into pre- and post-ZLB regimes
    # Note that the post-ZLB regime may be empty if we do not impose the ZLB
    regime_inds = zlb_regime_indices(m, data, start_date)

    # Get system matrices for each regime
    TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs = zlb_regime_matrices(m, system, start_date)

    # If the end of the first regime is assumed to be later than the data provided
    # change the regime to be up to the length of the data
    if regime_inds[1][end] > T
        regime_inds = [1:T]
        TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs = [TTTs[1]], [RRRs[1]], [CCCs[1]], [QQs[1]], [ZZs[1]], [DDs[1]], [EEs[1]]
    end

    # If s_0 and P_0 provided, check that rows and columns corresponding to
    # anticipated shocks are zero in P_0
    if !isempty(s_0) && !isempty(P_0)
        ant_state_inds = setdiff(1:n_states_augmented(m), inds_states_no_ant(m))
        @assert all(x -> x == 0, P_0[:, ant_state_inds])
        @assert all(x -> x == 0, P_0[ant_state_inds, :])
    end

    # Specify number of presample periods if we don't want to include them in
    # the final results
    Nt0 = include_presample ? 0 : n_presample_periods(m)

    # Run Kalman filter, construct Kalman object, and return
    out = kalman_filter(regime_inds, data, TTTs, RRRs, CCCs, QQs,
                        ZZs, DDs, EEs, s_0, P_0; outputs = outputs,
                        Nt0 = Nt0, tol = tol)
    return Kalman(out...)
end

function filter_likelihood(m::AbstractModel, df::DataFrame, system::System{S},
                           s_0::Vector{S} = Vector{S}(undef, 0),
                           P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
                           cond_type::Symbol = :none, include_presample::Bool = true,
                           in_sample::Bool = true,
                           tol::Float64 = 0.0) where {S<:AbstractFloat}

    data = df_to_matrix(m, df; cond_type = cond_type, in_sample = in_sample)
    start_date = max(date_presample_start(m), df[1, :date])

    filter_likelihood(m, data, system, s_0, P_0; start_date = start_date,
                      include_presample = include_presample, tol = tol)
end

function filter_likelihood(m::AbstractModel, data::Matrix{S}, system::System{S},
                           s_0::Vector{S} = Vector{S}(undef, 0),
                           P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
                           start_date::Date = date_presample_start(m),
                           include_presample::Bool = true,
                           tol::Float64 = 0.0) where {S<:AbstractFloat}

    # Partition sample into pre- and post-ZLB regimes
    # Note that the post-ZLB regime may be empty if we do not impose the ZLB
    regime_inds = zlb_regime_indices(m, data, start_date)

    # Get system matrices for each regime
    TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs = zlb_regime_matrices(m, system, start_date)

    # If s_0 and P_0 provided, check that rows and columns corresponding to
    # anticipated shocks are zero in P_0
    if !isempty(s_0) && !isempty(P_0)
        ant_state_inds = setdiff(1:n_states_augmented(m), inds_states_no_ant(m))
        @assert all(x -> x == 0, P_0[:, ant_state_inds])
        @assert all(x -> x == 0, P_0[ant_state_inds, :])
    end

    # Specify number of presample periods if we don't want to include them in
    # the final results
    Nt0 = include_presample ? 0 : n_presample_periods(m)

    # Run Kalman filter, construct Kalman object, and return
    kalman_likelihood(regime_inds, data, TTTs, RRRs, CCCs, QQs,
                      ZZs, DDs, EEs, s_0, P_0; Nt0 = Nt0, tol = tol)
end


"""
```
This section defines filter and filter_likelihood for the PoolModel type
```
"""
function filter(m::PoolModel, data::AbstractArray = Matrix{Float64}(undef,0,0),
                s_0::Matrix{S} = Matrix{Float64}(undef,0,0);
                start_date::Date = date_presample_start(m),
                cond_type::Symbol = :none, include_presample::Bool = true,
                in_sample::Bool = true,
                tol::Float64 = 0., parallel::Bool = false,
                tuning::Dict{Symbol,Any} = Dict{Symbol,Any}()) where {S<:AbstractFloat}

    # Handle data
    if isempty(data)
        data = zeros(1,get_periods(m))
    end
    Nt0 = include_presample ? 0 : n_presample_periods(m)
    if haskey(tuning, :n_presample_periods)
        tuning[:n_presample_periods] = Nt0
    end

    # Check initial states
    n_particles = haskey(tuning, :n_particles) ? tuning[:n_particles] : 1000
    if isempty(s_0)
        s_0 = reshape(rand(get_F_λ(m), n_particles), 1, 1000)
        s_0 = [s_0; 1 .- s_0]
    else
        if size(s_0,2) != n_particles
            error("s0 does not contain enough particles")
        end
    end

    # Check tuning
    if isempty(tuning)
        try
            tuning = get_setting(pm, :tuning)
        catch
            warn("no tuning parameters provided; using default tempered particle filter values")
        end
    end
    if haskey(tuning, :parallel) # contains parallel? change keyword to that if so
        parallel = tuning[:parallel]
    end

    # Check if PoolModel has fixed_sched. If not, assume no tempering
    fixed_sched = [1.]
    try
        fixed_sched = get_setting(m, :fixed_sched)
    catch KeyError
    end

    return tempered_particle_filter(data, get_Φ(m), get_Ψ(m), get_F_ϵ(m), get_F_u(m),
                                    s_0; parallel = parallel,
                                    dynamic_measurement = true, poolmodel = true,
                                    fixed_sched = fixed_sched,
                                    tuning..., verbose = :none)
end


function filter_likelihood(m::PoolModel, data::AbstractArray = Matrix{Float64}(undef,0,0),
                           s_0::Matrix{S} = Matrix{Float64}(undef,0,0);
                           start_date::Date = date_presample_start(m),
                           cond_type::Symbol = :none, include_presample::Bool = true,
                           in_sample::Bool = true, parallel::Bool = false, tol::Float64 = 0.,
                           tuning::Dict{Symbol,Any} = Dict{Symbol,Any}()) where {S<:AbstractFloat}

    ~, loglhconditional, ~ = filter(m, data, s_0; start_date = start_date,
                                    include_presample = include_presample,
                                    cond_type = cond_type, in_sample = in_sample, tol = tol,
                                    parallel = parallel, tuning = tuning)
    return loglhconditional
end
