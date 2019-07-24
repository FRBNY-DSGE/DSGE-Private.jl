"""
```
PoolModel{T}
```

The PoolModel type permits dynamic predictive pooling of structural models.

### Fields

#### Parameters
* `parameters::Vector{AbstractParameter}`: Vector of all time-invariant model
  parameters.

* `keys::OrderedDict{Symbol,Int}`: Maps human-readable names for all model
  parameters.

#### Inputs to Measurement and Equilibrium Condition Equations

* `model::OrderedDict{Symbol,AbstractModel}`: Maps name to its underlying model
  object.

#### Model Specifications and Settings

* `spec::String`: The model specification identifier, \"an_schorfheide\", cached
  here for filepath computation.

* `subspec::String`: The model subspecification number, indicating that some
  parameters from the original model spec (\"ss0\") are initialized
  differently. Cached here for filepath computation.

* `settings::Dict{Symbol,Setting}`: Settings/flags that affect computation
  without changing the economic or mathematical setup of the model.

* `test_settings::Dict{Symbol,Setting}`: Settings/flags for testing mode

#### Other Fields

* `rng::MersenneTwister`: Random number generator. Can be is seeded to ensure
  reproducibility in algorithms that involve randomness (such as
  Metropolis-Hastings).

* `testing::Bool`: Indicates whether the model is in testing mode. If `true`,
  settings from `m.test_settings` are used in place of those in `m.settings`.

* `observable_mappings::OrderedDict{Symbol,Observable}`: A dictionary that
  stores data sources, series mnemonics, and transformations to/from model units.
  DSGE.jl will fetch data from the Federal Reserve Bank of
  St. Louis's FRED database; all other data must be downloaded by the
  user. See `load_data` and `Observable` for further details.

* `pseudo_observable_mappings::OrderedDict{Symbol,PseudoObservable}`: A
  dictionary that stores names and transformations to/from model units. See
  `PseudoObservable` for further details.
"""
mutable struct PoolModel{T} <: AbstractModel{T}
    parameters::ParameterVector{T}                         # vector of all time-invariant model parameters
    steady_state::ParameterVector{T}
    keys::OrderedDict{Symbol,Int}                          # human-readable names for all the model
                                                           # parameters and steady-states
    observables::OrderedDict{Symbol,Int}
    pseudo_observables::OrderedDict{Symbol,Int}
    models::OrderedDict{Symbol,AbstractModel{T}}           # Model name mapped to model object
    datas::OrderedDict{Symbol,Matrix{T}}                   # Model name " "
    forecast_horizon::Int                                  # Number of periods for forecast
    periods::Int                                           # Number of periods for data time series
    cond_pred_dens::OrderedDict{Symbol, Vector{T}}            # Model name mapped to cond_pred_dens
    statespace::Dict{Symbol,Function}                      # Transition equation for linear weights
                                                           # Measurement eq for linear weights
    distributions::Dict{Symbol,Distribution}               # Distributions for state space

    spec::String                                           # Model specification number (eg "m990")
    subspec::String                                        # Model subspecification (eg "ss0")
    settings::Dict{Symbol,Setting}                         # Settings/flags for computation
    test_settings::Dict{Symbol,Setting}                    # Settings/flags for testing mode
    rng::MersenneTwister                                   # Random number generator
    testing::Bool                                          # Whether we are in testing mode or not

    observable_mappings::OrderedDict{Symbol, Observable}
    pseudo_observable_mappings::OrderedDict{Symbol, PseudoObservable}
end

description(m::PoolModel) = "Julia implementation of dynamic prediction pools defined in 'Dynamic prediction pools: An investigation of financial frictions and forecasting performance' by Marco Del Negro, Raiden B. Hasegawa, and Frank Schorfheide: PoolModel, $(m.subspec)"

# function PoolModel(data::Matrix{T}, h::Int, cond_pred_dens::Dict{Symbol,Vector{T}},
#                    models::Vector{<:AbstractModel{T}}, subspec::String="ss0";
#                    custom_settings::Dict{Symbol,Setting} = Dict{Symbol,Setting}(),
#                    testing = false, verbose::Symbol = :low,
#                    static::Bool = false) where T<:AbstractFloat
#     data_dict = Dict(name => data for name in keys(cond_pred_dens))
#     return PoolModel(data_dict, h, cond_pred_dens, models, subspec;
#                    custom_settings = custom_settings,
#                    testing = testing, verbose = verbose, static = static)
# end
# function PoolModel(datas::Dict{Symbol,Matrix{T}}, h::Int, pred_dens::Dict{Symbol,Vector{T}},
#                    models::Vector{<:AbstractModel{T}}, subspec::String="ss0";
#                    custom_settings::Dict{Symbol,Setting} = Dict{Symbol,Setting}(),
#                    testing = false, verbose::Symbol = :low,
#                    static::Bool = false) where T<:AbstractFloat
#     if length(data) > length(models)
#         error("number of data series exceeds number of models")
#     else
#         error("number of models exceeds number of data series")
#     end
#     name_vec = [Symbol(typeof(model)) for model in models]
#     try
#         data_vec = [datas[name] for name in name_vec]
#     catch
#         error("at least one key in data dictionary does not match any model")
#     end
#     return PoolModel(data_vec, h, models, subspec; custom_settings = custom_settings,
#                    testing = testing, verbose = verbose, static = static)
# end
# function PoolModel(data::Matrix{T}, h::Int, subspec::String="ss0", models::AbstractModel{T}...;
#                    custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
#                    testing::Bool = false, verbose::Bool = :low) where T<:AbstractFloat
#     return PoolModel(subspec, [data for i = 1:length(models)], h, [model for model in models];
#                    custom_settings = custom_settings, testing = testing, verbose = verbose)
# end
# function PoolModel(datas::Vector{Matrix{T}}, h::Int, subspec::String="ss0",
#                    models::AbstractModel{T}...;
#                    custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
#                    testing::Bool = false, verbose::Bool = :low) where T<:AbstractFloat
#     return PoolModel(subspec, datas, h, [model for model in models];
#                    custom_settings = custom_settings, testing = testing, verbose = verbose)
# end
# function PoolModel(datas::Dict{Symbol,Matrix{T}}, h::Int, subspec::String="ss0",
#                    models::AbstractModel{T}...;
#                    custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
#                    testing::Bool = false, verbose::Bool = :low) where T<:AbstractFloat
#     return PoolModel(subspec, datas, h, [model for model in models];
#                    custom_settings = custom_settings, testing = testing, verbose = verbose)
# end
function PoolModel(datas::Dict{Symbol,Matrix{T}}, h::Int, cond_pred_dens::Dict{Symbol,Vector{T}},
                   models::Vector{<:AbstractModel{T}}, subspec::String="ss0";
                   custom_settings::Dict{Symbol,Setting} = Dict{Symbol,Setting}(),
                   testing = false, verbose::Symbol = :low,
                   static::Bool = false) where T<:AbstractFloat

    # Model-specific specifications
    spec               = split(basename(@__FILE__),'.')[1]
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = PoolModel{Float64}(
        # model parameters and steady state values
        Vector{AbstractParameter{Float64}}(), Vector{Float64}(), OrderedDict{Symbol,Int}(),

        # model indices
        OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(),

        # Models-related data
        OrderedDict{Symbol,AbstractModel{Float64}}(), OrderedDict{Symbol,Matrix{Float64}}(),
        0, 0, OrderedDict{Symbol, Vector{Float64}}(),

        # nonlinear hidden state model info
        Dict{Symbol,Function}(), Dict{Symbol,Distribution}(),

        # settings
        spec,
        subspec,
        settings,
        test_settings,
        rng,
        testing,
        OrderedDict{Symbol,Observable}(),
        OrderedDict{Symbol,PseudoObservable}())

    # Set settings
    model_settings!(m)
    default_test_settings!(m)
    for custom_setting in values(custom_settings)
        m <= custom_setting
    end

    # Set observable and pseudo-observable transformations
    # init_observable_mappings!(m)
    # init_pseudo_observable_mappings!(m)

    # Initialize parameters
    init_parameters!(m; static = static)

    # Initialize model indices and subspec
    # init_model_indices!(m)
    init_subspec!(m)
    steadystate!(m)

    # Initialize models dictionary
    init_models!(m, models)

    # Initialize datas dictionary
    init_datas!(m, datas)

    # Initialize forecast horizong
    m.forecast_horizon = h

    # Initialize particle clouds
    # init_particles!(m)

    # Initialize conditional predictive densities
   if testing
        fill_vec = Vector{Float64}(undef,m.periods)
        for name in keys(m.models)
            m.cond_pred_dens[name] = fill_vec
        end
   else
        init_cond_pred_dens!(m, cond_pred_dens)
   end

    # Initialize state space equations
    init_statespace!(m)

    # Initialize distributions for state space
    init_distributions!(m)

    return m
end

"""
```
init_parameters!(m::PoolModel)
```

Initializes the model's parameters, as well as empty values for the steady-state
parameters (in preparation for `steadystate!(m)` being called to initialize
those).
"""
function init_parameters!(m::PoolModel; static::Bool = false)
    # Initialize parameters
    if static
        m <= parameter(:ρ, 1., fixed = true,
                       description="ρ: persistence of AR processing underlying λ.",
                       tex_label="\\rho")
        m <= parameter(:μ, 0., fixed = true,
                       description="μ: drift of AR processing underlying λ.",
                       tex_label="\\mu")
        m <= parameter(:σ, 1., fixed = true,
                       description="σ: volatility of AR processing underlying λ.",
                       tex_label="\\sigma")
    else
        m <= parameter(:ρ, 0.5, (1e-5,0.999), (1e-5,0.999), SquareRoot(), Uniform(0.,1.), fixed = false,
                       description="ρ: persistence of AR processing underlying λ.",
                       tex_label="\\rho")
        m <= parameter(:μ, 0., fixed = true,
                       description="μ: drift of AR processing underlying λ.",
                       tex_label="\\mu")
        m <= parameter(:σ, 1., fixed = true,
                       description="σ: volatility of AR processing underlying λ.",
                       tex_label="\\sigma")
    end
end
"""
```
steadystate!(m::PoolModel)
```

Calculates the model's steady-state values. `steadystate!(m)` must be called whenever
the parameters of `m` are updated.
"""
function steadystate!(m::PoolModel)
    return m
end


function model_settings!(m::PoolModel)
    default_settings!(m)

    # Data
    m <= Setting(:data_id, 0, "Dataset identifier")
    m <= Setting(:cond_full_names, [:obs_gdp, :obs_nominalrate],
        "Observables used in conditional forecasts")
    m <= Setting(:cond_semi_names, [:obs_nominalrate],
        "Observables used in semiconditional forecasts")

    # Metropolis-Hastings
    m <= Setting(:mh_cc, 0.27,
                 "Jump size for Metropolis-Hastings (after initialization)")

    # Forecast
    m <= Setting(:use_population_forecast, true,
                 "Whether to use population forecasts as data")

    # Tempered particle filter
    m <= Setting(:fixed_sched, [1.],
                 "schedule for tempering in tpf; leave empty if want adaptive tempering")
    tuning = Dict(:r_star => 2., :c_init => 0.3, :target_accept_rate => 0.4,
              :resampling_method => :systematic, :n_mh_steps => 1,
              :n_particles => 1000, :n_presample_periods => 0,
              :allout => true)
    m <= Setting(:tuning, tuning, "tuning parameters for TPF")

end

# """
# `init_model_indices!(m::PoolModel)`

# Arguments:
# `m:: PoolModel`: a model object

# Description:
# Initializes indices for all of `m`'s states, shocks, and equilibrium conditions.
# """
# function init_model_indices!(m::PoolModel)
#     # Observables
#     observables = keys(m.observable_mappings)

#     # Pseudo-observables
#     pseudo_observables = keys(m.pseudo_observable_mappings)

#     # Collect into model indices
#     for (i,k) in enumerate(observables);                 m.observables[k]                 = i end
#     for (i,k) in enumerate(pseudo_observables);          m.pseudo_observables[k]          = i end
# end

"""
`init_statespace!(m::PoolModel)`

Creates transition and measurement equations as passable functions.
"""
function init_statespace!(m::PoolModel{T}) where T<:AbstractFloat
    # transition equation
    @inline Φ(x::Vector{T}, ϵ::Vector{T}) = abs.([0;1] .- (cdf.(Normal(), (1 - m[:ρ]) * m[:μ] +
                                    m[:ρ] * quantile(Normal(),x[1]) +
                                    sqrt(1 - m[:ρ]^2) * m[:σ] * ϵ)))
    m.statespace[:Φ] = Φ

    # measurement equation
    pred_dens_mat = zeros(m.periods, length(m.models)) # matrix of conditional predictive densities
    for (i,key) in enumerate(keys(m.models))
        pred_dens_mat[:,i] = m.cond_pred_dens[key]
    end
    pred_dens_mat = reshape(pred_dens_mat, length(m.models), m.periods)

    @inline Ψ(x::Vector{T}, t::Int64) = dot(pred_dens_mat[:,t], x)
    m.statespace[:Ψ] = Ψ

    return m
end

function init_distributions!(m::PoolModel)
    # exogenous shock
    m.distributions[:F_ϵ] = Normal(0.,1.)

    # measurement error
    m.distributions[:F_u] = DiscreteUniform(0,0)

    # initialization distribution for lambda
    m.distributions[:F_λ] = Uniform(0.,1.)

    return m
end

function init_models!(m::PoolModel, models::Vector{<:AbstractModel{T}} = Vector{<:AbstractModel{T}}()) where T<:AbstractFloat
    for model in models
        name = replace(String(Symbol(typeof(model))), "{Float64}" => "")
        m.models[Symbol(name)] = model
    end
    return m
end

function init_datas!(m::PoolModel, datas::Dict{Symbol,Matrix{T}}) where T<:AbstractFloat
    S = minimum([size(data,2) for data in values(datas)])
    m.periods = S
    for (name,data) in zip(keys(m.models),values(datas))
        m.datas[name] = data
        S1 = size(data, 2)
        if S != S1
            error("Data time series must be the same length and assumed to start and end at same dates.")
        end
    end
    return m
end

# function init_particles!(m::PoolModel; names::Vector{Symbol} = Vector{Symbol}())
#     if isempty(names)
#         for kv in m.models
#             load(rawpath(kv[2], "estimate", "smc_cloud.jld2"))
#             m.particles[kv[1]] = cloud # need to check this is the correct name
#         end
#     else
#         for name in names
#             load(rawpath(m.models[name], "estimate", "smc_cloud.jld2"))
#             m.particles[name] = cloud
#         end
#     end

#     return m
# end

function init_cond_pred_dens!(m::PoolModel, cond_pred_dens::Dict{Symbol,Vector{T}}) where T<:AbstractFloat
    for kv in cond_pred_dens
        try
            m.cond_pred_dens[kv[1]] = kv[2]
        catch
            @warn "model named " * String(kv[1]) * " not found"
        end
    end
    return m
end

# function init_cond_pred_dens!(m::PoolModel; names::Vector{Symbol} = Vector{Symbol}(),
#                       verbose::Symbol = :low)
#     if isempty(names)
#         names = keys(m.models)
#     end
#     Nt = m.periods # since predict h periods ahead
#     for name in names
#         θs = load_draws(m.models[name], :full) # matrix of posterior draws, represents whole posterior
#         Nθ = length(θs)
#         Ns = size(compute_system(m)[:TTT],1)
#         cond_pred_dens = zeros(Nt, Nθ) # matrix of period t conditional pred_dens (on t-1 information set)
#         m.cond_pred_dens[name] = @sync @distributed (+) for θi in 1:Nθ
#             # Evaluate T, R, Z, D given theta
#             update!(m.models[name], θs[θi])
#             TTT, RRR, CCC = solve(m.models[name])
#             tmp = measurement(m.models[name], TTT, RRR, CCC)
#             QQ = tmp[:QQ]
#             ZZ = tmp[:ZZ]
#             DD = tmp[:DD]
#             EE = tmp[:EE]

#             # Precompute matrices
#             TTT_power = Dict{Int,typeof(TTT)}(1 => TTT)
#             for i in 2:m.h
#                 TTT_power[i] = TTT_power[i-1] * TTT
#             end
#             TTTtp_power = Dict(i => TTT_power[i]' for i in 1:m.h)
#             Dtild = kron(ones(m.h+1),DD)
#             Ztild = kron(Matrix(I*1.,m.h+1,m.h+1),ZZ)

#             # Run Kalman filter
#             k   = KalmanFilter(TTT, RRR, CCC, QQQ, ZZ, DD, EE)
#             SS_t = zeros(Ns * (m.h+1)) # initialize s_{t:t+h|t-1} vector
#             PP_t = zeros(Ns * (m.h+1), NS * (m.h+1)) # initialize P_{t:t+h|t-1} matrix
#             s_0 = k.s_t
#             P_0 = k.P_t
#             try
#                 semi_names = get_setting(m.models[name], :cond_semin_names)
#                 do_semi = true
#             catch
#                 do_semi = false
#             end

#             cond_loglh_θ = zeros(Nt) # vector of conditional pred_dens (on t-1 information set) and fixing θ
#             for t in 1:Nt
#                 # Compute unconditional forecast of time t
#                 DSGE.forecast!(k)

#                 # Compute semiconditional forecast
#                 if do_semi
#                     obs = get_dict(m.models[name], :obs)
#                     inds = [obs[semi_name] for semi_name in semi_names]
#                     update!(k, m.datas[name][inds,t])
#                 end

#                 # Recursively forecast by j = 1:h periods
#                 SS_t[1:Ns] = k.s_t
#                 PP_t[1:NS, 1:Ns] = k.P_t
#                 for j in 1:m.h
#                     PP_t[1:Ns, 1+Ns*j:Ns*(j+1)] = k.P_t * TTTtp_power[j]
#                     PP_t[1+Ns*j:Ns*(j+1)] = TTT_power[j] * k.P_t
#                 end
#                 for j in 1:m.h
#                     forecast!(k)
#                     indices = 1+Ns*j:Ns*(j+1)
#                     SS_t[indices] = k.s_t
#                     PP_t[indices, indices] = k.P_t
#                     for n in 1:m.h-j
#                         PP_t[indices,1+Ns*(n+j):Ns*(n+j+1)] = k.P_t * TTTtp_power[n]
#                         PP_t[1+Ns*(n+j):Ns*(n+j+1),indices] = TTT_power[n] * k.P_t
#                     end
#                 end

#                 # Compute conditional log likelihood p(y_t|θ, I_{t-1})
#                 μ_mv = Dtild + Ztild * SS_t
#                 Σ_mv = Ztild * PP_t * Ztild'
#                 inv_Σ_mv = inv(Σ_mv)
#                 det_Σ = det(mv)
#                 err = vec(datas[name][:,t:t+m.h]) - μ_mv
#                 cond_loglh_θ[t] = (2*pi)^(-length(err)/2) * det_Σ^(-1/2) * exp(-(1/2) * dot(err, inv_Σ * err))
#             end
#             cond_pred_dens[:,θi] = cond_loglh_θ

#         end
#         m.cond_pred_dens[name] ./= Nθ
#     end

#     return m
# end
"""
```
Access and update functions for PoolModel fields. Append and pop functions not added yet
because functionality may not be desirable. Instead, we may want the user to simply
construct a new PoolModel object.
```
"""
function get_models(m::PoolModel, names::Symbol)
    return get_models(m, [names])
end
function get_models(m::PoolModel, names::Vector{Symbol} = Vector{Symbol}())
    if isempty(names)
        return m.models
    else
        return Dict(name => m.models[name] for name in names)
    end
end
function get_datas(m::PoolModel, names::Symbol)
    return get_datas(m, [names])
end
function get_datas(m::PoolModel, names::Vector{Symbol} = Vector{Symbol}())
    if isempty(names)
        return m.datas
    else
        return Dict(name => m.datas[name] for name in names)
    end
end
function get_periods(m::PoolModel)
    return m.periods
end
function get_forecast_horizon(m::PoolModel)
    return m.forecast_horizon
end
# function get_particles(m::PoolModel, names::Symbol)
#     return get_particles(m, [names])
# end
# function get_particles(m::PoolModel, names::Vector{Symbol} = Vector{Symbol}())
#     if isempty(names)
#         return m.particles
#     else
#         return OrderedDict(name => m.particles[name] for name in names)
#     else
# end
function get_cond_pred_dens(m::PoolModel, name::Symbol)
    return m.cond_pred_dens[name]
end
function get_cond_pred_dens(m::PoolModel, names::Vector{Symbol} = Vector{Symbol}())
    if isempty(names)
        return m.cond_pred_dens
    else
        return Dict(name => m.cond_pred_dens[name] for name in names)
    end
end
function get_system(m::PoolModel)
    return Dict(:statespace => m.statespace, :distributions => m.distributions)
end
function get_statespace(m::PoolModel, F::Symbol = :all)
    if F == :all
        return m.statespace
    else
        return m.statespace[F]
    end
end
function get_distributions(m::PoolModel, F::Symbol = :all)
    if F == :all
        return m.distributions
    else
        return m.distributions[F]
    end
end
function get_Φ(m::PoolModel)
    return m.statespace[:Φ]
end
function get_Ψ(m::PoolModel)
    return m.statespace[:Ψ]
end
function get_F_ϵ(m::PoolModel)
    return m.distributions[:F_ϵ]
end
function get_F_u(m::PoolModel)
    return m.distributions[:F_u]
end
function get_F_λ(m::PoolModel)
    return m.distributions[:F_λ]
end


function update_models!(m::PoolModel, models::AbstractModel{T}...;
                        populate::Bool = true) where T<:AbstractFloat
    if length(models) > 1
        update_models!(m, [model for model in models]; populate = populate)
    else
        update_models!(m, [models]; populate = populate)
    end
end
function update_models!(m::PoolModel, models::Vector{<:AbstractModel{T}};
                        populate::Bool = true) where T<:AbstractFloat
    names = Vector{Symbol}(undef,length(models))
    for (i,model) in enumerate(models)
        name = replace(String(Symbol(typeof(model))), "{Float64}" => "")
        names[i] = Symbol(name)
    end
    update_models!(m, Dict(names[i] => model for (i,model) in enumerate(models));
                   populate = populate)
    return nothing
end
function update_models!(m::PoolModel, models::Dict{Symbol,AbstractModel{T}};
                        populate::Bool = true) where T<:AbstractFloat
    for kv in models
        if haskey(models,kv[1])
            m.models[kv[1]] = kv[2]
        else
            @warn "no model named " * String(kv[1]) * " found"
        end
    end
    if populate
        names = Vector(keys(models))
        # init_particles!(m; model_keys)
        # init_cond_pred_dens!(m; names = names)
    end
    return nothing
end
function update_datas!(m::PoolModel, datas::Dict{Symbol,Matrix{T}}) where T<:AbstractFloat
    for kv in datas
        try
            m.datas[kv[1]] = kv[2]
        catch
            @warn "no model named " * String(kv[1]) * " found"
        end
    end
    return nothing
end
# function update_particles!(m::PoolModel, p::Dict{Symbol,ParticleCloud})
#     for kv in p
#         try
#             m.particles[kv[1]] = kv[2]
#         catch
#             @warn "no model named " * String(kv[1]) * " found"
#         end
#     end
#     return nothing
# end
function update_cond_pred_dens!(m::PoolModel,
                             cond_pred_dens::Dict{Symbol,Vector{T}}) where T<:AbstractFloat
    for kv in cond_pred_dens
        try
            m.cond_pred_dens[kv[1]] = kv[2]
        catch
            @warn "no model named " * String(kv[1]) * " found"
        end
    end
    return nothing
end
function update_statespace!(m::PoolModel, statespace::Dict{Symbol,Function})
    if !haskey(statespace, :Φ) || !haskey(statespace, :Ψ) || length(statespace) > 2
        @warn "incorrect dictionary passed, must have only Φ and Ψ key-value pairs"
    else
        m.statespace = statespace
    end
    return nothing
end
function update_distributions!(m::PoolModel, distributions::Dict{Symbol,Function})
    if !haskey(distributions, :Φ) || !haskey(distributions, :Ψ) || length(distributions) > 2
        @warn "incorrect dictionary passed, must have only F_ϵ and F_u key-value pairs"
    else
        m.distributions = distributions
    end
    return nothing
end
function update_Φ!(m::PoolModel, f::Function)
    m.statespace[:Φ] = f
    return nothing
end
function update_Ψ!(m::PoolModel, f::Function)
    m.statespace[:Ψ] = f
    return nothing
end
function update_F_ϵ!(m::PoolModel, d::Distribution)
    m.distributions[:F_ϵ] = d
    return nothing
end
function update_F_u!(m::PoolModel, d::Distribution)
    m.distributions[:F_u] = d
    return nothing
end
function update_F_λ!(m::PoolModel, d::Distribution)
    m.distributions[:F_λ] = d
    return nothing
end

# function append_models!(m::PoolModel, models::AbstractModel{T}...)
#     append_models!(m, models)
# end
# function append_models!(m::PoolModel, models::Vector{AbstractModel{T}})
#     for model in models
#         if !haskey(Symbol(typeof(model)))
#             m.model[Symbol(typeof(model))] = model
#         else
#             @warn "model named " String(typeof(model)) " is already added, use update! or pop!"
#         end
#     end
#     return nothing
# end
# function append_models!(m::PoolModel, models::Vector{AbstractModel{T}},
#                         datas::Dict{Symbol,Matrix{T}}) where T<:AbstractModel{T}
#     append_models!(m, models)
#     for kv in datas
#         m.datas[kv[1]] = kv[2]
#     end
#     return nothing
# end
# function append_particles!()
# end
# function append_cond_pred_dens!()
# end
# function append_statespace!()
# end
# function append_distributions!()
# end

# function pop_models!()
# end
# function pop_datas!()
# end
# function pop_particles!()
# end
# function pop_cond_pred_dens!()
# end
# function pop_statespace!()
# end
# function pop_distributions!()
# end

"""
```
Other auxiliary functions that are useful for estimating pooled models.
```
"""
function draw_prior(m::PoolModel)
    return rand(Normal(m[:μ], m[:σ]), 1)
end

function Base.show(io::IO, m::PoolModel)
    @printf io "Dynamic Prediction Pools\n"
    @printf io "no. models:             %i\n" length(get_models(m))
    @printf io "data vintage:           %s\n" data_vintage(m)
    @printf io "description:\n %s\n"          description(m)
end
