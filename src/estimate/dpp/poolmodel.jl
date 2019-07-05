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
    models::OrderedDict{Symbol,AbstractModel}              # Model name mapped to model object
    datas::OrderedDict{Symbol,Matrix{T}}                   # Model name " "
    particles::OrderedDict{Symbol,ParticleCloud}           # Model name " " to ParticleCloud
    cond_loglhs::OrderedDict{Symbol,Vector{T}}             # Model name " " to conditional loglh
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

description(m::PoolModel) = "Julia implementation of model defined in 'Bayesian Estimation of DSGE Models' by Sungbae An and Frank Schorfheide: PoolModel, $(m.subspec)"

"""
`init_model_indices!(m::PoolModel)`

Arguments:
`m:: PoolModel`: a model object

Description:
Initializes indices for all of `m`'s states, shocks, and equilibrium conditions.
"""

# function PoolModel(data::Matrix{T}, h::Int, subspec::String="ss0", models::AbstractModel...;
#                    custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
#                    testing::Bool = false, verbose::Bool = :low) where T<:AbstractFloat
#     return PoolModel(subspec, [data for i = 1:length(models)], h, [model for model in models];
#                    custom_settings = custom_settings, testing = testing, verbose = verbose)
# end
# function PoolModel(datas::Vector{Matrix{T}}, h::Int, subspec::String="ss0",
#                    models::AbstractModel...;
#                    custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
#                    testing::Bool = false, verbose::Bool = :low) where T<:AbstractFloat
#     return PoolModel(subspec, datas, h, [model for model in models];
#                    custom_settings = custom_settings, testing = testing, verbose = verbose)
# end
# function PoolModel(datas::Dict{Symbol,Matrix{T}}, h::Int, subspec::String="ss0",
#                    models::AbstractModel...;
#                    custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
#                    testing::Bool = false, verbose::Bool = :low) where T<:AbstractFloat
#     return PoolModel(subspec, datas, h, [model for model in models];
#                    custom_settings = custom_settings, testing = testing, verbose = verbose)
# end
function PoolModel(data::Matrix{T}, h::Int,
                   models::Vector{AbstractModel}, subspec::String="ss0",
                   custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
                   testing::Bool = false, verbose::Bool = :low,
                   static::Bool = false) where T<:AbstractFloat
    return PoolModel(subspec, [data for i = 1:length(models)], h, models;
                   custom_settings = custom_settings,
                   testing = testing, verbose = verbose, static = static)
end
function PoolModel(datas::Dict{Symbol,Matrix{T}}, h::Int,
                   models::Vector{AbstractModel}, subspec::String="ss0",;
                   custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
                   testing::Bool = false, verbose::Bool = :low,
                   static::Bool = false) where T<:AbstractFloat
    if length(data) > length(models)
        error("number of data series exceeds number of models")
    else
        error("number of models exceeds number of data series")
    end
    name_vec = [Symbol(typeof(model)) for model in models]
    try
        data_vec = [datas[name] for name in name_vec]
    catch
        error("at least one key in data dictionary does not match any model")
    end
    return PoolModel(subspec, data_vec, h, models; custom_settings = custom_settings,
                   testing = testing, verbose = verbose, static = static)
end
function PoolModel(datas::Vector{Matrix{T}}, h::Int,
                   models::Vector{AbstractModel}(), subspec::String="ss0";
                   custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
                   testing::Bool = false, verbose::Bool = :low, static::Bool = false)

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

        # particles and nonlinear hidden state model info
        OrderedDict{Symbol,ParticleCloud}(), Dict{Symbol,Function}(),
        Dict{Symbol,Distribution}(),

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

    # Initialize state space equations
    init_statespace!(m)

    # Initialize distributions for state space
    init_distributions!(m)

    # Initialize model indices and subspec
    # init_model_indices!(m)
    init_subspec!(m)
    steadystate!(m)

    # Initialize models dictionary
    init_models!(m, models)

    # Initialize datas dictionary
    init_datas!(m, datas)

    # Initialize particle clouds
    init_particles!(m)

    # Initialize conditional predictive densities
    init_cond_loglhs!(m, h; verbose = verbose)

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
        m <= parameter(:ρ, 1, fixed = true,
                       description="ρ: persistence of AR processing underlying λ.",
                       text_label="\\rho")
        m <= parameter(:μ, 0, fixed = true,
                       description="μ: drift of AR processing underlying λ.",
                       text_label="\\rho")
        m <= parameter(:σ, 1, fixed = true,
                       description="σ: volatility of AR processing underlying λ.",
                       text_label="\\rho")
    else
        m <= parameter(:ρ, 0.5, (0,1), (0,1), Untransformed(), Uniform(0,1), fixed = false,
                       description="ρ: persistence of AR processing underlying λ.",
                       text_label="\\rho")
        m <= parameter(:μ, 0, fixed = true,
                       description="μ: drift of AR processing underlying λ.",
                       text_label="\\rho")
        m <= parameter(:σ, 1, fixed = true,
                       description="σ: volatility of AR processing underlying λ.",
                       text_label="\\rho")
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
end


function init_model_indices!(m::PoolModel)
    # Observables
    observables = keys(m.observable_mappings)

    # Pseudo-observables
    pseudo_observables = keys(m.pseudo_observable_mappings)

    # Collect into model indices
    for (i,k) in enumerate(observables);                 m.observables[k]                 = i end
    for (i,k) in enumerate(pseudo_observables);          m.pseudo_observables[k]          = i end
end

"""
```
init_statespace!(m::PoolModel)
```

Creates transition and measurement equations as passable functions.
"""
function init_statespace!(m::PoolModel)
    # transition equation
    m.statespace[:Φ] = (x,ϵ) -> cdf((1 - m[:ρ]) * m[:μ] + m[:ρ] * quantile(Normal(),x[1]) +
                              sqrt(1 - m[:ρ]^2) * m[:σ] * ϵ)

    # measurement equation
    T = [v for v in values(m.cond_loglhs)] case we have asymmetric lengths of estimation
    loglh_mat = zeros(T,length(m.cond_loglhs)) # matrix of conditional log likelihoods
    for (i,v) in enumerate(values(m.cond_loglhs)) # time period vs. model
        if tmp[i] > T
            loglh_mat[:,i] = v[1:T]
        else
            loglh_mat[:,i] = v
        end
    end
    loglh_mat = loglh_mat'

    m.statespace[:Ψ] = (x,t) -> dot(loglh_mat[:,t], x)

    return m
end

function init_distribution!(m::PoolModel)
    # exogenous shock
    m.distributions[:F_ϵ] = Normal(0,1)

    # measurement error
    m.distributions[:F_u] = DiscreteUniform(0,0)

    return m
end

function init_models!(m::PoolModel, models::Vector{AbstractModel} = Vector{AbstractModel}())
    m.models = models
    return m
end

function init_datas!(m::PoolModel, datas::Vector{Matrix{T}}) where T<:AbstractFloat
    for (name,data) in zip(keys(m.models),datas)
        m.datas[name] = data
    end
    return m
end

function init_particles!(m::PoolModel; names::Vector{Symbol} = Vector{Symbol}())
    if isempty(names)
        for kv in m.models
            load(rawpath(kv[2], "estimate", "smc_cloud.jld2"))
            m.particles[kv[1]] = cloud # need to check this is the correct name
        end
    else
        for name in names
            load(rawpath(m.models[name], "estimate", "smc_cloud.jld2"))
            m.particles[name] = cloud
        end
    end

    return m
end

function init_cond_loglhs!(m::PoolModel, h::Int; names::Vector{Symbol} = Vector{Symbol}(),
                      verbose::Symbol = :low)
    if isempty(names)
        names = keys(m.models)
    else
    for name in names
        θs = load_draws(m.models[name], :full) # matrix of posterior draws, represents whole posterior
        Nθ = length(θs)
        Nt = size(datas[name],2) - h # since predict h periods ahead
        Ns = size(compute_system(m)[:TTT],1)
        cond_loglhs = zeros(Nt, Nθ) # matrix of period t conditional loglhs (on t-1 information set)
        m.cond_loglhs[name] = @sync @distributed (+) for θi in 1:Nθ
            # Evaluate T, R, Z, D given theta
            update!(m.models[name], θs[θi])
            TTT, RRR, CCC = solve(m.models[name])
            tmp = measurement(m.models[name], TTT, RRR, CCC)
            QQ = tmp[:QQ]
            ZZ = tmp[:ZZ]
            DD = tmp[:DD]
            EE = tmp[:EE]

            # Precompute matrices
            TTT_power = Dict{Int,typeof(TTT)}(1 => TTT)
            for i in 2:h
                TTT_power[i] = TTT_power[i-1] * TTT
            end
            TTTtp_power = Dict(i => TTT_power[i]' for i in 1:h)
            Dtild = kron(ones(h+1),DD)
            Ztild = kron(Matrix(I1.0,h+1,h+1),ZZ)

            # Run Kalman filter
            k   = KalmanFilter(TTT, RRR, CCC, QQQ, ZZ, DD, EE)
            SS_t = zeros(Ns * (h+1)) # initialize s_{t:t+h|t-1} vector
            PP_t = zeros(Ns * (h+1), NS * (h+1)) # initialize P_{t:t+h|t-1} matrix
            s_0 = k.s_t
            P_0 = k.P_t
            try
                semi_names = get_setting(m.models[name], :cond_semin_names)
                do_semi = true
            catch
                do_semi = false
            end

            cond_loglh_θ = zeros(Nt) # vector of conditional loglhs (on t-1 information set) and fixing θ
            for t in 1:Nt
                # Compute unconditional forecast of time t
                DSGE.forecast!(k)

                # Compute semiconditional forecast
                if do_semi
                    obs = get_dict(m.models[name], :obs)
                    inds = [obs[semi_name] for semi_name in semi_names]
                    update!(k, m.datas[name][inds,t])
                end

                # Recursively forecast by j = 1:h periods
                SS_t[1:Ns] = k.s_t
                PP_t[1:NS, 1:Ns] = k.P_t
                for j in 1:h
                    PP_t[1:Ns, 1+Ns*j:Ns*(j+1)] = k.P_t * TTTtp_power[j]
                    PP_t[1+Ns*j:Ns*(j+1)] = TTT_power[j] * k.P_t
                end
                for j in 1:h
                    forecast!(k)
                    indices = 1+Ns*j:Ns*(j+1)
                    SS_t[indices] = k.s_t
                    PP_t[indices, indices] = k.P_t
                    for m in 1:h-j
                        PP_t[indices,1+Ns*(m+j):Ns*(m+j+1)] = k.P_t * TTTtp_power[m]
                        PP_t[1+Ns*(m+j):Ns*(m+j+1),indices] = TTT_power[m] * k.P_t
                    end
                end

                # Compute conditional log likelihood p(y_t|θ, I_{t-1})
                μ_mv = Dtild + Ztild * SS_t
                Σ_mv = Ztild * PP_t * Ztild'
                inv_Σ_mv = inv(Σ_mv)
                det_Σ = det(mv)
                err = vec(datas[name][:,t:t+h]) - μ_mv
                cond_loglh_θ[t] = (2*pi)^(-length(err)/2) * det_Σ^(-1/2) * exp(-(1/2) * dot(err, inv_Σ * err))
            end
            cond_loglhs[:,θi] = cond_loglh_θ

        end
        m.cond_loglhs[name] ./= Nθ
    end

    return m
end
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
        return OrderedDict(name => m.models[name] for name in names)
    end
end
function get_datas(m::PoolModel, names::Symbol)
    return get_datas(m, [names])
end
function get_datas(m::PoolModel, names::Vector{Symbol} = Vector{Symbol}())
    if isempty(names)
        return m.datas
    else
        return OrderedDict(name => m.datas[name] for name in names)
    end
end
function get_particles(m::PoolModel, names::Symbol)
    return get_particles(m, [names])
end
function get_particles(m::PoolModel, names::Vector{Symbol} = Vector{Symbol}())
    if isempty(names)
        return m.particles
    else
        return OrderedDict(name => m.particles[name] for name in names)
    else
end
function get_cond_loglhs(m::PoolModel, names::Symbol)
    return get_cond_loglhs(m, [names])
end
function get_cond_loglhs(m::PoolModel, names::Vector{Symbol} = Vector{Symbol}())
    if isempty(names)
        return m.cond_loglhs
    else
        return OrderedDict(name => m.cond_loglhs[name] for name in names)
    end
end
function get_system(m::PoolModel)
    return Dict(:statespace => m.statespace, :distributions => m.distributions)
end
function get_statespace(m::PoolModel; F::Symbol = :all)
    if F == :all
        return m.statespace
    else
        return m.statespace[F]
    end
end
function get_distributions(m::PoolModel; F::Symbol = :all)
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

function update_models!(m::PoolModel, models::AbstractModel...;
                        populate::Bool = true)
    update_models!(m, models; populate = populate)
end
function update_models!(m::PoolModel, models::Vector{AbstractModel};
                        populate::Bool = true)
    update_models!(m, Dict(Symbol(typeof(model)) => model for model in models);
                   populate = populate)
    return nothing
end
function update_models!(m::PoolModel, models::Dict{Symbol,AbstractModel};
                        populate::Bool = true)
    for kv in models
        if haskey(kv[1])
            m.model[kv[1]] = kv[2]
        else
            @warn "no model named " * String(kv[1]) * " found"
        end
    end
    if populate
        model_keys = Vector(keys(models))
        init_particles!(m; model_keys)
        init_cond_loglhs!(m; model_keys)
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
function update_particles!(m::PoolModel, p::Dict{Symbol,ParticleCloud})
    for kv in p
        try
            m.particles[kv[1]] = kv[2]
        catch
            @warn "no model named " * String(kv[1]) * " found"
        end
    end
    return nothing
end
function update_cond_loglhs!(m::PoolModel,
                             cond_loglhs::Dict{Symbol,Vector{T}}) where T<:AbstractFloat
    for kv in cond_loglhs
        try
            m.cond_loglhs[kv[1]] = kv[2]
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

# function append_models!(m::PoolModel, models::AbstractModel...)
#     append_models!(m, models)
# end
# function append_models!(m::PoolModel, models::Vector{AbstractModel})
#     for model in models
#         if !haskey(Symbol(typeof(model)))
#             m.model[Symbol(typeof(model))] = model
#         else
#             @warn "model named " String(typeof(model)) " is already added, use update! or pop!"
#         end
#     end
#     return nothing
# end
# function append_models!(m::PoolModel, models::Vector{AbstractModel},
#                         datas::Dict{Symbol,Matrix{T}}) where T<:AbstractModel
#     append_models!(m, models)
#     for kv in datas
#         m.datas[kv[1]] = kv[2]
#     end
#     return nothing
# end
# function append_particles!()
# end
# function append_cond_loglhs!()
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
# function pop_cond_loglhs!()
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
