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

* `model::OrderedDict{Symbol,AbstractDSGEModel}`: Maps name to its underlying model
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
mutable struct PoolModel{T} <: AbstractDSGEModel{T}
    parameters::ParameterVector{T}                         # vector of all time-invariant model parameters
    keys::OrderedDict{Symbol,Int}                          # human-readable names for all the model
                                                           # parameters and steady-states
    observables::OrderedDict{Symbol,Int}                   # Model names to observables (predictive densities)

    spec::String                                           # Model specification number (eg "m990")
    subspec::String                                        # Model subspecification (eg "ss0")
    settings::Dict{Symbol,Setting}                         # Settings/flags for computation
    test_settings::Dict{Symbol,Setting}                    # Settings/flags for testing mode
    rng::MersenneTwister                                   # Random number generator
    testing::Bool                                          # Whether we are in testing mode or not

    observable_mappings::OrderedDict{Symbol, Observable}
end

description(m::PoolModel) = "Julia implementation of dynamic prediction pools defined in 'Dynamic prediction pools: An investigation of financial frictions and forecasting performance' by Marco Del Negro, Raiden B. Hasegawa, and Frank Schorfheide: PoolModel, $(m.subspec)"

function init_model_indices!(m::AnSchorfheide)
    # Observables
    observables = keys(m.observable_mappings)
    for (i,k) in enumerate(observables); m.observables[k] = i end
end

function PoolModel(subspec::String="ss0";
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

        # Observable indices
        OrderedDict{Symbol,Int}(),

        # settings
        spec,
        subspec,
        settings,
        test_settings,
        rng,
        testing,
        OrderedDict{Symbol,Observable}())

    # Set settings
    model_settings!(m)
    default_test_settings!(m)
    for custom_setting in values(custom_settings)
        m <= custom_setting
    end

    # Set observable and pseudo-observable transformations
    init_observable_mappings!(m)

    # Initialize parameters
    init_parameters!(m; static = static)

    # Initialize model indices and subspec
    init_model_indices!(m)
    init_subspec!(m)

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
        m <= parameter(:ρ, 0.8, (1e-5,0.999), (1e-5,0.999), SquareRoot(), Uniform(0.,1.), fixed = false,
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

function model_settings!(m::PoolModel)
    default_settings!(m)

    # Data

    # SMC estimation
    m <= Setting(:sampling_method, :SMC)
    m <= Setting(:n_particles, 2000)

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

    # MH estimation
    m <= Setting(:mh_cc, 0.15^2)
    m <= Setting(:mh_cc0, 0.15^2)
    m <= Setting(:calculate_hessian, false)
    m <= Setting(:reoptimize, false)
    m <= Setting(:n_mh_simulations, 1000)
    m <= Setting(:n_mh_blocks, 1)
    m <= Setting(:mh_thin, 1)
    m <= Setting(:n_mh_burn, 0)
end

"""
```
Access, update, and show functions for PoolModel.
```
"""
# function get_forecast_horizon(m::PoolModel{T}) where T<:AbstractFloat
#     return m.forecast_horizon
# end

# function update_forecast_horizon!(m::PoolModel{T}, h::Int) where T<:AbstractFloat
#     m.forecast_horizon = h
# end

# function Base.show(io::IO, m::PoolModel)
#     @printf io "Dynamic Prediction Pools\n"
#     @printf io "no. models:             %i\n" length(get_models(m))
#     @printf io "data vintage:           %s\n" data_vintage(m)
#     @printf io "description:\n %s\n"          description(m)
# end
