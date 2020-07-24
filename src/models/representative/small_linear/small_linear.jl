"""
```
SmallLinear{T} <: AbstractRepModel{T}
```

The `SmallLinear` type defines the structure of the simple New Keynesian DSGE
model described in 'Bayesian Estimation of DSGE Models' by Sungbae An and Frank
Schorfheide.

### Fields

#### Parameters and Steady-States
* `parameters::Vector{AbstractParameter}`: Vector of all time-invariant model
  parameters.

* `steady_state::Vector{AbstractParameter}`: Model steady-state values, computed
  as a function of elements of `parameters`.

* `keys::OrderedDict{Symbol,Int}`: Maps human-readable names for all model
  parameters and steady-states to their indices in `parameters` and
  `steady_state`.

#### Inputs to Measurement and Equilibrium Condition Equations

The following fields are dictionaries that map human-readable names to row and
column indices in the matrix representations of of the measurement equation and
equilibrium conditions.

* `endogenous_states::OrderedDict{Symbol,Int}`: Maps each state to a column in
  the measurement and equilibrium condition matrices.

* `exogenous_shocks::OrderedDict{Symbol,Int}`: Maps each shock to a column in
  the measurement and equilibrium condition matrices.

* `expected_shocks::OrderedDict{Symbol,Int}`: Maps each expected shock to a
  column in the measurement and equilibrium condition matrices.

* `equilibrium_conditions::OrderedDict{Symbol,Int}`: Maps each equlibrium
  condition to a row in the model's equilibrium condition matrices.

* `endogenous_states_augmented::OrderedDict{Symbol,Int}`: Maps lagged states to
  their columns in the measurement and equilibrium condition equations. These
  are added after `gensys` solves the model.

* `observables::OrderedDict{Symbol,Int}`: Maps each observable to a row in the
  model's measurement equation matrices.

* `pseudo_observables::OrderedDict{Symbol,Int}`: Maps each pseudo-observable to
  a row in the model's pseudo-measurement equation matrices.

#### Model Specifications and Settings

* `spec::String`: The model specification identifier, \"small_linear\", cached
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
mutable struct SmallLinear{T} <: AbstractRepModel{T}
    parameters::ParameterVector{T}                         # vector of all time-invariant model parameters
    steady_state::ParameterVector{T}                       # model steady-state values
    keys::OrderedDict{Symbol,Int}                          # human-readable names for all the model
                                                           # parameters and steady-states

    endogenous_states::OrderedDict{Symbol,Int}             # these fields used to create matrices in the
    exogenous_shocks::OrderedDict{Symbol,Int}              # measurement and equilibrium condition equations.
    expected_shocks::OrderedDict{Symbol,Int}               #
    equilibrium_conditions::OrderedDict{Symbol,Int}        #
    endogenous_states_augmented::OrderedDict{Symbol,Int}   #
    observables::OrderedDict{Symbol,Int}                   #
    pseudo_observables::OrderedDict{Symbol,Int}            #

    spec::String                                           # Model specification number (eg "m990")
    subspec::String                                        # Model subspecification (eg "ss0")
    settings::Dict{Symbol,Setting}                         # Settings/flags for computation
    test_settings::Dict{Symbol,Setting}                    # Settings/flags for testing mode
    rng::MersenneTwister                                   # Random number generator
    testing::Bool                                          # Whether we are in testing mode or not

    observable_mappings::OrderedDict{Symbol, Observable}
    pseudo_observable_mappings::OrderedDict{Symbol, PseudoObservable}
end

description(m::SmallLinear) = "Julia implementation of linear model using AnSchorfheide model as template: SmallLinear, $(m.subspec)"

"""
`init_model_indices!(m::SmallLinear)`

Arguments:
`m:: SmallLinear`: a model object

Description:
Initializes indices for all of `m`'s states, shocks, and equilibrium conditions.
"""
function init_model_indices!(m::SmallLinear)
    # Endogenous states
    endogenous_states = collect([
        :c_t, :r_n_t, :Eπ_ct1, :Ec_t1, :w_t, :l_t, :c_dt, :𝜏_t, :m_ct, :y_t, :mc_t, :π_t, :Eπ_t1, :π_ct, :e_rt, :z_t, :y_t1, :uip_t, :c_t_f, :r_n_t_f, :Eπ_ct1_f, :Ec_t1_f, :w_t_f, :l_t_f, :c_dt_f, :m_ct_f, :y_t_f, :mc_t_f, :π_t_f, :Eπ_t1_f, :π_ct_f, :e_f_rt, :z_t_f, :y_t1_f])

    # Exogenous shocks
    exogenous_shocks = collect([
	:r_sh, :r_f_sh, :z_sh, :z_f_sh, :uip_sh])

    # Expectations shocks
    expected_shocks = collect([:Eπ_ct_sh, :Ec_sh, :Eπ_t_sh, :Eπ_ct_f_sh, :Ec_f_sh, :Eπ_t_f_sh])

    # Equilibrium conditions
    equilibrium_conditions = collect([
        :eq_c_t, :eq_w_t, :eq_c_dt, :eq_m_ct, :eq_y_t_1, :eq_mc_t, :eq_π_t, :eq_π_ct, :eq_y_t_2, :eq_r_n_t, :eq_e_rt, :eq_z_t, :eq_Eπ_ct, :eq_Ec, :eq_Eπ_t, :eq_y_t1, :eq_c_t_s, :eq_uip_t, :eq_c_t_f, :eq_w_t_f, :eq_c_dt_f, :eq_m_ct_f, :eq_y_t_1_f, :eq_mc_t_f, :eq_π_t_f, :eq_π_ct_f, :eq_y_t_2_f, :eq_r_n_t_f, :eq_e_f_rt, :eq_z_t_f, :eq_Eπ_ct_f, :eq_Ec_f, :eq_Eπ_t_f, :eq_y_t1_f])

    # Additional states added after solving model
    # Lagged states and observables measurement error
    endogenous_states_augmented = []

    # Observables
    observables = keys(m.observable_mappings)

    # Pseudo-observables
    pseudo_observables = keys(m.pseudo_observable_mappings)

    for (i,k) in enumerate(endogenous_states);           m.endogenous_states[k]           = i end
    for (i,k) in enumerate(exogenous_shocks);            m.exogenous_shocks[k]            = i end
    for (i,k) in enumerate(expected_shocks);             m.expected_shocks[k]             = i end
    for (i,k) in enumerate(equilibrium_conditions);      m.equilibrium_conditions[k]      = i end
    for (i,k) in enumerate(endogenous_states);           m.endogenous_states[k]           = i end
    for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i+length(endogenous_states) end
    for (i,k) in enumerate(observables);                 m.observables[k]                 = i end
    for (i,k) in enumerate(pseudo_observables);          m.pseudo_observables[k]          = i end
end


function SmallLinear(subspec::String="ss0";
                       custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
                       testing = false)

    # Model-specific specifications
    spec               = split(basename(@__FILE__),'.')[1]
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = SmallLinear{Float64}(
            # model parameters and steady state values
            Vector{AbstractParameter{Float64}}(), Vector{Float64}(), OrderedDict{Symbol,Int}(),

            # model indices
            OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(),

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
    DSGE.default_test_settings!(m)
    for custom_setting in values(custom_settings)
        m <= custom_setting
    end

    # Set observable and pseudo-observable transformations
    init_observable_mappings!(m)
    # init_pseudo_observable_mappings!(m)

    # Initialize parameters
    init_parameters!(m)

    init_model_indices!(m)
    init_subspec!(m)
    steadystate!(m)

    return m
end

"""
```
init_parameters!(m::SmallLinear)
`

Initializes the model's parameters, as well as empty values for the steady-state
parameters (in preparation for `steadystate!(m)` being called to initialize
those).
"""
function init_parameters!(m::SmallLinear)
    # Initialize parameters
    m <= parameter(:γ_Q, 1.5, (1e-20, 1e5), (1e-20, 1e5), ModelConstructors.Exponential(), Normal(0.40, 0.20), fixed=false, description="γ_Q: Domestic steady state growth rate of technology.", tex_label="\\gamma_Q")

    m <= parameter(:γ_Q_f, 1.5, (1e-20, 1e5), (1e-20, 1e5), ModelConstructors.Exponential(), Normal(0.40, 0.20), fixed=false, description="γ_Q_f: Foreign steady state growth rate of technology.", tex_label="\\gamma_Q^f")

    m <= parameter(:π_star, 8.1508, (1e-20, 1e5), (1e-20, 1e5), ModelConstructors.Exponential(), GammaAlt(7., 2.), fixed=false, description="π_star: Domestic target inflation rate.", tex_label="\\pi*")

    m <= parameter(:π_star_f, 8.1508, (1e-20, 1e5), (1e-20, 1e5), ModelConstructors.Exponential(), GammaAlt(7., 2.), fixed=false, description="π_star_f: Foreign target inflation rate.", tex_label="\\pi*^f")

    m <= parameter(:rA, 1.0025, (1e-20, 1e5), (1e-20, 1e5), ModelConstructors.Exponential(), GammaAlt(0.5, 0.5), fixed=false, description="rA: β (discount factor) = 1/(1+ rA/400).", tex_label="rA")

    m <= parameter(:σ, 1.0, fixed=true,
                   description="σ: Inverse elasticity of substitution.",
                   tex_label="\\sigma")

    m <= parameter(:𝛘, 3.79, fixed=true,
                   description="𝛘: Inverse labor supply elasticity.",
                   tex_label="\\chi")

    m <= parameter(:η, 1.5, fixed=true,
                   description="η: Trade price elasticity.",
                   tex_label="\\eta")

    m <= parameter(:ω, 0.2, fixed=true,
                   description="ω: Trade openness.",
                   tex_label="\\omega")

    m <= parameter(:α, 0.33, fixed=true,
                   description="α: Output elasticity of capital.",
                   tex_label="\\alpha")

    m <= parameter(:β, 0.9975, fixed=true,
                   description="β: Home consumer's discount rate.",
                   tex_label="\\beta")

    m <= parameter(:ξ_p, 0.84, fixed=true,
                   description="ξ_p: Price stickiness.",
                   tex_label="\\xi_p")

    m <= parameter(:γ_r, 0.82, fixed=true,
                   description="γ_r: Taylor rule inertia coefficient.",
                   tex_label="\\gamma_r")

    m <= parameter(:γ_π, 1.5, fixed=true,
                   description="γ_π: Response in Taylor rule to inflation.",
                   tex_label="\\gamma_\\pi")

    m <= parameter(:θ_p, 0.2, fixed=true,
                   description="θ_p: Net price markup.",
                   tex_label="\\theta_p")

    # Exogenous process
    m <= parameter(:ρ_m, 0.14, fixed=true,
                   description="ρ_m: Monetary shock persistence AR(1) coefficient.",
                   tex_label="\\rho_m")

    m <= parameter(:σ_m, 0.01, fixed=true,
                   description="σ_m: Monetary shock standard deviation.",
                   tex_label="\\sigma_m")

    m <= parameter(:ρ_z, 0.8, fixed=true,
                   description="ρ_z: Technology shock persistence AR(1) coefficient.",
                   tex_label="\\rho_z")

    m <= parameter(:σ_z, 0.01, fixed=true,
                   description="σ_z: Technology shock standard deviation.",
                   tex_label="\\sigma_z")

    m <= parameter(:ρ_uip, 0.66, fixed=true,
                   description="ρ_uip: UIP shock persistence AR(1) coefficient.",
                   tex_label="\\rho_uip")

    m <= parameter(:σ_uip, 0.01, fixed=true,
                   description="σ_uip: UIP shock standard deviation.",
                   tex_label="\\sigma_uip")

    m <= parameter(:e_y, 0.20*0.579923, fixed=true,
                   description="e_y: Measurement error on domestic GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_y_f, 0.20*0.579923, fixed=true,
                   description="e_y_f: Measurement error on foreign GDP growth.",
                   tex_label="e_y^f")

    m <= parameter(:e_π, 0.20*1.470832, fixed=true,
                   description="e_π: Measurement error on domestic inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_π_f, 0.20*1.470832, fixed=true,
                   description="e_π_f: Measurement error on foreign inflation.",
                   tex_label="e_\\pi^f")

    m <= parameter(:e_r_n, 0.20*2.237937, fixed=true,
                   description="e_r_n: Measurement error on the domestic interest rate.",
                   tex_label="e_r^n")

    m <= parameter(:e_r_n_f, 0.20*2.237937, fixed=true,
                   description="e_r_n_f: Measurement error on the foreign interest rate.",
                   tex_label="e_r^{n,f}")

    # Steady states
    m <= SteadyStateParameter(:l_ss, NaN, description="Home steady state labor supply", tex_label="l_ss")
    m <= SteadyStateParameter(:m_c, NaN, tex_label="m_c")
    m <= SteadyStateParameter(:c_d, NaN, tex_label="c_d")
    m <= SteadyStateParameter(:mc, NaN, tex_label="mc")
    m <= SteadyStateParameter(:w, NaN, tex_label="w")
    m <= SteadyStateParameter(:r_n, NaN, tex_label="r^n")
    m <= SteadyStateParameter(:l_ss_f, NaN, description="Foreign steady state labor supply", tex_label="l_ss^*")
    m <= SteadyStateParameter(:m_c_f, NaN, tex_label="m_c^*")
    m <= SteadyStateParameter(:c_d_f, NaN, tex_label="c_d^*")
    m <= SteadyStateParameter(:mc_f, NaN, tex_label="mc^*")
    m <= SteadyStateParameter(:w_f, NaN, tex_label="w^*")
    m <= SteadyStateParameter(:r_n_f, NaN, tex_label="r^{n*}")


end

"""
```
steadystate!(m::SmallLinear)
```

Calculates the model's steady-state values. `steadystate!(m)` must be called whenever
the parameters of `m` are updated.
"""
function steadystate!(m::SmallLinear)

    m[:l_ss] = 0
    m[:m_c] = log(m[:ω])
    m[:c_d] = log(1-m[:ω])
    m[:mc] = -log(1+m[:θ_p])
    m[:w] = m[:mc]
    m[:r_n] = -log(m[:β])
    m[:l_ss_f] = 0
    m[:m_c_f] = log(m[:ω])
    m[:c_d_f] = log(1-m[:ω])
    m[:mc_f] = -log(1+m[:θ_p])
    m[:w_f] = m[:mc_f]
    m[:r_n_f] = -log(m[:β])

    return m
end

function model_settings!(m::SmallLinear)
    DSGE.default_settings!(m)

    # Data
    m <= Setting(:population_mnemonic_f, Nullable(:LFEMTTTTEZQ647S__FRED), "Mnemonic of FRED data series for computing foreign per-capita values (a Nullable{Symbol})")
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
    m <= Setting(:forecast_zlb_value, 0.13,
        "Value of the zero lower bound in forecast periods, if we choose to enforce it")
end

function shock_groupings(m::SmallLinear)
    #=
    gov = ShockGroup("g", [:g_sh], RGB(0.70, 0.13, 0.13)) # firebrick
    tfp = ShockGroup("z", [:z_sh], RGB(1.0, 0.55, 0.0)) # darkorange
    pol = ShockGroup("pol", vcat([:rm_sh], [Symbol("rm_shl$i") for i = 1:n_anticipated_shocks(m)]),
                     RGB(1.0, 0.84, 0.0)) # gold
    det = ShockGroup("dt", [:dettrend], :gray40)

    return [gov, tfp, pol, det]
    =#

    domestic = ShockGroup("domestic", [:r_sh, :z_sh], RGB(0.70, 0.13, 0.13)) # firebrick
    foreign = ShockGroup("foriegn", [:r_f_sh, :z_f_sh], RGB(1.0, 0.55, 0.0)) # darkorange
    other = ShockGroup("other", [:uip_sh], RGB(1.0, 0.84, 0.0)) # gold
    return [domestic, foreign, other]

end
