"""
```
BayerBornLuetticke{T} <: AbstractHeterogeneousModel{T}
```

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

* `observables::OrderedDict{Symbol,Int}`: Maps each observable to a row in the
  model's measurement equation matrices.

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

"""
mutable struct BayerBornLuetticke{T} <: AbstractHetModel{T}
    parameters::ParameterVector{T}               # vector of all time-invariant model parameters
    steady_state::ParameterVector{T}             # model steady-state values

    # Temporary to get it to work. Need to
    # figure out a more flexible way to define
    # "grids" that are not necessarily quadrature
    # grids within the model
# TODO: add field/type to hold reduction information e.g. DCT indices (but not coefficient values), copula info
    grids::OrderedDict{Symbol,Union{Grid, Array}}
    keys::OrderedDict{Symbol,Int}                    # Human-readable names for all the model
                                              # parameters and steady-states

    state_variables::Vector{Symbol}                  # Vector of symbols of the state variables
    jump_variables::Vector{Symbol}                   # Vector of symbols of the jump variables
    normalized_model_states::Vector{Symbol}          # All of the distributional model
                                                     # state variables that need to be normalized

    # Vector of ranges corresponding to normalized (post Klein solution) indices
    endogenous_states::OrderedDict{Symbol,UnitRange}
    endogenous_states_unreduced::OrderedDict{Symbol,UnitRange} # full dense grid indices

    exogenous_shocks::OrderedDict{Symbol,Int}
    expected_shocks::OrderedDict{Symbol,Int}
    equilibrium_conditions::OrderedDict{Symbol,UnitRange}
    endogenous_states_augmented::OrderedDict{Symbol, Int}
    observables::OrderedDict{Symbol,Int}
    pseudo_observables::OrderedDict{Symbol,Int}

    spec::String                                     # Model specification number (eg "m990")
    subspec::String                                  # Model subspecification (eg "ss0")
    settings::Dict{Symbol,Setting}                   # Settings/flags for computation
    test_settings::Dict{Symbol,Setting}              # Settings/flags for testing mode
    rng::MersenneTwister                             # Random number generator
    testing::Bool                                    # Whether we are in testing mode or not
    observable_mappings::OrderedDict{Symbol, Observable}
    pseudo_observable_mappings::OrderedDict{Symbol, PseudoObservable}
end

description(m::BayerBornLuetticke) = "BayerBornLuetticke, $(m.subspec)"

"""
`init_model_indices!(m::BayerBornLuetticke)`

Arguments:
`m:: BayerBornLuetticke`: a model object

Description:
Initializes indices for all of `m`'s model states, shocks, and equilibrium conditions.
By model states, we mean the states in the model's reduced form state-space representation,
hence these states include both predetermined states and jumps.
"""
function init_model_indices!(m::BayerBornLuetticke)
    # TODO: maybe want to rename some of the indices to make it explicit
    #       exactly what we're perturbing (e.g. we are perturbing DCT coefficients or CDFs)

    # Predetermined states
    # TODO: delete states that should just be augmented states
    m.state_variables = [# Endogenous function-valued states
                         :marginal_m′_t, :marginal_k′_t, :marginal_y′_t, :copula′_t,

                         # Endogenous scalar-valued states (e.g. lags)
                         :union_retained′_t, :retained′_t,
                         :Y′_t1, :B′_t1, :T′_t1, :I′_t1, :w′_t1, :q′_t1, :C′_t1,
                         :avg_tax_rate′_t1, :τ_prog′_t1,

                         # Exogenous scalar-valued states:
                         :A′_t, :Z′_t, :Ψ′_t, :RB′_t, :μ_p′_t, :μ_w′_t, :σ′_t,
                         :G_sh′_t, :P_sh′_t, :R_sh′_t, :S_sh′_t]

    # Jumps
    # TODO: delete jump variables that should just be pseudo-observables
    m.jump_variables = [# Function-valued jumps
                        :Vm′_t, :Vk′_t,

                        # Endogenous scalar-valued jumps
                        :rk′_t, :w′_t, :K′_t, :π′_t, :π_w′_t, :Y′_t, :C′_t, :q′_t, :N′_t, :mc′_t,
                        :mc_w′_t, :u′_t, :Ht′_t, :avg_tax_rate′_t, :T′_t, :I′_t, :B′_t,
                        :BD′_t, :BY′_t, :TY′_t, :mc_w_w′_t, :G′_t, :τ_level′_t, :τ_prog′_t,
                        :Gini_C′_t, :Gini_X′_t, :sd_log_y′_t, :I90_share′_t,
                        :I90_share_net′_t, :W90_share′_t, :Ygrowth′_t,
                        :Bgrowth′_t, :Igrowth′_t, :wgrowth′_t, :Cgrowth′_t,
                        :Tgrowth′_t, :LP′_t, :LP_X_A′_t, :tot_retained_Y′_t,
                        :union_firm_profits′_t, :union_profits′_t, :firm_profits′_t,
                        :profits′_t]

    # Exogenous shocks
    exogenous_shocks = collect([:A_sh, :Z_sh, :Ψ_sh, :μ_p_sh, :μ_w_sh, :G_sh, :R_sh, :S_sh, :P_sh])

    # Observables
    observables = keys(m.observable_mappings)

    # Pseudo-Observables
    pseudo_observables = keys(m.pseudo_observable_mappings)

    # Initialize indices for exogenous shocks, observables, and pseudo-observables
    for (i,k) in enumerate(exogenous_shocks);   m.exogenous_shocks[k] = i end
    for (i,k) in enumerate(observables);        m.observables[k]      = i end
    for (i,k) in enumerate(pseudo_observables); m.pseudo_observables[k]      = i end

    # Initialize indices for
    # reduced-form endogenous_states (from gensys notation) and equilibrium conditions
    setup_indices!(m)

    # Create dict for unreduced endogenous_states
    # to facilitate repeated solutions of steady state
    m.endogenous_states_unreduced = deepcopy(m.endogenous_states)
    m <= Setting(:n_model_states_unreduced, length(m.endogenous_states),
                 "Number of model states (incl. both predetermined states and jumps) before any state-space reduction")

    # Additional states added after solving model, namely
    # lagged states and observables measurement error
    # TODO: add augmented variables back
#=    endogenous_states_augmented = [:C_t1]

    for (i, k) in enumerate(endogenous_states_augmented)
        m.endogenous_states_augmented[k] = i + length(m.endogenous_states[get_setting(m, :jumps)[end]])
    end
    m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states) +
                 length(m.endogenous_states_augmented))=#
end
# TODO: maybe add coarse as a kwarg
function BayerBornLuetticke(subspec::String="ss0";
                            custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
                            testing = false)

    # Model-specific specifications
    spec               = "bayer_born_luetticke"
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = BayerBornLuetticke{Float64}(
            # model parameters and steady state values
            Vector{AbstractParameter{Float64}}(), Vector{Float64}(),
            # grids and keys
            OrderedDict{Symbol,Union{Grid, Array}}(), OrderedDict{Symbol,Int}(),

            # normalized_model_states, state_inds, jump_inds
            Vector{Symbol}(), Vector{Symbol}(), Vector{Symbol}(),

            # model indices
            # endogenous states unnormalized, endogenous states normalized
            OrderedDict{Symbol,UnitRange}(), OrderedDict{Symbol,UnitRange}(),
            OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(),
            OrderedDict{Symbol,UnitRange}(), # OrderedOrderedDict{Symbol,UnitRange}(),
            OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(),
            OrderedDict{Symbol,Int}(),

            spec,
            subspec,
            settings,
            test_settings,
            rng,
            testing,
            OrderedDict{Symbol,Observable}(),
            OrderedDict{Symbol,PseudoObservable}())

    default_settings!(m)

    # Set observable transformations
    init_observable_mappings!(m)

    # Set settings
    model_settings!(m)
    for custom_setting in values(custom_settings)
        m <= custom_setting
    end

    # Initialize model indices
    init_model_indices!(m)

    # Init what will keep track of # of states, jumps, and states/jump indices
    # (need model indices first)
    # init_states_and_jumps!(m, states, jumps)

    # Initialize parameters
    init_parameters!(m)

    # Initialize aggregate steady state parameters (necessary for grid construction) # TODO: may not be necessary for grid
    # aggregate_steadystate!(m)

    # Initialize grids
    # init_grids!(m; coarse = true)

    # Solve for the steady state
    # steadystate!(m)

    # So that the indices of m.endogenous_states reflect the normalization
    # normalize_model_state_indices!(m)

    init_subspec!(m)

    return m
end

"""
```
init_parameters!(m::BayerBornLuetticke)
```

Initializes the model's parameters, as well as empty values for the steady-state
parameters (in preparation for `steadystate!(m)` being called to initialize
those).
"""
function init_parameters!(m::BayerBornLuetticke)
    ######################################
    # Parameters that affect steady-state
    ######################################

    # HH preferences
    m <= parameter(:ξ, 4., fixed = true,
                   description = "ξ: risk aversion of households",
                   tex_label = "\\xi")
    m <= parameter(:γ, 2., fixed = true,
                   description = "γ: inverse Frisch elasticity",
                   tex_label = "\\gamma")
    m <= parameter(:β, 0.9842, fixed = true,
                   description = "β: discount factor",
                   tex_label = "\\beta")
    m <= parameter(:λ, 0.095, fixed = true,
                   description = "λ: portfolio adjustment probability",
                   tex_label = "\\lambda")
    m <= parameter(:γ_scale, 0.2, fixed = true,
                   description = "γ_scale: disutility of labor",
                   tex_label = "\\gamma_{\\text{scale}}")

    # Individual income process
    m <= parameter(:ρ_h, 0.98, fixed = true,
                   description = "ρ_h: autocorrelation of income shock",
                   tex_label = "\\rho_{h}")
    m <= parameter(:σ_h, 0.12, fixed = true,
                   description = "σ_h: standard deviation of income shock",
                   tex_label = "\\sigma_{h}")
    m <= parameter(:ι, 1. / 16, fixed = true,
                   description = "ι: probability of return to worker",
                   tex_label = "\\iota")
    m <= parameter(:ζ, 1. / 3750., fixed = true,
                   description = "ζ: probability of becoming an entrepreneur",
                   tex_label = "\\zeta")

    # Technological parameters
    m <= parameter(:α, 0.318, fixed = true,
                   description = "Capital share", tex_label = "\\alpha")
    m <= parameter(:δ_0, (.07 + .016) / 4., fixed = true,
                   description = "Depreciation rate", tex_label = "\\delta_0")

    # NK Phillips Curve
    m <= parameter(:μ_p, 1.1, fixed = true,
                   description = "Price markup", tex_label = "\\mu_p")
    m <= parameter(:μ_w, 1.1, fixed = true,
                   description = "Wage markup", tex_label = "\\mu_w")
    m <= parameter(:π, 1.0 ^ 0.25, fixed = true, # THIS MIGHT BE SET AS A STEADY STATE PARAMETER RATHER THAN HERE
                   description = "Steady-state inflation", tex_label = "\\pi")

    # Monetary policy
    m <= parameter(:RB, m[:π] * 1.0 ^ 0.25, fixed = true, # THIS MIGHT BE SET AS A STEADY STATE PARAMETER RATHER THAN HERE
                   description = "Steady-state nominal interest rate", tex_label = "\\RB")

    # Remaining steady-state parameters
    m <= parameter(:ψ, 0.1, fixed = true,
                   description = "Steady-state bond to capital ratio", tex_label = "\\psi")
    m <= parameter(:τ_lev, 0.825, fixed = true,
                   description = "Steady-state income tax rate level", tex_label = "\\tau^L")
    m <= parameter(:τ_prog, 0.825, fixed = true,
                   description = "Steady-state income tax rate progressivity", tex_label = "\\tau^P")
    m <= parameter(:R, 1.01, fixed = true, # THIS MIGHT BE SET AS A STEADY STATE PARAMETER RATHER THAN HERE
                   description = "Steady-state return of capital (unused)", tex_label = "R")
    m <= parameter(:K, 40., fixed = true, # THIS MIGHT BE SET AS A STEADY STATE PARAMETER RATHER THAN HERE
                   description = "Steady-state quantity of capital (unused)", tex_label = "K")
    m <= parameter(:Rbar, (m[:π] * 1.0675 ^ 0.25 - 1.), fixed = true, # THIS MIGHT BE SET AS A STEADY STATE PARAMETER RATHER THAN HERE
                   description = "Borrowing wedge in interest rate", tex_label = "\\bar{R}")

    #######################################################
    # Parameters that affect dynamics but not steady-state
    #######################################################
    # Retained earnings
    m <= parameter(:ω_F, 0.1, (0., 1.), (0., 1.), SquareRoot(),
                   Uniform(0., 1.), fixed = false,
                   description = "fraction of retained earnings (profits) that is disbursed to HH",
                   tex_label = "\\omega_F")
    m <= parameter(:ω_U, 0.1, (0., 1.), (0., 1.), SquareRoot(),
                   Uniform(0., 1.), fixed = false,
                   description = "fraction of retained earnings (wages) that is disbursed to HH",
                   tex_label = "\\omega_U")

    # Technological parameters
    m <= parameter(:δ_s, 5., (0., 1e2), (0., 1e2), ModelConstructors.Exponential(),
                   GammaAlt(5., 2.), fixed = false,
                   description = "Depreciation increase from flexible utilization",
                   tex_label = "\\delta_s")
    m <= parameter(:ϕ, 4., (0., 1e2), (0., 1e2), ModelConstructors.Exponential(),
                   GammaAlt(4., 2.), fixed = false,
                   description = "Depreciation increase from flexible utilization",
                   tex_label = "\\phi")

    # NK Phillips Curve
    m <= parameter(:κ_p, 1. / 11., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   GammaAlt(0.1, 0.01), fixed = false,
                   description = "Price adjustment cost (Calvo probability)",
                   tex_label = "\\kappa_p")
    m <= parameter(:κ_w, 1. / 11., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   GammaAlt(0.1, 0.01), fixed = false,
                   description = "Wage adjustment cost (Calvo probability)",
                   tex_label = "\\kappa_w")

    # Monetary policy
    m <= parameter(:ρ_R , 0.75, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(),
                   BetaAlt(0.75, 0.10), fixed = false,
                   description = "ρ: The degree of inertia in the monetary policy rule.",
                   tex_label="\\rho_R")
    m <= parameter(:θ_π, 1.5, (1., 10.), (1e-5, 10.0), ModelConstructors.Exponential(),
                   Normal(1.7, 0.3), fixed = false, # Note second tuple is parameterization for Exponential transform
                   description = "ψ1: Weight on inflation gap in monetary policy rule.",
                   tex_label = "\\theta_{\\pi}")
    m <= parameter(:θ_Y, 0.5, (-5., 5.), (-5., 5.), SquareRoot(),
                   Normal(0.125, 0.05), fixed = false,
                   description = "ψy: Weight on output gap in monetary policy rule",
                   tex_label = "\\theta_y")

    # Fiscal policy
    m <= parameter(:γ_B, 0.2, (0., 5.), (0., 5.), SquareRoot(),
                   GammaAlt(0.1, 0.075), fixed = false,
                   description = "γ_B: Reaction of deficit to debt",
                   tex_label = "\\gamma_B")
    m <= parameter(:γ_π, -0.1, (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0.1, 1.), fixed = false,
                   description = "γ_π: Reaction of deficit to inflation",
                   tex_label = "\\gamma_{\\pi}")
    m <= parameter(:γ_Y, -1., (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0.1, 1.), fixed = false,
                   description = "γ_Y: Reaction of deficit to output",
                   tex_label = "\\gamma_{Y}")
    m <= parameter(:ρ_τ, 0.5, (1e-5, 1. - 1e-5), (1e-5, 1. - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_τ: Persistence in tax level",
                   tex_label = "\\rho_{\\tau}")
    m <= parameter(:γ_B_τ, 0.2, (-10., 10.), (10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_B_τ: Reaction of tax level to debt",
                   tex_label = "\\gamma_{B, \\tau}")
    m <= parameter(:γ_Y_τ, -1., (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_Y_τ: Reaction of tax level to output",
                   tex_label = "\\gamma_{Y, \\tau}")
    m <= parameter(:ρ_P, 0.5, (1e-5, 1. - 1e-5), (1e-5, 1. - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_P: Persistence in tax level",
                   tex_label = "\\rho_{P}")
    m <= parameter(:γ_B_P, 0.2, (-10., 10.), (10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_B_P: Reaction of tax level to debt",
                   tex_label = "\\gamma_{B, P}")
    m <= parameter(:γ_Y_P, -1., (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_Y_P: Reaction of tax level to output",
                   tex_label = "\\gamma_{Y, P}")

    # Exogenous processes - autocorrelation
    m <= parameter(:ρ_A, 0.9, (1e-5, 1. - 1e-5), (1e-5, 1. - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_A: AR(1) coefficient in the bond-spread process.",
                   tex_label = "\\rho_A")
    m <= parameter(:ρ_z, 0.9, (1e-5, 1. - 1e-5), (1e-5, 1. - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_z: AR(1) coefficient in the technology process.",
                   tex_label = "\\rho_z")
    m <= parameter(:ρ_Ψ, 0.9, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_Ψ: AR(1) coefficient in marginal efficiency of investment (MEI) process.",
                   tex_label = "\\rho_{\\Psi}")
    m <= parameter(:ρ_μ_p, 0.9, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_μ_p: AR(1) coefficient in the price mark-up shock process.",
                   tex_label = "\\rho_{\\mu_p}")
    m <= parameter(:ρ_μ_w, 0.9, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_μ_w: AR(1) coefficient in the wage mark-up shock process.",
                   tex_label = "\\rho_{\\mu_w}")
    m <= parameter(:ρ_S, 0.84, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.7, 0.2), fixed = false,
                   description = "ρ_S: AR(1) coefficient in the idiosyncratic income risk process.",
                   tex_label = "\\rho_S")
    m <= parameter(:ρ_G, 0.98, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_G: AR(1) coefficient in the government structural deficit process.",
                   tex_label = "\\rho_G")

    # Auxiliary exogenous processes - autocorrelations (fixed to a very small number in baseline specification)
    m <= parameter(:ρ_R_ϵ, 1e-8, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_R_ϵ: AR(1) coefficient in the monetary policy shock process.",
                   tex_label = "\\rho_{R, \\epsilon}")
    m <= parameter(:ρ_P_ϵ, 1e-8, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_P_ϵ: AR(1) coefficient in the tax progressivity shock process.",
                   tex_label = "\\rho_{P, \\epsilon}")
    m <= parameter(:ρ_S_ϵ, 1e-8, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_S_ϵ: AR(1) coefficient in shock process of the shock in the idiosyncatic income shock process.",
                   tex_label = "\\rho_{S, \\epsilon}")

    # Exogenous processes - standard deviations
    m <= parameter(:σ_A, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false, # Note second tuple is parameterization for Exponential transform
                   description = "σ_A: standard dev. of the bond-spread process.",
                   tex_label = "\\sigma_{A}")
    m <= parameter(:σ_z, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false,
                   description = "σ_z: standard dev. of the process describing the " *
                   "stationary component of productivity.", tex_label = "\\sigma_z")
    m <= parameter(:σ_Ψ, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false,
                   description = "σ_Ψ: standard dev. of the exogenous marginal efficiency" *
                   " of investment shock process.", tex_label = "\\sigma_{\\Psi}")
    m <= parameter(:σ_μ_p, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false,
                   description = "σ_μ_p: standard dev. of the price mark-up shock process",
                   tex_label = "\\sigma_{\\mu_p}")
    m <= parameter(:σ_μ_w, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false,
                   description = "σ_μ_w: : standard dev. of the wage mark-up shock process",
                   tex_label = "\\sigma_{\\mu_w}")
    m <= parameter(:σ_S, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   GammaAlt(0.65, 0.3), fixed = false,
                   description = "σ_S: standard dev. of the idiosyncratic income risk shock process",
                   tex_label = "\\sigma_{S}")
    m <= parameter(:Σ_n, 0., (-1e3, 1e3), (-1e3, 1e3), ModelConstructors.SquareRoot(),
                   Normal(0., 100.), fixed = false,
                   description = "Σ_n: reaction of income risk to employment status",
                   tex_label = "\\Sigma_{n}")
    m <= parameter(:σ_R, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false,
                   description = "σ_R_ϵ: standard dev. of the monetary policy shock process",
                   tex_label = "\\sigma_{R, \\epsilon}")
    m <= parameter(:σ_G, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   GammaAlt(0.65, 0.3), fixed = false,
                   description = "σ_G: standard dev. of the structural deficit shock process",
                   tex_label = "\\sigma_{G}")
    m <= parameter(:σ_P, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2., 0.10), fixed = false,
                   description = "σ_P: standard dev. of the tax progressivity shock process",
                   tex_label = "\\sigma_{P}")

    # Measurement error
    m <= parameter(:σ_W90_share, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2., 0.10), fixed = false,
                   description = "σ_W90_share: standard dev. of measurement error for 90th percentile of wealth distribution",
                   tex_label = "\\sigma_{W^{(90)}}")
    m <= parameter(:σ_I90_share, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2., 0.10), fixed = false,
                   description = "σ_I90_share: standard dev. of measurement error for 90th percentile of income distribution",
                   tex_label = "\\sigma_{I^{(90)}}")
    m <= parameter(:σ_P_me, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2., 0.10), fixed = false,
                   description = "σ_P_me: standard dev. of measurement error for tax progressivity",
                   tex_label = "\\sigma_{P, me}")
    m <= parameter(:σ_S_me, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2., 0.10), fixed = false,
                   description = "σ_S_me: standard dev. of measurement error for idiosyncratic income risk",
                   tex_label = "\\sigma_{S, me}")

    # Steady-state parameters

    # Parameters pertaining to idiosyncratic state space
    m <= SteadyStateParameter(:H, NaN,  description = "Long-run average human capital " *
                              "in stationary equilibrium", tex_label = "H")
    m <= SteadyStateParameter(:HW, NaN, description =
                              "Long-run fraction of workers in stationary equilibrium",
                              tex_label = "HW") # there are workers and entrepreneurs in equilibrium

    # Aggregate scalars (just initialized here, these will be populated by the steadystate!)
    m <= SteadyStateParameter(:K_star, NaN, description = "Capital stock (steady-state)",
                              tex_label = "\\K_*")
    m <= SteadyStateParameter(:N_star, NaN, description = "Labor supply (steady-state)",
                              tex_label = "\\N_*")
    m <= SteadyStateParameter(:Y_star, NaN, description = "Output (steady-state)",
                              tex_label = "\\Y_*")
    m <= SteadyStateParameter(:G_star, NaN, description = "Government spending (steady-state)",
                              tex_label = "\\G_*")
    m <= SteadyStateParameter(:w_star, NaN, description = "Wage (steady-state)",
                              tex_label = "\\w_*")
    m <= SteadyStateParameter(:T_star, NaN, description = "Tax revenue (steady-state)",
                              tex_label = "\\T_*")
    m <= SteadyStateParameter(:I_star, NaN, description = "Investment (steady-state)",
                              tex_label = "\\I_*")
    m <= SteadyStateParameter(:B_star, NaN, description = "Bond supply (steady-state)",
                              tex_label = "\\B_*")

    # Steady state grids for functional/distributional variables pre-reduction
    total_idio_states = get_setting(m, :nm) * get_setting(m, :nk) * get_setting(m, :ny)
    m <= SteadyStateParameterGrid(:distr_star, fill(1. / total_idio_states, # populate with a uniform guess
                                                    get_setting(m, :nm), get_setting(m, :nk), get_setting(m, :ny)),
                                  description = "Distribution over idiosyncratic states (steady-state)", tex_label = "D_*")
    m <= SteadyStateParameterGrid(:marginal_pdf_m_star, fill(NaN, get_setting(m, :nm)),
                                  description = "Steady-state expected discounted
                                  marginal_pdf utility of consumption", tex_label = "l_*")
    m <= SteadyStateParameterGrid(:marginal_pdf_k_star, fill(NaN, get_setting(m, :nm)),
                                  description = "Steady-state expected discounted
                                  marginal_pdf utility of consumption", tex_label = "l_*")
    m <= SteadyStateParameterGrid(:marginal_pdf_y_star, fill(NaN, get_setting(m, :nm)),
                                  description = "Steady-state expected discounted
                                  marginal utility of consumption", tex_label = "l_*")
    m <= SteadyStateParameterGrid(:Vm_star, fill(NaN, get_setting(m, :nm), get_setting(m, :nk), get_setting(m, :ny)),
                                  description = "Marginal value of liquid bonds (steady-state)", tex_label = "V_{m, *}")
    m <= SteadyStateParameterGrid(:Vk_star, fill(NaN, get_setting(m, :nm), get_setting(m, :nk), get_setting(m, :ny)),
                                  description = "Marginal value of illiquid capital (steady-state)", tex_label = "V_{k, *}")

    # Steady state grids for functional/distributional variables post-reduction
    m <= SteadyStateParameterGrid(:marginal_cdf_m_star, fill(NaN, get_setting(m, :nm)),
                                  description = "Steady-state expected discounted
                                  marginal_cdf utility of consumption", tex_label = "l_*")
    m <= SteadyStateParameterGrid(:marginal_cdf_k_star, fill(NaN, get_setting(m, :nm)),
                                  description = "Steady-state expected discounted
                                  marginal_cdf utility of consumption", tex_label = "l_*")
    m <= SteadyStateParameterGrid(:marginal_cdf_y_star, fill(NaN, get_setting(m, :nm)),
                                  description = "Steady-state expected discounted
                                  marginal utility of consumption", tex_label = "l_*")
    m <= SteadyStateParameterGrid(:copula_star, fill(NaN, get_setting(m, :nm)),
                                  description = "Steady-state expected discounted
                                  marginal utility of consumption", tex_label = "l_*")
end

"""
```
aggregate_steadystate!(m::BayerBornLuetticke)
```
computes the steady state of aggregate scalar variables.

This function breaks out the "analytic" steady-state solution here instead of in steadystate!
since some of these parameters are required to construct the grids that are used in the
functional/distributional steady state variable calculation
(where init_grids! is called before steadystate! in model construction).

# TODO: this may actually not be necessary to call before solving steady state
"""
function aggregate_steadystate!(m::BayerBornLuetticke{T}) where {T <: Real}

    # Create steady state values information
    ss = OrderedDict{Symbol, Tuple{T, String, String}}() # use ordered dict to make sure values are entered sequentially
    ss[:A] = (1., "Bond spread (steady-state)", "")
    ss[:Z] = (1., "TFP (steady-state)", "")
    ss[:Ψ] = (1., "MEI (steady-state)", "")
    ss[:μ_p] = (get_untransformed_values(m[:μ_p]), "Price mark-up (steady-state)", "")
    ss[:μ_w] = (get_untransformed_values(m[:μ_w]), "Wage mark-up (steady-state)", "")
    ss[:τ_prog] = (get_untransformed_values(m[:τ_prog]), "Tax progressivity (steady-state)", "")
    ss[:τ_level] = (get_untransformed_values(m[:τ_level]), "Tax level (steady-state)", "")
    ss[:σ] = (1., "Idiosyncratic risk (steady-state)", "")
    ss[:τ_prog_obs] = (1., "Observed tax progressivity (steady-state)", "")
    ss[:G_sh] = (1., "Government spending shock (steady-state)", "")
    ss[:R_sh] = (1., "MP shock (steady-state)", "")
    ss[:P_sh] = (1., "Progressivity shock (steady-state)", "")
    ss[:S_sh] = (1., "Idiosyncratic risk shock (steady-state)", "")
    ss[:rk] = (1. + _bbl_interest(get_untransformed_values(m[:K_star]), 1. / m[:μ_p],
                                  get_untransformed_values(m[:N_star]), m[:α], m[:δ_0]), "Rental rate on capital (steady-state)", "")
    ss[:LP] = (1. + ss[:rk][1] - ss[:RB][1], "Liquidity premium (ex-post) (steady-state)", "")
    ss[:LPXA] = (1. + ss[:rk][1] - ss[:RB][1], "Liquidity premium (ex-ante) (steady-state)", "")
    ss[:π] = (1., "Inflation (steady-state)", "")
    ss[:π_w] = (1., "Wage Inflation (steady-state)", "")
    ss[:BD] = (-sum(m[:marginal_pdf_m_star] .* (m.grids[:m_grid].points .< 0.) .* m.grids[:m_grid].points), "Debt (steady-state)", "")
    ss[:C] = (m[:Y_star] - m[:δ_0] * m[:K_star] - m[:G_star] - m[:Rbar] * ss[:BD][1], "Consumption (steady-state)", "")
    ss[:q] = (1., "Capital price (steady-state)", "")
    ss[:mc] = (1. / m[:μ_p], "Marginal cost (steady-state)", "")
    ss[:mc_w] = (1. / m[:μ_w], "Wage marginal cost (steady-state)", "")
    ss[:mc_w_w] = (m[:w_star] * ss[:mc_w][1], "Wage marginal cost (steady-state)", "")
    ss[:u] = (1., "Capital utilization rate (steady-state)", "")
    ss[:profits] = ((1. - ss[:mc][1]) * m[:Y_star], "Profits (steady-state)", "")
    ss[:union_profits] = ((1. - ss[:mc_w][1]) * m[:w_star] * m[:N_star], "Union Profits (steady-state)", "")
    ss[:BY] = (m[:B_star] / m[:Y_star], "Bond to output ratio (steady-state)", "")
    ss[:TY] = (m[:T_star] / m[:Y_star], "Tax revenue to output ratio (steady-state)", "")
    ss[:T_l1] = (get_untransformed_values(m[:T_star]), "Tax revenue first lag (steady-state)", "")
    ss[:Y_l1] = (get_untransformed_values(m[:Y_star]), "Output first lag (steady-state)", "")
    ss[:B_l1] = (get_untransformed_values(m[:B_star]), "Bond supply first lag (steady-state)", "")
    ss[:G_l1] = (get_untransformed_values(m[:G_star]), "Government spending first lag (steady-state)", "")
    ss[:I_l1] = (get_untransformed_values(m[:I_star]), "Investment first lag (steady-state)", "")
    ss[:w_l1] = (get_untransformed_values(m[:w_star]), "Wages first lag (steady-state)", "")
    ss[:q_l1] = (ss[:q][1], "Capital price first lag (steady-state)", "")
    ss[:C_l1] = (ss[:C][1], "Consumption first lag (steady-state)", "")
    ss[:avg_tax_rate_l1] = (get_untransformed_values(m[:avg_tax_rate_star]), "Average tax rate first lag (steady-state)", "")
    ss[:τ_prog_l1] = (get_untransformed_values(m[:τ_prog]), "Tax progressivity first lag (steady-state)", "")
    ss[:Ygrowth] = (1., "Output growth (steady-state)", "")
    ss[:Bgrowth] = (1., "Bond supply growth (steady-state)", "")
    ss[:Igrowth] = (1., "Investment growth (steady-state)", "")
    ss[:wgrowth] = (1., "Wages growth (steady-state)", "")
    ss[:Cgrowth] = (1., "Consumption growth (steady-state)", "")
    ss[:Tgrowth] = (1., "Tax revenue growth (steady-state)", "")
    ss[:Ht] = (1., "Ht (steady-state)", "")
    ss[:retained] = (1., "Retained earnings of the monopolisticaly competitive intermediate firm sector, shifted by 1.0 (steady-state)", "")
    ss[:firm_profits] = (ss[:profits][1], "Firm profits (steady-state)", "")
    ss[:union_retained] = (1., "Retained earnings of the monopolistic union sector, shifted by 1.0 (steady-state)", "")
    ss[:union_firm_profits] = (ss[:union_profits][1], "Union profits (steady-state)", "")
    ss[:tot_retained_Y] = (1., "Exponential of the retained earnings to gdp ratio (equal to zero) (steady-state)", "")

    for (k, v) in ss
        m <= SteadyStateParameter(Symbol(k, :_star), v[1], description = v[2], tex_label = v[3])
    end

    return m
end

function model_settings!(m::BayerBornLuetticke)

    ## Defaults and overrides of defaults
    default_settings!(m)

    # Likelihood method
    m <= Setting(:use_chand_recursion, false)

    # Anticipated shocks
    m <= Setting(:n_anticipated_shocks, 0, "Number of anticipated policy shocks")
    m <= Setting(:n_anticipated_shocks_padding, 0, "Padding for anticipated policy shocks")

    ## Numerical settings for steady state

    # Coarse grid settings
    m <= Setting(:coarse_ϵ,  1e-5, "Steady-state tolerance for coarse grid")
    m <= Setting(:coarse_ny, 6, "Number of idiosyncratic income states for coarse grid")
    m <= Setting(:coarse_nm, 40, "Number of liquid asset (bond) points for coarse grid")
    m <= Setting(:coarse_nk, 41, "Number of illiquid asset (capital) points for coarse grid")
    m <= Setting(:coarse_y_bin_bounds, Vector{Float64}(undef, 0),
                 "Bounds of the bins of the income states on coarse grid")
    m <= Setting(:coarse_ymin, 0.5, "Minimum grid value for income states on coarse grid")
    m <= Setting(:coarse_ymax, 1.5, "Maximum grid value for income states on coarse grid")
    m <= Setting(:coarse_mmin, -6.6, "Minimum grid value for liquid assets (bond) on coarse grid")
    m <= Setting(:coarse_mmax, 1000., "Maximum grid value for liquid assets (bond) on coarse grid")
    m <= Setting(:coarse_kmin, 0., "Minimum grid value for illiquid assets (capital) on coarse grid")
    m <= Setting(:coarse_kmax, 1500., "Maximum grid value for illiquid assets (capital) on coarse grid")

    # Refined grid settings
    m <= Setting(:ϵ, 1e-10, "Steady-state tolerance for refined grid")
    m <= Setting(:ny, 22, "Number of idiosyncratic income states for refined grid")
    m <= Setting(:nm, 80, "Number of liquid asset (bond) points for refined grid")
    m <= Setting(:nk, 80, "Number of illiquid asset (capital) points for refined grid")
    m <= Setting(:y_bin_bounds, Vector{Float64}(undef, 0),
                 "Bounds of the bins of the income states on coarse grid")
    m <= Setting(:ymin, 0.5, "Minimum grid value for income states on refined grid")
    m <= Setting(:ymax, 1.5, "Maximum grid value for income states on refined grid")
    m <= Setting(:mmin, -6.6, "Minimum grid value for liquid assets (bond) on refined grid")
    m <= Setting(:mmax, 1000., "Maximum grid value for liquid assets (bond) on refined grid")
    m <= Setting(:kmin, 0., "Minimum grid value for illiquid assets (capital) on refined grid")
    m <= Setting(:kmax, 1500., "Maximum grid value for illiquid assets (capital) on refined grid")

    # Consumption policy iteration
    m <= Setting(:max_value_function_iters, 1000, "Maximum number of fixed point iterations for the marginal value functions")

    # Kolmogorov forward equation
    m <= Setting(:kfe_method, :krylov, "Method for solving Kolmogorov forward equation")
    m <= Setting(:n_direct_transition_iters, 10_000,
                 "Number of iterations when approximating stationary distribution directly as a limit of the transition equation")

    # Reduction settings for the following reduction strategy:
    # (1) Keep DCT coefficients of value functions that explain some fraction of total "energy"
    # (2) Perturb marginals, which are mapped into perturbations of the
    #     actual distribution through the copula. However, the copula is fixed in this perturbation.
    # (3) Keep DCT coefficients of distribution over idiosyncratic states to approximate perturbations
    #     in the copula while keeping the marginals fixed.
    # (4) Remove even more basis functions
    m <= Setting(:dct_energy_loss, 1e-5, "Lost fraction of 'energy' in the DCT compression of 'value functions'")
    m <= Setting(:n_copula_dct_coefficients, 10, "Number of coefficients in the DCT compression of the " *
                 "distribution over idiosyncratic states to approximate a perturbation in the copula")
    m <= Setting(:remove_non_volatile_basis_functions, false, "Remove non-volatile basis functions for further compression")

    # Whether one wishes to re-compute the steady state
    m <= Setting(:recompute_steady_state, false, "Flag to avoid recomputing steady-state.")

    ## Numbers of indices
    #  We declare these settings here to initialize them. The numbers of indices
    #  will be updated during solution due to reduction steps by setup_indices!,
    #  which will also update the mappings from variable names to indices.
    m <= Setting(:n_scalar_jumps, 16, "Number of scalar jumps")
    m <= Setting(:n_scalar_states, 16, "Number of scalar states")
    m <= Setting(:n_scalar_variables,  get_setting(m, :n_scalar_jumps) + get_setting(m, :n_scalar_states), "Number of scalars (jumps and states)")
    m <= Setting(:n_idiosyncratic_states, get_setting(m, :ny) + get_setting(m, :nk) + get_setting(m, :nm),
                 "Number of idiosyncratic states")
    m <= Setting(:n_dct_variables, Dict(:Vm => 1, :Vk => 1, :copula => 1),
                  "Dictionary specifying the number of DCT coefficients for function-valued variables")

    m <= Setting(:n_states, get_setting(m, :n_idiosyncratic_states) + get_setting(m, :n_scalar_states) - 3, # subtract 3 b/c remove degree of freedom
                 "Total number of states after reduction steps")                           # for each dimension of copula (marginals integrate to 1)
    m <= Setting(:n_jumps, (sum(values(get_setting(m, :n_dct_variables))) - get_setting(m, :n_dct_variables)[:copula]) +
                 get_setting(m, :n_scalar_jumps), "Total number of jumps after reduction steps")
    m <= Setting(:nvars, get_setting(m, :n_states) + get_setting(m, :n_jumps), "Number of variables")

    # Number of states and jumps
    m <= Setting(:n_predetermined_variables, 0, "Number of predetermined variables after
                 removing extra degrees of freedom from the distribution states.
                 This setting is initialized at 0 as a default value because it will always be
                 overwritten once the Jacobian is calculated.")

    # Function-valued variables include distributional variables
    m <= Setting(:n_function_valued_backward_looking_states, 1, "Number of function-valued" *
                 " backward looking state variables")
    m <= Setting(:n_backward_looking_distributional_vars, 1,
                 "Number of state variables that are distributional variables.")
    m <= Setting(:n_function_valued_jumps, 1, "Number of function-valued jump variables")
    m <= Setting(:n_jump_distributional_vars, 1,
                 "Number of jump variables that are distributional variables.")

    # The settings below specify how to remove extra degrees of freedom
    # from the states representing the distribution.
    # The n degrees of freedom removed depends on the distributions/dimensions
    # of heterogeneity that we have discretized over, in this case,
    # cash on hand and the skill distribution. In general the rule of
    # thumb is, remove one degree of freedom for the first endogenous distribution (cash on
    # hand), then one additional degree of freedom for each exogenous distribution (skill
    # distribution). Multiple endogenous distributions only permit removing a single degree
    # of freedom since it is then non-trivial to obtain the marginal distributions.
    m <= Setting(:remove_extra_distribution_states, true, "Whether or not to perform the" *
                 "remove distributional states before the Klein solution step. " *
                 "This removal is necessary to apply the Klein algorithm, or else indeterminacy will occur. Intuitively, " *
                 "some grid points in a histogram representation of the distribution cannot be freely perturbed because " *
                 "they are needed to ensure that the the sum of the distribution of weights equals one. " *
                 "These grid points need to be removed prior to linearization.")
    m <= Setting(:n_degrees_of_freedom_removed_state, 2,
                 "Number of degrees of freedom from the distributional variables to remove.")
    m <= Setting(:n_degrees_of_freedom_removed_jump, 0,
                 "Number of degrees of freedom from the distributional variables to remove.")

    ## Linearization method
    m <= Setting(:solution_method, :schmitt_grohe_uribe, false, "",
                 "Solution method for obtaining a reduced-form state space representation from equilibrium conditions")

    ## Estimation Settings
    #  Just using defaults for now . . .
end

"""
```
setup_indices!(m::BayerBornLuetticke)
```
sets up the indices of model states (predetermined states and jumps) and
equilibrium conditions associated with states.

This function is called during the model's initialization and
during reduction steps, which changes indices.
"""
function setup_indices!(m::BayerBornLuetticke)
# TODO: check if we need to ensure that this function doesn't normalize indices
    # Abbreviate some fields
    state_vars = m.state_variables
    jump_vars = m.jump_variables
    endo = m.endogenous_states # note that these are the model states, so they include predetermined states and jumps
    eqconds = m.equilibrium_conditions

    # Update number of scalar states and jumps
    m <= Setting(:n_scalar_states, length(get_aggregate_state_variables(m)))
    m <= Setting(:n_scalar_jumps, length(get_aggregate_jump_variables(m)))

    # Compute size of idiosyncratic state space
    n_idio_states = get_setting(m, :n_idiosyncratic_states)
    n_dct_vars    = get_setting(m, :n_dct_variables)
    nm, nk, ny    = get_setting(m, :nm), get_setting(m, :nk), get_setting(m, :ny)

    ## Populate endo using "next-period" name (i.e., using ′) since
    #  it is easier to remove the ′ than to add it, a feature that
    #  is used by normalize_state_indices!
    endo[:marginal_m′_t] = 1:nm
    endo[:marginal_k′_t] = (1 + nm):(nm + nk)
    endo[:marginal_y′_t] = (1 + nm + nk):n_idio_states
    n_distr_states       = n_idio_states + n_dct_vars[:copula] # number of distributional states: marginals + DCT of copula
    endo[:copula′_t]     = (1 + n_idio_states):n_distr_states
    for (i, k) in enumerate(state_vars[2:end])
        endo[k] = (n_distr_states + i):(n_distr_states + i)
    end

    # Update n_states to be consistent with the number of
    # idiosyncratic states and jumps after reduction
    n_states = first(endo[state_vars[end]])
    m <= Setting(:n_states, n_states)

    # Now populate jump indices
    n_idio_jumps = n_dct_vars[:Vm] + n_dct_vars[:Vk]
    endo[:Vm′_t] = (n_states + 1):(n_states + n_dct_vars[:Vm])
    endo[:Vk′_t] = (n_states + n_dct_vars[:Vm] + 1):(n_states + n_idio_jumps)
    for (i, k) in enumerate(jump_vars[3:end])
        endo[k] = (n_states + n_idio_jumps + i):(n_states + n_idio_jumps + i)
    end
    n_jumps = first(endo[jump_vars[end]])
    m <= Setting(:n_jumps, n_jumps)

    ## Populate equation indices

    # Function blocks which output a function
    n_eqconds = n_distr_states + n_idio_jumps # number of function-valued equilibrium conditions
    eqconds[:eq_marginal_distr_m] = endo[:marginal_m′_t] # note that since endo holds UnitRanges, this assignment results in a copy
    eqconds[:eq_marginal_distr_k] = endo[:marginal_k′_t]
    eqconds[:eq_marginal_distr_y] = endo[:marginal_y′_t]
    eqconds[:eq_copula]           = endo[:copula′_t]
    eqconds[:eq_marginal_value_bonds] = (1 + n_distr_states):(n_dct_vars[:Vm] + n_distr_states)
    eqconds[:eq_marginal_value_capital] = (1 + n_distr_states + n_dct_vars[:Vk]):n_eqconds

    # Function blocks which map functions to scalars
    for (i, name) in enumerate([:eq_agg_capital, :eq_agg_bond, :eq_agg_hh_debt,
                                :eq_τ_level, :eq_total_tax_revenue, :eq_Ht,
                                :eq_GiniX, :eq_I90_share, :eq_I90_share_net,
                                :eq_W90_share, :eq_sd_log_y, :eq_GiniC])
        eqconds[name] = (n_eqconds + i):(n_eqconds + i)
    end
    n_eqconds += first(eqconds[:eq_GiniC]) # Increment this variable, so it now includes functional (function to scalar) blocks

    # Scalar blocks (for aggregate variable)
    for (i, name) in enumerate([# Endogenous model states (for the jumps)
                                :eq_mp, :eq_tax_progressivity, :eq_tax_level,
                                :eq_tax_revenue, :eq_avg_tax_rate,
                                :eq_deficit_rule, :eq_gov_budget_constraint,
                                :eq_price_phillips_curve, :eq_wage_phillips_curve,
                                :eq_wage_growth, :eq_capital_util, :eq_capital_return,
                                :eq_received_wages, :eq_wages_firms_pay, :eq_union_firm_profits,
                                :eq_firm_profits, :eq_profits_distr_to_hh,
                                :eq_tobins_q, :eq_expost_liquidity_premium,
                                :eq_exante_liquidity_premium,
                                :eq_capital_accum, :eq_labor_supply, :eq_output,
                                :eq_resource_constraint, :eq_capital_market_clear,
                                :eq_debt_market_clear, :eq_bond_market_clear,
                                :eq_bond_output_ratio, :eq_tax_output_ratio,
                                :eq_retained_earnings_gdp_ratio, :eq_Ht,

                                # Growth rates
                                :eq_Ygrowth, :eq_Tgrowth, :eq_Bgrowth,
                                :eq_Igrowth, :eq_wgrowth, :eq_Cgrowth,

                                # Lagged variables
                                :eq_LY, :eq_LB, :eq_LI, :eq_Lw, :eq_LT,
                                :eq_Lq, :eq_LC, :eq_Lavg_tax_rate,
                                :eq_Lτ_prog,

                                # Exogenous shocks
                                :eq_A, :eq_Z, :eq_Ψ, :eq_μ_p, :eq_μ_w,
                                :eq_σ, :eq_G, :eq_P, :eq_R, :eq_S])
        eqconds[name] = (n_eqconds + i):(n_eqconds + i)
    end
end

# TODO: init_states_and_jumps! is not done. Its role appears to be prepping
#       index info for Klein, but since we're using SGU, so maybe we don't need it
function init_states_and_jumps!(m::AbstractModel, states::Vector{Symbol},
                                jumps::Vector{Symbol}, states_only::Bool = false)

    endo = m.endogenous_states

    m <= Setting(:states, states)
    m <= Setting(:jumps,  jumps)

    state_indices = stack_indices(endo, states)
    jump_indices  = stack_indices(endo, jumps)

    m <= Setting(:state_indices, state_indices, "Indices of m.endogenous_states " *
                 "corresponding to backward-looking state variables")
    m <= Setting(:jump_indices, jump_indices, "Indices of m.endogenous_states " *
                 "corresponding to jump
                 variables")

    n_dof_removed_state = get_setting(m, :n_degrees_of_freedom_removed_state)
    n_dof_removed_jump  = get_setting(m, :n_degrees_of_freedom_removed_jump)


    ####################################################
    # Calculating the number of backward-looking states
    ####################################################
    n_backward_looking_distr_vars = get_setting(m, :n_backward_looking_distributional_vars)
    m <= Setting(:backward_looking_states_normalization_factor,
                 n_dof_removed_state*n_backward_looking_distr_vars,
                 "Number of dimensions removed from the backward looking state variables
                  for the normalization.")

    nxns_state = (get_setting(m, :nx1_state) + get_setting(m, :nx2_state))* get_setting(m, :ns)
    nxns_jump  = (get_setting(m, :nx1_jump)  + get_setting(m, :nx2_jump)) * get_setting(m, :ns)
    nx1_state  = get_setting(m, :nx1_state)
    nx2_state  = get_setting(m, :nx2_state)
    nx1_jump   = get_setting(m, :nx1_jump)
    nx2_jump   = get_setting(m, :nx2_jump)

    n_backward_looking_vars = length(get_setting(m, :state_indices))
    n_backward_looking_function_valued_vars = get_setting(m,
                                               :n_function_valued_backward_looking_states)
    n_backward_looking_scalar_vars = get_setting(m, :nxscalars)

    m <= Setting(:n_backward_looking_states, (nx1_state + nx2_state) *
                 n_backward_looking_distr_vars +
                 n_backward_looking_scalar_vars -
                 get_setting(m, :backward_looking_states_normalization_factor),
                 "Number of state variables, in the true sense (fully
                  backward looking) accounting for the discretization across the grid
                  of function-valued variables and the normalization of
                  distributional variables.")

    ##################################
    # Calculating the number of jumps
    ##################################
    n_jump_distr_vars = get_setting(m, :n_jump_distributional_vars)
    m <= Setting(:jumps_normalization_factor,
                 n_dof_removed_jump*n_jump_distr_vars, "number of dimensions removed from" *
                 " the jump variables for the normalization.")

    n_jump_vars = length(get_setting(m, :jump_indices))
    n_jump_function_valued_vars = get_setting(m, :n_function_valued_jumps)
    n_jump_scalar_vars = get_setting(m, :nyscalars) #n_jump_vars - (nx1+nx2)*n_jump_function_valued_vars

    m <= Setting(:n_jumps, (nx1_jump+nx2_jump)*n_jump_function_valued_vars +
                 n_jump_scalar_vars - get_setting(m, :jumps_normalization_factor),
                 "Number of jump variables (forward looking) accounting for
                  the discretization across the grid of function-valued variables and the
                  normalization of distributional variables.")

    m <= Setting(:n_model_states, get_setting(m, :n_backward_looking_states) +
                 get_setting(m, :n_jumps),
                 "Number of 'states' in the state space model. Because backward and forward
                 looking variables need to be explicitly tracked for the Klein solution
                 method, we have n_states and n_jumps")
end

"""
```
reset_grids!(m::BayerBornLuetticke)
```
This is a very important function. ANy time you reduce the grid, you need to restore to the full grid before evaluating steadystate/jacobian/likelihood again. Otherwise, you'll keep redsucing and after just a few times, you'll be left with NOTHING. So PSA: reset your grid when you're done with whatever you need to do with the smaller grid!
"""
function reset_grids!(m::BayerBornLuetticke)
    # TODO: reset_grids is not done, may need to update settings to have "default" values saved for the grid

    m <= Setting(:nx1_state, 300)
    m <= Setting(:nx2_state, 300)
    m <= Setting(:nx1_jump,  300)
    m <= Setting(:nx2_jump,  300)

    setup_indices!(m)
    init_states_and_jumps!(m, get_setting(m, :states), get_setting(m, :jumps))
    init_grids!(m)

    # So that the indices of m.endogenous_states reflect the normalization
    normalize_model_state_indices!(m)

    endogenous_states_augmented = [:C_t1]
    for (i,k) in enumerate(endogenous_states_augmented)
        m.endogenous_states_augmented[k] = i +
            first(m.endogenous_states[get_setting(m, :jumps)[end]])
    end
    m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states) +
                 length(m.endogenous_states_augmented))
end
