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
# TODO: maybe add field/type to hold reduction information e.g. DCT indices (but not coefficient values), copula info
    grids::OrderedDict{Symbol,Union{Grid, Array, T}}
    keys::OrderedDict{Symbol,Int}                    # Human-readable names for all the model
                                              # parameters and steady-states

    state_variables::Vector{Symbol}                  # Vector of symbols of the state variables
    jump_variables::Vector{Symbol}                   # Vector of symbols of the jump variables
    aggregate_state_variables::Vector{Symbol}                  # Vector of symbols of the state variables
    aggregate_jump_variables::Vector{Symbol}                   # Vector of symbols of the jump variables
#=    normalized_model_states::Vector{Symbol}          # All of the distributional model
                                                     # state variables that need to be normalized=#

    # Vector of ranges corresponding to normalized (post Klein solution) indices
    aggregate_endogenous_states::OrderedDict{Symbol,Int}
    endogenous_states::OrderedDict{Symbol,UnitRange{Int}}
    exogenous_shocks::OrderedDict{Symbol,Int}
    expected_shocks::OrderedDict{Symbol,Int}
    equilibrium_conditions::OrderedDict{Symbol,UnitRange{Int}}
    aggregate_equilibrium_conditions::OrderedDict{Symbol,Int}
    endogenous_states_augmented::OrderedDict{Symbol,Int}
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
    m.state_variables = [:marginal_pdf_m′_t, :marginal_pdf_k′_t, :marginal_pdf_y′_t, :copula′_t,

                         :A′_t, :Z′_t, :Ψ′_t, :RB′_t, :μ_p′_t, :μ_w′_t, :σ′_t, # TODO: rename σ′_t to σ_sq′_t to make it clear it's the variance
                         :union_retained′_t, :retained′_t,
                         :Y′_t1, :B′_t1, :T′_t1, :I′_t1, :w′_t1, :q′_t1, :C′_t1,
                         :avg_tax_rate′_t1, :τ_prog′_t1,

                         :G_sh′_t, :P_sh′_t, :R_sh′_t, :S_sh′_t]

#=    m.state_variables = [# Endogenous function-valued states
                         :marginal_pdf_m′_t, :marginal_pdf_k′_t, :marginal_pdf_y′_t, :copula′_t,

                         # Endogenous scalar-valued states (e.g. lags)
                         :union_retained′_t, :retained′_t,
                         :Y′_t1, :B′_t1, :T′_t1, :I′_t1, :w′_t1, :q′_t1, :C′_t1,
                         :avg_tax_rate′_t1, :τ_prog′_t1,

                         # Exogenous scalar-valued states:
                         :A′_t, :Z′_t, :Ψ′_t, :RB′_t, :μ_p′_t, :μ_w′_t, :σ′_t,
                         :G_sh′_t, :P_sh′_t, :R_sh′_t, :S_sh′_t]=#

    m.aggregate_state_variables = m.state_variables[5:end]

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
                        :Tgrowth′_t, :LP′_t, :LP_XA′_t, :tot_retained_Y′_t,
                        :union_firm_profits′_t, :union_profits′_t, :firm_profits′_t,
                        :profits′_t]

    m.aggregate_jump_variables = m.jump_variables[3:end]

    # Update number of scalar states and jumps
    m <= Setting(:n_scalar_states, length(get_aggregate_state_variables(m)))
    m <= Setting(:n_scalar_jumps, length(get_aggregate_jump_variables(m)))
    m <= Setting(:n_scalar_variables,  get_setting(m, :n_scalar_jumps) + get_setting(m, :n_scalar_states))

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
    # setup_indices!(m) # TODO: This might have to be called AFTER steadystate! is called and leave entries empty otherwise

    # Create dict for unreduced endogenous_states
    # to facilitate repeated solutions of steady state
#=    m <= Setting(:n_model_states_unreduced, length(m.endogenous_states),
                 "Number of model states (incl. both predetermined states and jumps) before any state-space reduction")=#

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
                            load_steadystate::Bool = false, load_jacobian::Bool = false,
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
            OrderedDict{Symbol,Union{Grid, Array, Float64}}(), OrderedDict{Symbol,Int}(),

            # state_variables, jump_variables,
            Vector{Symbol}(), Vector{Symbol}(),

            # aggregate_state_variables, aggregate_jump_variables
            Vector{Symbol}(), Vector{Symbol}(),

            # model indices
            # endogenous states
            OrderedDict{Symbol, Int}(), # TODO: label these
            OrderedDict{Symbol, UnitRange{Int}}(), # TODO: label these
            OrderedDict{Symbol, Int}(), OrderedDict{Symbol, Int}(),
            OrderedDict{Symbol, UnitRange{Int}}(), OrderedDict{Symbol, Int}(),
            OrderedDict{Symbol, Int}(), OrderedDict{Symbol, Int}(),
            OrderedDict{Symbol, Int}(),

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

    if get_setting(m, :original_dataset)
        _init_original_observable_mappings!(m, m.observables)
    end

    # Initialize model indices
    init_model_indices!(m)

    # Init what will keep track of # of states, jumps, and states/jump indices
    # (need model indices first)
    # init_states_and_jumps!(m, states, jumps)

    # Initialize parameters
    init_parameters!(m)

    # Initialize grids
    init_grids!(m; coarse = !load_steadystate) # if steady state has not been computed, we start from a coarse grid

    # Load the steady state if it has already been computed
    # from the filepath get_setting(m, :steadystate_output_file)
    if load_steadystate
        load_steadystate!(m)
    end
    # steadystate!(m)

    # So that the indices of m.endogenous_states reflect the normalization
    # normalize_model_state_indices!(m)

    init_subspec!(m)

    # Load Jacobian if it has already been computed
    if load_jacobian
        load_jacobian!(m)
    end

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
    m <= parameter(:π, 1.0 ^ 0.25, fixed = true,
                   description = "Steady-state inflation", tex_label = "\\pi")

    # Monetary policy
    m <= parameter(:RB, m[:π] * 1.0 ^ 0.25, fixed = true,
                   description = "Steady-state nominal interest rate", tex_label = "\\RB")

    # Remaining parameters affecting the steady-state
    m <= parameter(:ψ, 0.1, fixed = true,
                   description = "Steady-state bond to capital ratio", tex_label = "\\psi")
    m <= parameter(:τ_lev, 0.825, fixed = true,
                   description = "Steady-state income tax rate level", tex_label = "\\tau^L")
    m <= parameter(:τ_prog, 0.12, fixed = true,
                   description = "Steady-state income tax rate progressivity", tex_label = "\\tau^P")
    m <= parameter(:R, 1.01, fixed = true, # TODO: delete b/c unused
                   description = "Steady-state return of capital (unused)", tex_label = "R")
    m <= parameter(:K, 40., fixed = true, # TODO: delete b/c unused
                   description = "Steady-state quantity of capital (unused)", tex_label = "K")
    m <= parameter(:Rbar, (m[:π] * (1.0675 ^ 0.25) - 1.), fixed = true,
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
    m <= parameter(:ρ_R , 0.9, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(),
                   BetaAlt(0.5, 0.20), fixed = false,
                   description = "ρ: The degree of inertia in the monetary policy rule.",
                   tex_label="\\rho_R")
    m <= parameter(:θ_π, 2., (1., 10.), (1e-5, 10.0), ModelConstructors.Exponential(),
                   Normal(1.7, 0.3), fixed = false, # Note second tuple is parameterization for Exponential transform
                   description = "ψ1: Weight on inflation gap in monetary policy rule.",
                   tex_label = "\\theta_{\\pi}")
    m <= parameter(:θ_Y, 0.125, (-5., 5.), (-5., 5.), SquareRoot(),
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
    m <= parameter(:γ_B_τ, 0., (-10., 10.), (10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_B_τ: Reaction of tax level to debt",
                   tex_label = "\\gamma_{B, \\tau}")
    m <= parameter(:γ_Y_τ, 0., (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_Y_τ: Reaction of tax level to output",
                   tex_label = "\\gamma_{Y, \\tau}")
    m <= parameter(:ρ_P, 0.5, (1e-5, 1. - 1e-5), (1e-5, 1. - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_P: Persistence in tax level",
                   tex_label = "\\rho_{P}")
    m <= parameter(:γ_B_P, 0., (-10., 10.), (10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_B_P: Reaction of tax level to debt",
                   tex_label = "\\gamma_{B, P}")
    m <= parameter(:γ_Y_P, 0., (-10., 10.), (-10., 10.), SquareRoot(),
                   Normal(0., 1.), fixed = false,
                   description = "γ_Y_P: Reaction of tax level to output",
                   tex_label = "\\gamma_{Y, P}")

    # Exogenous processes - autocorrelation
    m <= parameter(:ρ_A, 0.9, (1e-5, 1. - 1e-5), (1e-5, 1. - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_A: AR(1) coefficient in the bond-spread process.",
                   tex_label = "\\rho_A")
    m <= parameter(:ρ_Z, 0.9, (1e-5, 1. - 1e-5), (1e-5, 1. - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_Z: AR(1) coefficient in the technology process.",
                   tex_label = "\\rho_Z")
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
    m <= parameter(:ρ_R_sh, 1e-8, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_R_ϵ: AR(1) coefficient in the monetary policy shock process.",
                   tex_label = "\\rho_{R, \\epsilon}")
    m <= parameter(:ρ_P_sh, 1e-8, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_P_ϵ: AR(1) coefficient in the tax progressivity shock process.",
                   tex_label = "\\rho_{P, \\epsilon}")
    m <= parameter(:ρ_S_sh, 1e-8, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = true,
                   description = "ρ_S_ϵ: AR(1) coefficient in shock process of the shock in the idiosyncatic income shock process.",
                   tex_label = "\\rho_{S, \\epsilon}")

    # Exogenous processes - standard deviations
    m <= parameter(:σ_A, 0.01, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false, # Note second tuple is parameterization for Exponential transform
                   description = "σ_A: standard dev. of the bond-spread process.",
                   tex_label = "\\sigma_{A}")
    m <= parameter(:σ_Z, 0.01, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false,
                   description = "σ_Z: standard dev. of the process describing the " *
                   "stationary component of productivity.", tex_label = "\\sigma_Z")
    m <= parameter(:σ_Ψ, 0.01, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false,
                   description = "σ_Ψ: standard dev. of the exogenous marginal efficiency" *
                   " of investment shock process.", tex_label = "\\sigma_{\\Psi}")
    m <= parameter(:σ_μ_p, 0.01, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false,
                   description = "σ_μ_p: standard dev. of the price mark-up shock process",
                   tex_label = "\\sigma_{\\mu_p}")
    m <= parameter(:σ_μ_w, 0.01, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false,
                   description = "σ_μ_w: : standard dev. of the wage mark-up shock process",
                   tex_label = "\\sigma_{\\mu_w}")
    m <= parameter(:σ_S, 0.01, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   GammaAlt(0.65, 0.3), fixed = false,
                   description = "σ_S: standard dev. of the idiosyncratic income risk shock process",
                   tex_label = "\\sigma_{S}")
    m <= parameter(:Σ_n, 0., (-1e3, 1e3), (-1e3, 1e3), ModelConstructors.SquareRoot(),
                   Normal(0., 100.), fixed = false,
                   description = "Σ_n: reaction of income risk to employment status",
                   tex_label = "\\Sigma_{n}")
    m <= parameter(:σ_R, 0.01, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.10), fixed = false,
                   description = "σ_R_ϵ: standard dev. of the monetary policy shock process",
                   tex_label = "\\sigma_{R, \\epsilon}")
    m <= parameter(:σ_G, 0.01, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   GammaAlt(0.65, 0.3), fixed = false,
                   description = "σ_G: standard dev. of the structural deficit shock process",
                   tex_label = "\\sigma_{G}")
    m <= parameter(:σ_P, 0.01, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2., 0.10), fixed = false,
                   description = "σ_P: standard dev. of the tax progressivity shock process",
                   tex_label = "\\sigma_{P}")

    # Measurement error
    m <= parameter(:e_W90_share, 0.2, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2., 0.10), fixed = false,
                   description = "σ_W90_share: standard dev. of measurement error for 90th percentile of wealth distribution",
                   tex_label = "\\sigma_{W^{(90)}}")
    m <= parameter(:e_I90_share, 0.2, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2., 0.10), fixed = false,
                   description = "σ_I90_share: standard dev. of measurement error for 90th percentile of income distribution",
                   tex_label = "\\sigma_{I^{(90)}}")
    m <= parameter(:e_τ_prog, 0.2, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2., 0.10), fixed = false,
                   description = "σ_P_me: standard dev. of measurement error for tax progressivity",
                   tex_label = "\\sigma_{P, me}")
    m <= parameter(:e_σ, 0.2, (0., 5.), (0., 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2., 0.10), fixed = false,
                   description = "σ_S_me: standard dev. of measurement error for idiosyncratic income risk",
                   tex_label = "\\sigma_{S, me}")

    # Steady-state parameters

    # Aggregate scalars (just initialized here, these will be populated by the steadystate!)
    m <= SteadyStateParameter(:K_star, NaN, description = "Capital stock (steady-state)",
                              tex_label = "K_*")
    m <= SteadyStateParameter(:N_star, NaN, description = "Labor supply (steady-state)",
                              tex_label = "N_*")
    m <= SteadyStateParameter(:Y_star, NaN, description = "Output (steady-state)",
                              tex_label = "Y_*")
    m <= SteadyStateParameter(:G_star, NaN, description = "Government spending (steady-state)",
                              tex_label = "G_*")
    m <= SteadyStateParameter(:w_star, NaN, description = "Wage (steady-state)",
                              tex_label = "w_*")
    m <= SteadyStateParameter(:T_star, NaN, description = "Tax revenue (steady-state)",
                              tex_label = "T_*")
    m <= SteadyStateParameter(:I_star, NaN, description = "Investment (steady-state)",
                              tex_label = "I_*")
    m <= SteadyStateParameter(:B_star, NaN, description = "Bond supply (steady-state)",
                              tex_label = "B_*")
    m <= SteadyStateParameter(:avg_tax_rate_star, NaN, description = "Average tax rate (steady-state)",
                              tex_label = "avgtaxrate_*")

    # Aggregate scalars computed after solving steady state
    m <= SteadyStateParameter(:A_star, NaN, description = "Bond spread (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Z_star, NaN, description = "TFP (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Ψ_star, NaN, description = "MEI (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:RB_star, NaN, description = "MEI (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:μ_p_star, NaN, description = "Price mark-up (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:μ_w_star, NaN, description = "Wage mark-up (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:τ_prog_star, NaN, description = "Tax progressivity (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:τ_level_star, NaN, description = "Tax level (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:σ_star, NaN, description = "Idiosyncratic risk (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:τ_prog_obs_star, NaN, description = "Observed tax progressivity (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:G_sh_star, NaN, description = "Government spending shock (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:R_sh_star, NaN, description = "MP shock (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:P_sh_star, NaN, description = "Progressivity shock (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:S_sh_star, NaN, description = "Idiosyncratic risk shock (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:rk_star, NaN, description = "Rental rate on capital (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:LP_star, NaN, description = "Liquidity premium (ex-post) (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:LP_XA_star, NaN, description = "Liquidity premium (ex-ante) (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:π_star, NaN, description = "Inflation (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:π_w_star, NaN, description = "Wage Inflation (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:BD_star, NaN, description = "Debt (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:C_star, NaN, description = "Consumption (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:q_star, NaN, description = "Capital price (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:mc_star, NaN, description = "Marginal cost (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:mc_w_star, NaN, description = "Wage marginal cost (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:mc_w_w_star, NaN, description = "Wage marginal cost (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:u_star, NaN, description = "Capital utilization rate (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:profits_star, NaN, description = "Profits (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:union_profits_star, NaN, description = "Union Profits (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:BY_star, NaN, description = "Bond to output ratio (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:TY_star, NaN, description = "Tax revenue to output ratio (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:T_l1_star, NaN, description = "Tax revenue first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Y_l1_star, NaN, description = "Output first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:B_l1_star, NaN, description = "Bond supply first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:G_l1_star, NaN, description = "Government spending first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:I_l1_star, NaN, description = "Investment first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:w_l1_star, NaN, description = "Wages first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:q_l1_star, NaN, description = "Capital price first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:C_l1_star, NaN, description = "Consumption first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:avg_tax_rate_l1_star, NaN, description = "Average tax rate first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:τ_prog_l1_star, NaN, description = "Tax progressivity first lag (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Ygrowth_star, NaN, description = "Output growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Bgrowth_star, NaN, description = "Bond supply growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Igrowth_star, NaN, description = "Investment growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:wgrowth_star, NaN, description = "Wages growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Cgrowth_star, NaN, description = "Consumption growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Tgrowth_star, NaN, description = "Tax revenue growth (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:Ht_star, NaN, description = "Ht (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:retained_star, NaN,
                              description = "Retained earnings of the monopolisticaly competitive intermediate" *
                              " firm sector, shifted by 1.0 (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:firm_profits_star, NaN, description = "Firm profits (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:union_retained_star, NaN, description = "Retained earnings of the monopolistic union" *
                              " sector, shifted by 1.0 (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:union_firm_profits_star, NaN, description = "Union profits (steady-state)", tex_label = "")
    m <= SteadyStateParameter(:tot_retained_Y_star, NaN, description = "Exponential of the retained earnings to gdp ratio" *
                              " (equal to zero) (steady-state)", tex_label = "")

    # Scalar summary statistics about the idiosyncratic states (e.g. inequality measures)
    # TODO: add descriptions to these parameters
    m <= SteadyStateParameter(:share_borrower_star, NaN)
    m <= SteadyStateParameter(:Gini_wealth_star, NaN)
    m <= SteadyStateParameter(:W90_share_star, NaN)
    m <= SteadyStateParameter(:I90_share_star, NaN)
    m <= SteadyStateParameter(:I90_share_net_star, NaN)
    m <= SteadyStateParameter(:Gini_income_star, NaN)
    m <= SteadyStateParameter(:P90_minus_P10_income_star, NaN)
    m <= SteadyStateParameter(:sd_log_y_star, NaN)
    m <= SteadyStateParameter(:Gini_X_star, NaN)
    m <= SteadyStateParameter(:sd_log_X_star, NaN)
    m <= SteadyStateParameter(:P90_minus_P10_C_star, NaN)
    m <= SteadyStateParameter(:Gini_C_star, NaN)
    m <= SteadyStateParameter(:sd_log_C_star, NaN)
    m <= SteadyStateParameter(:P10_C_star, NaN)
    m <= SteadyStateParameter(:P50_C_star, NaN)
    m <= SteadyStateParameter(:P90_C_star, NaN)

    # Steady state grids for functional/distributional variables
    total_idio_states = get_setting(m, :nm) * get_setting(m, :nk) * get_setting(m, :ny)
    m <= SteadyStateParameterGrid(:distr_star, Array{Float64, 3}(undef, 0, 0, 0), # populated by init_grids! to ensure it's consistent
                                  description = "Distribution over idiosyncratic states (steady-state)", tex_label = "D_*")
    m <= SteadyStateParameterGrid(:marginal_pdf_m_star, Vector{Float64}(undef, 0), # populated later, just need to have the right typing
                                  description = "Marginal PDF of liquid bonds (steady-state)", tex_label = "D_{m, *}")
    m <= SteadyStateParameterGrid(:marginal_pdf_k_star, Vector{Float64}(undef, 0),
                                  description = "Marginal PDF of illiquid capital (steady-state)", tex_label = "D_{k, *}")
    m <= SteadyStateParameterGrid(:marginal_pdf_y_star, Vector{Float64}(undef, 0),
                                  description = "Marginal PDF of income (steady-state)", tex_label = "D_{y, *}")
    m <= SteadyStateParameterGrid(:marginal_cdf_m_star, Vector{Float64}(undef, 0), # populated later, just need to have the right typing
                                  description = "Marginal CDF of liquid bonds (steady-state)", tex_label = "D_{m, *}")
    m <= SteadyStateParameterGrid(:marginal_cdf_k_star, Vector{Float64}(undef, 0),
                                  description = "Marginal CDF of illiquid capital (steady-state)", tex_label = "D_{k, *}")
    m <= SteadyStateParameterGrid(:marginal_cdf_y_star, Vector{Float64}(undef, 0),
                                  description = "Marginal CDF of income (steady-state)", tex_label = "D_{y, *}")
    m <= SteadyStateParameterGrid(:Vm_star, Array{Float64, 3}(undef, 0, 0, 0),
                                  description = "Marginal value of liquid bonds (steady-state)", tex_label = "V_{m, *}")
    m <= SteadyStateParameterGrid(:Vk_star, Array{Float64, 3}(undef, 0, 0, 0),
                                  description = "Marginal value of illiquid capital (steady-state)", tex_label = "V_{k, *}")

    # Steady state grids for reduction-related variables (indices for perturbation kept elsewhere) # TODO: document where compression indices are
    m <= SteadyStateParameterGrid(:dct_Vm_star, Vector{Float64}(undef, 0),
                                  description = "DCT coefficients of the marginal value of liquid bonds (steady-state)",
                                  tex_label = "\\theta_{Vm, *}")
    m <= SteadyStateParameterGrid(:dct_Vk_star, Vector{Float64}(undef, 0),
                                  description = "DCT coefficients of the marginal value of illiquid capital (steady-state)",
                                  tex_label = "\\theta_{Vk, *}")
    m <= SteadyStateParameterGrid(:dct_copula_star, Vector{Float64}(undef, 0),
                                  description = "DCT coefficients of the copula for distribution " *
                                  "over idiosyncratic states (steady-state)",
                                  tex_label = "\\theta_{D, *}")

    # Jacobians to be updated
    m <= SteadyStateParameterGrid(:A, Matrix{Float64}(undef, 0, 0),
                                  description = "The A matrix computed by jacobian(m)")
    m <= SteadyStateParameterGrid(:B, Matrix{Float64}(undef, 0, 0),
                                  description = "The B matrix computed by jacobian(m)")
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
    m <= Setting(:max_value_function_iters, 1000,
                 "Maximum number of fixed point iterations for the marginal value functions")

    # Kolmogorov forward equation
    m <= Setting(:kfe_method, :krylov, "Method for solving Kolmogorov forward equation")
    m <= Setting(:n_direct_transition_iters, 10_000,
                 "Number of iterations when approximating stationary distribution " *
                 "directly as a limit of the transition equation")

    # Reduction settings for the following reduction strategy:
    # (1) Keep DCT coefficients of value functions that explain some fraction of total "energy"
    # (2) Perturb marginals, which are mapped into perturbations of the
    #     actual distribution through the copula. However, the copula is fixed in this perturbation.
    # (3) Keep DCT coefficients of distribution over idiosyncratic states to approximate perturbations
    #     in the copula while keeping the marginals fixed.
    # (4) Remove even more basis functions
    m <= Setting(:dct_energy_loss, 1e-5, "Lost fraction of 'energy' in the DCT compression of 'value functions'")
    m <= Setting(:n_copula_dct_coefficients, 11, "Number of coefficients in the DCT compression of the " *
                 "distribution over idiosyncratic states to approximate a perturbation in the copula")
    m <= Setting(:remove_non_volatile_basis_functions, false, "Remove non-volatile basis functions for further compression")

    # Initialize storages for settings/objects related to reduction
    m <= Setting(:dct_compression_indices, Dict{Symbol, Vector{Int}}(), "DCT compression indices")
    m <= Setting(:copula, identity, "Steady-state copula")

    # Whether one wishes to re-compute the steady state
    m <= Setting(:compute_full_steadystate, true, "Flag to avoid re-computing the full steady-state.")

    ## Numbers of indices
    #  We declare these settings here to initialize them. The numbers of indices
    #  will be updated during solution due to reduction steps by setup_indices!,
    #  which will also update the mappings from variable names to indices.

    # Update number of scalar states and jumps
    m <= Setting(:n_scalar_jumps, 16, "Number of scalar jumps")
    m <= Setting(:n_scalar_states, 16, "Number of scalar states")
    m <= Setting(:n_scalar_variables,  get_setting(m, :n_scalar_jumps) + get_setting(m, :n_scalar_states),
                 "Number of scalars (jumps and states)")

    m <= Setting(:n_backward_looking_states, 1, "Total number of states after reduction steps") # just initializing, will count later
    m <= Setting(:n_jumps, 1, "Total number of jumps after reduction steps")
    m <= Setting(:n_model_states, get_setting(m, :n_backward_looking_states) + get_setting(m, :n_jumps),
                 "Number of model states (predetermined states and jump variables)")

    # Number of states and jumps
    m <= Setting(:n_predetermined_variables, 0, "Number of predetermined variables after
                 removing extra degrees of freedom from the distribution states.
                 This setting is initialized at 0 as a default value because it will always be
                 overwritten once the Jacobian is calculated.")

    ## Linearization settings
    m <= Setting(:linearize_heterogeneous_block, true, false, "", "Boolean for whether the " *
                 "heterogeneous block of the Jacobians should be linearized")
    m <= Setting(:solution_method, :klein, false, "",
                 "Solution method for obtaining a reduced-form state space representation from equilibrium conditions")
    m <= Setting(:klein_inversion_method, :minimum_norm, false, "",
                 "Inversion method to obtain gx and hx during the Klein algorithm")

    ## Replication-related settings
    m <= Setting(:replicate_original_output, false, "Use steady state and linearization functions that exactly " *
                 "replicate output from the original implementation by Bayer, Born, and Luetticke.")
    m <= Setting(:original_dataset, false, "Load original dataset used by Bayer, Born, and Luetticke for their paper.")

    ## Saving and loading steady state output and Jacobians
    m <= Setting(:save_steadystate, true)
    m <= Setting(:save_jacobian, true)
    if !ispath(rawpath(m, "estimate"))
        mkpath(rawpath(m, "estimate"))
    end
    m <= Setting(:steadystate_output_file, rawpath(m, "estimate", "steadystate.jld2"))
    m <= Setting(:jacobian_output_file, rawpath(m, "estimate", "jacobian.jld2"))

    ## Dates
    m <= Setting(:data_vintage, "210504")
    m <= Setting(:cond_vintage, "210504")
    m <= Setting(:data_id, 1793)
    m <= Setting(:cond_id, 1793)
    m <= Setting(:date_zlb_start, quartertodate("2009-Q1")) # ZLB measured using Wu and Xia (2016) shadow FFR, which starts in 2009:Q1
    m <= Setting(:date_mainsample_start, quartertodate("1954-Q4"))
    m <= Setting(:date_presample_start, quartertodate("1954-Q4"))
    m <= Setting(:date_forecast_start, quartertodate("2020-Q1"))
    m <= Setting(:date_conditional_end, quartertodate("2020-Q1"))

    ## Monetary Policy
    m <= Setting(:n_mon_anticipated_shocks_padding, 0) # anticipated shocks are not used
    m <= Setting(:monetary_policy_shock, :R_sh)

    ## Estimation and Forecasting settings
    # defaults for now
end

"""
```
setup_indices!(m::BayerBornLuetticke)
```
sets up the indices of model states (predetermined states and jumps) and
equilibrium conditions associated with states.

This function is called during the model's initialization and # TODO maybe note initialization
during reduction steps, which changes indices.
"""
function setup_indices!(m::BayerBornLuetticke)
    # Abbreviate some fields
    state_vars = m.state_variables
    jump_vars = m.jump_variables
    endo = m.endogenous_states # note that these are the model states, so they include predetermined states and jumps
    aggr_endo = m.aggregate_endogenous_states # note that these are the model states, so they include predetermined states and jumps
    eqconds = m.equilibrium_conditions
    aggr_eqconds = m.aggregate_equilibrium_conditions

    # Compute size of idiosyncratic state space
    nm, nk, ny = get_idiosyncratic_dims(m)

    ## Populate endo using "next-period" name (i.e., using ′) since
    #  it is easier to remove the ′ than to add it
    n_idio_states            = nm + nk + ny - 3 # subtract 3 for dof
    dof_to_remove            = 3
    endo[:marginal_pdf_m′_t] = 1:(nm - 1)
    endo[:marginal_pdf_k′_t] = (1 + nm - 1):(nm + nk - 2)
    endo[:marginal_pdf_y′_t] = (1 + nm + nk - 2):n_idio_states
    n_distr_states           = n_idio_states + get_setting(m, :n_copula_dct_coefficients)
    endo[:copula′_t]         = (1 + n_idio_states):n_distr_states
    for (i, k) in enumerate(get_aggregate_state_variables(m))
        endo[k] = (n_distr_states + i):(n_distr_states + i)
        aggr_endo[k] = i
    end

    # Update n_states to be consistent with the number of
    # idiosyncratic states and jumps after reduction
    n_states      = first(endo[state_vars[end]])
    n_aggr_states = n_states - n_distr_states
    m            <= Setting(:n_backward_looking_states, n_states)

    # Now populate jump indices
    n_dct_Vm            = length(get_setting(m, :dct_compression_indices)[:Vm])
    n_dct_Vk            = length(get_setting(m, :dct_compression_indices)[:Vk])
    n_idio_jumps        = n_dct_Vm + n_dct_Vk
    n_states_idio_jumps = n_states + n_idio_jumps

    endo[:Vm′_t] = (n_states + 1):(n_states + n_dct_Vm)
    endo[:Vk′_t] = (n_states + n_dct_Vm + 1):(n_states + n_idio_jumps)
    for (i, k) in enumerate(get_aggregate_jump_variables(m))
        endo[k] = (n_states_idio_jumps + i):(n_states_idio_jumps + i)
        aggr_endo[k] = i + n_aggr_states
    end
    m <= Setting(:n_model_states, first(endo[jump_vars[end]]))
    m <= Setting(:n_jumps, get_setting(m, :n_model_states) - n_states)

    ## Populate equation indices

    # Function blocks which output a function
    eqconds[:eq_marginal_pdf_m] = endo[:marginal_pdf_m′_t] # note that since endo holds UnitRanges, this assignment results in a copy
    eqconds[:eq_marginal_pdf_k] = endo[:marginal_pdf_k′_t]
    eqconds[:eq_marginal_pdf_y] = endo[:marginal_pdf_y′_t]
    eqconds[:eq_copula]         = endo[:copula′_t]
#=    eqconds[:eq_marginal_value_bonds]   = (first(endo[:Vm′_t]) - n_aggr_states):(last(endo[:Vm′_t]) - n_aggr_states)
    eqconds[:eq_marginal_value_capital] = (first(endo[:Vk′_t]) - n_aggr_states):(last(endo[:Vk′_t]) - n_aggr_states)=#
    eqconds[:eq_marginal_value_bonds]   = endo[:Vm′_t]
    eqconds[:eq_marginal_value_capital] = endo[:Vk′_t]

    # Aggregate blocks
    for (i, name) in enumerate([# Exogenous shocks
                                :eq_A, :eq_Z, :eq_Ψ, :eq_mp, :eq_μ_p, :eq_μ_w,
                                :eq_σ, :eq_union_retained, :eq_retained,

                                :eq_LY, :eq_LB, :eq_LT, :eq_LI, :eq_Lw,
                                :eq_Lq, :eq_LC, :eq_Lavg_tax_rate,
                                :eq_Lτ_prog,

                                :eq_G, :eq_P, :eq_R, :eq_S])
        eqconds[name] = (n_distr_states + i):(n_distr_states + i)
        aggr_eqconds[name] = i
    end

    n_aggr_states = length(aggr_eqconds)

    for (i, name) in enumerate([# Endogenous model states (for the jumps)
                                :eq_capital_return, :eq_wages_firms_pay, :eq_capital_market_clear,
                                :eq_deficit_rule, :eq_real_wage_inflation,
                                :eq_output, :eq_resource_constraint, :eq_tobins_q,
                                :eq_labor_supply, :eq_price_phillips_curve, :eq_wage_phillips_curve,
                                :eq_capital_util, :eq_Ht, :eq_avg_tax_rate,
                                :eq_tax_revenue, :eq_capital_accum,
                                :eq_bond_market_clear, :eq_debt_market_clear,
                                :eq_bond_output_ratio, :eq_tax_output_ratio,
                                :eq_received_wages, :eq_gov_budget_constraint,
                                :eq_tax_level, :eq_tax_progressivity,
                                :eq_Gini_C, :eq_Gini_X, :eq_sd_log_y, :eq_I90_share,
                                :eq_I90_share_net, :eq_W90_share,
                                :eq_Ygrowth, :eq_Bgrowth, :eq_Igrowth,
                                :eq_wgrowth, :eq_Cgrowth, :eq_Tgrowth,
                                :eq_expost_liquidity_premium,
                                :eq_exante_liquidity_premium,
                                :eq_retained_earnings_gdp_ratio,
                                :eq_union_firm_profits, :eq_union_profits,
                                :eq_firm_profits, :eq_profits_distr_to_hh])
        eqconds[name] = (n_states_idio_jumps + i):(n_states_idio_jumps + i)
        aggr_eqconds[name] = i + n_aggr_states
    end

    # Augmented states
    m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states))

#=    # Aggregate blocks
    n_distr_states_idio_jumps = last(eqconds[:eq_marginal_value_capital])
    for (i, name) in enumerate([# Endogenous model states (for the jumps)
                                :eq_mp, :eq_tax_progressivity, :eq_tax_level,
                                :eq_tax_revenue, :eq_avg_tax_rate,
                                :eq_deficit_rule, :eq_gov_budget_constraint,
                                :eq_price_phillips_curve, :eq_wage_phillips_curve,
                                :eq_real_wage_inflation, :eq_capital_util, :eq_capital_return,
                                :eq_received_wages, :eq_wages_firms_pay, :eq_union_firm_profits,
                                :eq_union_profits, :eq_union_retained,
                                :eq_firm_profits, :eq_profits_distr_to_hh, :eq_retained,
                                :eq_tobins_q, :eq_expost_liquidity_premium,
                                :eq_exante_liquidity_premium, :eq_capital_accum,
                                :eq_labor_supply, :eq_output, :eq_resource_constraint,

                                # Growth rates
                                :eq_Ygrowth, :eq_Tgrowth, :eq_Bgrowth,
                                :eq_Igrowth, :eq_wgrowth, :eq_Cgrowth,

                                # Lagged variables
                                :eq_LY, :eq_LB, :eq_LI, :eq_Lw, :eq_LT,
                                :eq_Lq, :eq_LC, :eq_Lavg_tax_rate,
                                :eq_Lτ_prog,

                                # Exogenous shocks
                                :eq_A, :eq_Z, :eq_Ψ, :eq_μ_p, :eq_μ_w,
                                :eq_σ, :eq_G, :eq_P, :eq_R, :eq_S,

                                # Blocks mapping functions to scalars
                                :eq_capital_market_clear,
                                :eq_debt_market_clear, :eq_bond_market_clear,
                                :eq_bond_output_ratio, :eq_tax_output_ratio,
                                :eq_retained_earnings_gdp_ratio, :eq_Ht,
                                :eq_GiniX, :eq_I90_share, :eq_I90_share_net,
                                :eq_W90_share, :eq_sd_log_y, :eq_GiniC])
        eqconds[name] = (n_distr_states_idio_jumps + i):(n_distr_states_idio_jumps + i)
        aggr_eqconds[name] = i
    end=#
end
