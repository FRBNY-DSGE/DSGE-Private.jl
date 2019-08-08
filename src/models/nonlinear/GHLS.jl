"""
GHLS{T} <: AbstractModel{T}

The `GHLS` type defines the structure of the linearized version of the non-linear model in Gust et. al (2017).

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

* `spec::String`: The model specification identifier, \"smets_wouters\",
cached here for filepath computation.

* `subspec::String`: The model subspecification number,
indicating that some parameters from the original model spec (\"ss0\")
are initialized differently. Cached here for filepath computation.

* `settings::Dict{Symbol,Setting}`: Settings/flags that affect
computation without changing the economic or mathematical setup of
the model.

* `test_settings::Dict{Symbol,Setting}`: Settings/flags for testing mode

#### Other Fields

* `rng::MersenneTwister`: Random number generator. Can be seeded to ensure
  reproducibility in algorithms that involve randomness (such as
  Metropolis-Hastings).

* `testing::Bool`: Indicates whether the model is in testing mode. If `true`,
  settings from `m.test_settings` are used in place of those in `m.settings`.

* `observable_mappings::OrderedDict{Symbol,Observable}`: A dictionary that
  stores data sources, series mnemonics, and transformations to/from model
  units. DSGE.jl will fetch data from the Federal Reserve Bank of St. Louis's
  FRED database; all other data must be downloaded by the user. See `load_data`
  and `Observable` for further details.

* `pseudo_observable_mappings::OrderedDict{Symbol,PseudoObservable}`: A
  dictionary that stores names and transformations to/from model units. See
  `PseudoObservable` for further details.
"""
mutable struct GHLS{T} <: AbstractModel{T}
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

    approx::Approximation
end

description(m::GHLS) = "GHLS Model"

"""
`init_model_indices!(m::GHLS)`

Arguments:
`m:: GHLS`: a model object

Description:
Initializes indices for all of `m`'s states, shocks, and equilibrium conditions.
"""
function init_model_indices!(m::GHLS)
    # Endogenous states, note ztil_t is techshock
    endogenous_states = [[
        :k_t, :c_t, :i_t, :w_t, :rm_t, :π_t, :y_t, :x_t, :R_t, :λc, :qk_t, :L_t, :u_t, :mc_t, :rk_t, :muc_t, :Vi_t, :Vp_t, :Vw_t, :π_w, :bc_t, :bi_t, :b_t, :μ_t, :ztil_t, :mon_t, :g_t, :elast_t, :elastw_t, :unk_t, :y_t1, :c_t1, :i_t1, :w_t1, :Ei_t, :Erk_t, :Ec_t, :EVw_t, :Etechshock_t, :Eπ_t, :Eqk_t, :Eλ_c];
        [Symbol("rm_tl$i") for i = 1:n_anticipated_shocks(m)]]

    # Exogenous shocks
    exogenous_shocks = [[
        :b_sh, :μ_sh, :ztil_sh, :rm_sh, :g_sh, :elast_sh, :elastw_sh, :unk_sh];
        [Symbol("rm_shl$i") for i = 1:n_anticipated_shocks(m)]]

    # Expectations shocks - WHAT ARE THESE?
    expected_shocks = [:Ei_sh, :Erk_sh, :Ec_sh, :EVw_sh, :Etechshock_sh, :Eπ_sh, :Eqk_sh, :Eλc_sh]

    # Equilibrium conditions
    equilibrium_conditions = [[
        :eq_capval, :eq_euler, :eq_inv, :eq_wage, :eq_mp, :eq_phlps, :eq_output, :eq_outgap, :eq_mpnom, :eq_λc, :eq_tobq, :eq_L, :eq_caputil,:eq_mcost, :eq_capsrv, :eq_muc, :eq_vi, :eq_vp, :eq_vw, :eq_π_w, :eq_bc, :eq_bi, :eq_laggdp, :eq_lagcc, :eq_lagit, :eq_lagwage, :eq_b, :eq_μ, :eq_ztil, :eq_mon, :eq_g, :eq_elast, :eq_elastw, :eq_unk, :eq_Ei, :eq_Erk, :eq_Ec, :eq_EVw, :eq_Ez, :eq_Eπ, :eq_Eqk, :eq_Eλc];
        [Symbol("eq_rml$i") for i=1:n_anticipated_shocks(m)]]

    # Additional states added after solving model
    # Lagged states and observables measurement error - WHAT ARE THESE?
    endogenous_states_augmented = [
        :y_t1, :c_t1, :i_t1]

    # Observables
    observables = keys(m.observable_mappings)

    # Pseudo-observables
    pseudo_observables = keys(m.pseudo_observable_mappings)

    for (i,k) in enumerate(endogenous_states);           m.endogenous_states[k]           = i end
    for (i,k) in enumerate(exogenous_shocks);            m.exogenous_shocks[k]            = i end
    for (i,k) in enumerate(expected_shocks);             m.expected_shocks[k]             = i end
    for (i,k) in enumerate(equilibrium_conditions);      m.equilibrium_conditions[k]      = i end
    for (i,k) in enumerate(endogenous_states);           m.endogenous_states[k]           = i end
    for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i+length(endogenous_states) - 14 end #THIS IS NOT A GREAT WAY TO DO THIS
    for (i,k) in enumerate(observables);                 m.observables[k]                 = i end
    for (i,k) in enumerate(pseudo_observables);          m.pseudo_observables[k]          = i end
end


function GHLS(subspec::String="ss0";
                      custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
                      testing = false, nonlin = false)

    # Model-specific specifications
    spec               = split(basename(@__FILE__),'.')[1]
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = GHLS{Float64}(
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
            OrderedDict{Symbol,PseudoObservable}(),
            Approximation())

    # Set settings
    model_settings!(m)
    default_test_settings!(m)
    for custom_setting in values(custom_settings)
        m <= custom_setting
    end

    # Set observable and pseudo-observable transformations
    init_observable_mappings!(m)

    # Initialize parameters
    init_parameters!(m)

    init_model_indices!(m)
    init_subspec!(m)
    steadystate!(m)

    return m
end

"""
```
init_parameters!(m::GHLS)
```

Initializes the model's parameters, as well as empty values for the steady-state
parameters (in preparation for `steadystate!(m)` being called to initialize
those).
"""
function init_parameters!(m::GHLS)

   #Fixed parameters
   m <= parameter(:δ, 0.025, fixed=true,
                   description="δ: The capital depreciation rate.",
                   tex_label="\\delta")

   m <= parameter(:shrgy, 0.2, fixed=true,
                   description="shrgy: Government spending as share of output.",
                   tex_label="\\shrgy")

   m <= parameter(:ϵ_p, 0.2, fixed=true, scaling=x -> 1/x + 1,
                   description="ϵ_p: Steady state net price markup.",
                   tex_label="\\epsilon_p")

   m <= parameter(:ϵ_w, 0.2, fixed=true, scaling=x -> 1/x + 1,
                   description="ϵ_w: Steady state net wage markup.",
                   tex_label="\\epsilon_w")

   m <= parameter(:ψ_L, 1., fixed=true,
                   description="ψ_L: The disutility of labor.",
                   tex_label="\\psi_L")

   m <= parameter(:ρ_η, 0.85, fixed=true,
                   description="ρ_η: Persistence of liquidity shock.",
                   tex_label="\\rho_{\\eta}")

   m <= parameter(:ρ_int, 0., fixed=true,
                   description="ρ_int: Persistence of interest rate shock.",
                   tex_label="\\rho_int")

   m <= parameter(:ρ_elast, 0., fixed=true,
                   description="ρ_elast: Persistence of price elasticity shock.",
                   tex_label="\\rho_elast")

   m <= parameter(:ρ_elastw, 0., fixed=true,
                   description="ρ_elastw: Persistence of wage elasticity shock.",
                   tex_label="\\rho_elastw")

   m <= parameter(:σ_elast, 1e-5, fixed=true,
                   description="σ_elast: Standard deviation of price elasticity shock.",
                   tex_label="\\sigma_elast")

   m <= parameter(:σ_elastw, 1e-5, fixed=true,
                   description="σ_elastw: Standard deviation of wage elasticity shock.",
                   tex_label="\\sigma_elastw")

    #Steady state parameters
    m <= parameter(:α, 0.24, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(), Normal(0.30, 0.05), fixed=false,
                   description="α: Capital elasticity in the intermediate goods sector's Cobb-Douglas production function.",
                   tex_label="\\alpha")

    m <= parameter(:β, 0.7420, (1e-5, 10.), (1e-5, 10.), Exponential(), GammaAlt(0.25, 0.1), fixed=false, scaling = x -> 1/(1 + x/100),
                   description="β: Discount rate.",
                   tex_label="\\beta")

    m <= parameter(:π_bar, 0.7000, (1e-5, 10.), (1e-5, 10.), Exponential(), Normal(.62, .1), fixed=false, scaling = x -> 1 + x/100,
                   description="π_bar: The targeted rate of inflation.",
                   tex_label="\\pi_bar")

    m <= parameter(:gz, 0.3982, (-5.0, 5.0), (-5., 5.), Untransformed(), Normal(0.5, 0.03), fixed=false, scaling = x -> exp(x/100),
                   description="gz: The deterministic growth rate of technology.",
                   tex_label="gz")

    # Monetary Policy Rule Parameters
    m <= parameter(:γ_π, 1.7985, (1e-5, 10.), (1e-5, 10.00), Exponential(), Normal(1.7, 0.3), fixed=false,
                   description="γ_π: Weight on inflation gap in monetary policy rule.",
                   tex_label="\\gamma_{\\pi}")

    m <= parameter(:γ_x, 0.0893, (1e-5, 5.), (1e-5, 5.), Exponential(), Normal(0.4, 0.3), fixed=false,
                   description="γ_x: Weight on output gap in monetary policy rule.",
                   tex_label="\\gamma_x") #Different from Smets Wouters output gap

    m <= parameter(:γ_g, 0.2239, (1e-5, 5.), (1e-5, 5.), Exponential(), Normal(0.4, 0.3), fixed=false,
                   description="γ_g: Weight on output growth in the monetary policy rule.",
                   tex_label="\\γ_g") #Is this correct?

    m <= parameter(:ρ_R, 0.3000, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(), BetaAlt(0.6, 0.2), fixed=false,
                   description="ρ_R: Coefficient on past interest rate in the monetary policy shock process.",
                   tex_label="\\rho_{R}")

    #Endogenous Propogation Parameters
    m <= parameter(:γ, 0.7205, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(), BetaAlt(0.6, 0.1), fixed=false,
                   description="γ: Consumption habit persistence.",
                   tex_label="γ")

    m <= parameter(:σ_a, 5., (1e-5, 100.), (1e-5, 100.), Exponential(), GammaAlt(5., 1.), fixed=false,
                   description="σ_a: Capital utilization costs.",
                   tex_label="\\sigma_a")

    m <= parameter(:σ_L, 1.7985, (1e-5, 10.), (1e-5, 10.00), Exponential(), GammaAlt(2., 0.75), fixed=false,
                   description="σ_L: Affects  Frisch elasticity of labor supply.",
                   tex_label="\\sigma_L")

    m <= parameter(:ϕ_w, 3000.00, (-5e5, 5e5), (-5e5, 5e5), Untransformed(), Normal(3000.00, 5000.00), fixed=false,
                   description="ϕ_w: Wage adjustment cost.",
                   tex_label="\\phi_w")

    m <= parameter(:ϕ_p, 100.00, (-5e3, 5e3), (-5e3, 5e3), Untransformed(), Normal(100.00, 25.00), fixed=false,
                   description="ϕ_p: Price adjustment cost.",
                   tex_label="\\phi_p")

    m <= parameter(:ϕ_I, 4., (1e-5, 100.), (1e-5, 100.), Exponential(), GammaAlt(4., 1.), fixed=false,
                   description="ϕ_I: Investment adjustment cost.",
                   tex_label="\\phi_I")

    m <= parameter(:ap, 0.8258, (1e-5, .999), (1e-5, .999), SquareRoot(), BetaAlt(.5, .15), fixed=false, scaling = x -> 1 - x,
                   description="ap: Determines extent to which price indexation is tied to inflation target versus lagged inflation rate.",
                   tex_label="ap")

    m <= parameter(:aw, 0.8258, (1e-5, .999), (1e-5, .999), SquareRoot(), BetaAlt(.5, .15), fixed=false, scaling = x -> 1 - x,
                   description="aw: Determines extent to which wage indexation is tied to inflation target versus lagged inflation rate.",
                   tex_label="aw")

    # Exogenous Process Parameters - Autocorrelations
    m <= parameter(:ρ_g, 0.9930, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(), BetaAlt(0.6, 0.2), fixed=false,
                   description="ρ_g: AR(1) coefficient in the government spending process.",
                   tex_label="\\rho_g")

    m <= parameter(:ρ_μ, 0.5724, (1e-5, 0.999), (1e-5, 0.999), SquareRoot(), BetaAlt(0.6, 0.2), fixed=false,
                   description="ρ_μ: AR(1) coefficient in marginal efficiency of investment process.",
                   tex_label="\\rho_{\\mu}")

    #Exogenous Process Parameters - Standard Deviation
    m <= parameter(:σ_g, 0.6090, (1e-8, 5.), (1e-8, 5.), Exponential(), InverseGamma(2., 0.33), fixed=false, scaling = x -> x/100,
                   description="σ_g: The standard deviation of the government spending process.",
                   tex_label="\\sigma_{g}")

    m <= parameter(:σ_η, 0.1818, (1e-8, 5.), (1e-8, 5.), Exponential(), InverseGamma(2., 0.33), fixed=false, scaling = x -> x/100,
                   description="σ_η: The standard deviation of the return on the risk-free bond process.",
                   tex_label="\\sigma_{\\eta}")

    m <= parameter(:σ_μ, 0.4601, (1e-8, 5.), (1e-8, 5.), Exponential(), InverseGamma(2., 0.33), fixed=false, scaling = x -> x/100,
                   description="σ_μ: The standard deviation of the exogenous marginal efficiency of investment shock process.",
                   tex_label="\\sigma_{\\mu}")

    m <= parameter(:σ_Z, 0.4618, (1e-8, 5.), (1e-8, 5.), Exponential(), InverseGamma(2., 0.33), fixed=false, scaling = x -> x/100,
                   description="σ_Z: The standard deviation of the disturbance to technological growth.",
                   tex_label="\\sigma_{Z}")

    m <= parameter(:σ_R, 0.2397, (1e-8, 5.), (1e-8, 5.), Exponential(), InverseGamma(2., 0.33), fixed=false, scaling = x -> x/100,
               description="σ_R: The standard deviation of the monetary policy shock.",
               tex_label="\\sigma_{R}")


    #Measurement errors
    m <= parameter(:e_y, 0.00323357^2, fixed = true, description = "e_y: Measurement error on GDP", tex_label = "e_y")
    m <= parameter(:e_π, 0.00122426^2, fixed = true, description = "e_π: Measurement error on GDP deflator", tex_label = "e_π")
    m <= parameter(:e_R, 0.00358562^2, fixed = true, description = "e_R: Measurement error on nominal rate of interest", tex_label = "e_R")
    m <= parameter(:e_c, 0.00256811^2, fixed = true, description = "e_c: Measurement error on consumption", tex_label = "e_c")
    m <= parameter(:e_i, 0.01218603^2, fixed = true, description = "e_i: Measurement error on investment", tex_label = "e_i")


    # Steady states
    m <= SteadyStateParameter(:cap, NaN, description="Steady-state capital", tex_label="x")
    m <= SteadyStateParameter(:cc, NaN, description="Steady-state consumption", tex_label="x")
    m <= SteadyStateParameter(:inv, NaN, description="Steady-state investment", tex_label="x")
    m <= SteadyStateParameter(:rw, NaN, description="Steady-state real wage", tex_label="x")
    m <= SteadyStateParameter(:notr, NaN, description="Steady-state notional interest rate", tex_label="x")
    m <= SteadyStateParameter(:dp, NaN, description="Steady-state inflation", tex_label="x")
    m <= SteadyStateParameter(:gdp, NaN, description="Steady-state output", tex_label="x")
    m <= SteadyStateParameter(:xhp, NaN, description="Steady-state output gap", tex_label="x")
    m <= SteadyStateParameter(:nomr, NaN, description="Steady-state nominal interest rate", tex_label="g")
    m <= SteadyStateParameter(:lam, NaN, description="Steady-state marginal value of consumption", tex_label="g")
    m <= SteadyStateParameter(:qq, NaN, description="Steady-state Tobin's q", tex_label="g")
    m <= SteadyStateParameter(:lab, NaN, description="Steady-state number of hours worked", tex_label="L")
    m <= SteadyStateParameter(:util, NaN, description="Steady-state capital utilization cost", tex_label="g")
    m <= SteadyStateParameter(:mc, NaN, description="Steady-state marginal cost", tex_label="g")
    m <= SteadyStateParameter(:rentalk, NaN, description="Steady-state rental rate of capital", tex_label="g")
    m <= SteadyStateParameter(:muc, NaN, description="Steady-state marginal utility of consumption", tex_label="g")
    m <= SteadyStateParameter(:vi, NaN, description="Steady-state investment v", tex_label="g")
    m <= SteadyStateParameter(:vp, NaN, description="Steady-state inflation v", tex_label="g")
    m <= SteadyStateParameter(:vw, NaN, description="Steady-state wage v", tex_label="g")
    m <= SteadyStateParameter(:dw, NaN, description="Steady-state wage inflation", tex_label="g")
    m <= SteadyStateParameter(:bc, NaN, description="Steady-state bc", tex_label="g")
    m <= SteadyStateParameter(:bi, NaN, description="Steady-state bi", tex_label="g")
    m <= SteadyStateParameter(:liqshk, NaN, description="Steady-state liquidity shock", tex_label="g")
    m <= SteadyStateParameter(:invshk, NaN, description="Steady-state investment shock", tex_label="g")
    m <= SteadyStateParameter(:techshk, NaN, description="Steady-state technology shock", tex_label="g")
    m <= SteadyStateParameter(:intshk, NaN, description="Steady-state interest rate shock", tex_label="g")
    m <= SteadyStateParameter(:gshk, NaN, description="Steady-state government shock", tex_label="g")
    m <= SteadyStateParameter(:ashk, NaN, description="Steady-state a shock", tex_label="g")
    m <= SteadyStateParameter(:gg, NaN, description="Steady-state government spending", tex_label="g")
    m <= SteadyStateParameter(:gamtil, NaN, tex_label="gamtil")
    m <= SteadyStateParameter(:mc, NaN, description="Steady-state marginal cost of capital", tex_label="mc")
    m <= SteadyStateParameter(:k2yrat, NaN, description="Steady-state capital to output ratio", tex_label="\\frac{\\bar{k}}{y}")
    m <= SteadyStateParameter(:shriy, NaN, description="Steady-state investment to output ratio", tex_label="\\frac{i}{y}")
    m <= SteadyStateParameter(:shrcy, NaN, description="Steady-state consumption to output ratio", tex_label="\\frac{c}{y}")
    m <= SteadyStateParameter(:labss, NaN, description="Steady-state hours", tex_label="N")
    m <= SteadyStateParameter(:κ_w, NaN, tex_label="\\kappa_{w}")
    m <= SteadyStateParameter(:κ_p, NaN, tex_label="\\kappa_{p}")
    m <= SteadyStateParameter(:kss, NaN, description="Steady-state capital", tex_label="\\bar{\\kappa}")
    m <= SteadyStateParameter(:gdpss, NaN, description="Steady-state output", tex_label="y")
    m <= SteadyStateParameter(:invss, NaN, description="Steady-state investment", tex_label="i")
    m <= SteadyStateParameter(:phii_jpt, NaN, tex_label="phii_jpt")
    m <= SteadyStateParameter(:css, NaN, description="Steady-state consumption", tex_label="c")
    m <= SteadyStateParameter(:rwss, NaN, description="Steady-state real wage", tex_label="rw")
    m <= SteadyStateParameter(:mucss, NaN, description="Steady-state marginal utility of consumption", tex_label="muc")
    m <= SteadyStateParameter(:lamss, NaN, description="Steady-state lagrange multiplier on  consumption", tex_label="\\lambda")
    m <= SteadyStateParameter(:rss, NaN, description="Steady-state interest rate", tex_label="R")
    m <= SteadyStateParameter(:rkss, NaN, description="Steady-state rental rate of capital", tex_label="rk")
    m <= SteadyStateParameter(:bw, NaN, tex_label="bw")
end

"""
```
steadystate!(m::GHLS)
```

Calculates the model's steady-state values. `steadystate!(m)` must be called whenever the parameters of `m` are updated.
"""
function steadystate!(m::GHLS)
    m[:gg] = 1.0/(1.0-m[:shrgy])
    m[:gamtil] = m[:γ]/m[:gz]
    m[:mc] = (m[:ϵ_p]-1.0)/m[:ϵ_p]
    m[:k2yrat] = ((m[:mc]*m[:α])/(m[:gz]/m[:β]-(1.0-m[:δ])))*m[:gz]
    m[:shriy] = (1.0 - (1.0-m[:δ])/m[:gz])*m[:k2yrat]
    m[:shrcy] = (1.0 - m[:shrgy] - m[:shriy])
    m[:labss] = (((m[:ϵ_w] - 1.0)/m[:ϵ_w])*(1.0-m[:α])*(1.0 - m[:β]*m[:gamtil])*((m[:ϵ_p]-1.0)/m[:ϵ_p])*(1.0/(m[:ψ_L]*(1.0 - m[:gamtil])))*(1.0/m[:shrcy]))^(1.0/(m[:σ_L]+1))
    m[:κ_w] = ((1.0-m[:gamtil])/(1.0-m[:β]*m[:gamtil]))*m[:ϵ_w]*m[:ψ_L]*m[:labss]^(1.0+m[:σ_L])/m[:ϕ_w]
    m[:κ_p] = (m[:ϵ_p]-1.0)/(m[:ϕ_p]*(1.0+m[:β]*(1-m[:ap])))
    m[:kss] = m[:labss]*(m[:gz]^(m[:α]/(m[:α]-1.0)))*m[:k2yrat]^(1.0/(1.0-m[:α]))
    m[:gdpss] = (m[:kss]/m[:gz])^m[:α]*m[:labss]^(1.0-m[:α])
    m[:invss]  = m[:shriy]*m[:gdpss]
    m[:phii_jpt] = m[:ϕ_I]/m[:invss]
    m[:css] = m[:shrcy]*m[:gdpss]
    m[:rwss] = (1.0 - m[:α])*m[:mc]*m[:gdpss]/m[:labss]
    m[:mucss] = (1.0/m[:css])*(1.0/(1.0 - m[:gamtil]))
    m[:lamss] = m[:mucss]*(1.0 - m[:β]*m[:gamtil])
    m[:rss] = m[:gz]*m[:π_bar]/m[:β]
    m[:rkss] = m[:gz]/m[:β] - 1.0 + m[:δ]
    m[:bw]  =  m[:aw]

    #Start of real steady states
    m[:cap] = log(m[:kss])
    m[:cc] = log(m[:css])
    m[:inv] = log(m[:invss])
    m[:rw] = log(m[:rwss])
    m[:notr] = log(m[:rss])
    m[:dp] = log(m[:π_bar])
    m[:gdp] = log(m[:gdpss])
    m[:xhp] = 0.0
    m[:nomr] = log(m[:rss])
    m[:lam] = log(m[:lamss])
    m[:qq] = 0.0
    m[:lab] = log(m[:labss])
    m[:util] = 0.0
    m[:mc] = log(m[:mc])
    m[:rentalk] = log(m[:rkss])
    m[:muc] = log(m[:mucss])
    m[:vi] = 0.0
    m[:vp] = 0.0
    m[:vw] = 0.0
    m[:dw] = log(m[:π_bar]*m[:gz])
    m[:bc] = log(m[:mucss])
    m[:bi] = 0.0
    m[:liqshk] = 0.0
    m[:invshk] = 0.0
    m[:techshk] = 0.0
    m[:intshk] = 0.0
    m[:gshk] = log(m[:gg])
    m[:ashk] = 0.0
    return m

end

function model_settings!(m::GHLS)

    default_settings!(m)

    # Anticipated shocks
    m <= Setting(:n_anticipated_shocks, 0)
    m <= Setting(:n_anticipated_shocks_padding, 20)

    # Estimation
    m <= Setting(:reoptimize, true)
    m <= Setting(:recalculate_hessian, true)

    # Data
    m <= Setting(:data_vintage, "190712")
    m <= Setting(:cond_vintage, "190719")
    m <= Setting(:data_id, 1, "Dataset identifier")
    m <= Setting(:cond_full_names, [:obs_gdp, :obs_gdpdeflator, :obs_nominalrate, :obs_consumption, :obs_investment])
    m <= Setting(:cond_semi_names, [:obs_nominalrate])
    m <= Setting(:date_mainsample_start, quartertodate("1983-Q1"))
    m <= Setting(:date_presample_start, quartertodate("1983-Q1"))
    m <= Setting(:date_mainsample_end, quartertodate("2014-Q1"))
    m <= Setting(:date_conditional_end, quartertodate("2014-Q1"))
    m <= Setting(:date_forecast_start, quartertodate("2014-Q1"))

    m <= Setting(:n_particles, 100)
    m <= Setting(:sampling_method, :SMC)

    m <= Setting(:zero_lower_bound, true)
end

"""
```
shock_groupings(m::GHLS)
```

Returns a `Vector{ShockGroup}`, which must be passed in to
`plot_shock_decomposition`. See `?ShockGroup` for details.
"""
function shock_groupings(m::GHLS)
    gov = ShockGroup("g", [:g_sh], RGB(0.70, 0.13, 0.13)) # firebrick
    bet = ShockGroup("b", [:b_sh], RGB(0.3, 0.3, 1.0))
    tfp = ShockGroup("z", [:ztil_sh], RGB(1.0, 0.55, 0.0)) # darkorange
    rm = ShockGroup("rm", [:rm_sh], RGB(0.60, 0.80, 0.20)) # yellowgreen
    elast = ShockGroup("elast_sh", [:elast_sh], RGB(0.0, 0.5, 0.5)) # teal
    elastw = ShockGroup("elastw", vcat([:elastw_sh], [Symbol("rm_shl$i") for i = 1:n_anticipated_shocks(m)]),
                     RGB(1.0, 0.84, 0.0)) # gold
    mei = ShockGroup("mu", [:μ_sh], :cyan)
    unk = ShockGroup("unk", [:unk_sh], :gray40)

    return [gov, bet, tfp, rm, elast, elastw, mei, unk]
end
