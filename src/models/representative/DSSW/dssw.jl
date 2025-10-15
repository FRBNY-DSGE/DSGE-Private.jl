"""
DSSW{T} <: AbstractRepModel{T}

The `DSSW` type defines the structure of the New York Fed DSGE
model.

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

* `rng::MersenneTwister`: Random number generator. Can be is seeded to ensure
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
mutable struct DSSW{T} <: AbstractRepModel{T}
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

description(m::DSSW) = "Del Negro, Schorfheide, Smets, and Wouters 2007 Model"

"""
`init_model_indices!(m::DSSW)`

Arguments:
`m:: DSSW`: a model object

Description:
Initializes indices for all of `m`'s states, shocks, and equilibrium conditions.
"""
function init_model_indices!(m::DSSW)
    # Endogenous states
    #=
    endogenous_states = [[
        :y_t, :c_t, :i_t, :qk_t, :k_t, :kbar_t, :u_t, :rk_t, :mc_t,
        :π_t, :μ_ω_t, :w_t, :L_t, :R_t, :g_t, :b_t, :μ_t, :z_t,
        :λ_f_t, :λ_f_t1, :λ_w_t, :λ_w_t1, :rm_t, :Ec_t, :Eqk_t, :Ei_t,
        :Eπ_t, :EL_t, :Erk_t, :Ew_t, :y_f_t, :c_f_t,
        :i_f_t, :qk_f_t, :k_f_t, :kbar_f_t, :u_f_t, :rk_f_t, :w_f_t,
        :L_f_t, :r_f_t, :Ec_f_t, :Eqk_f_t, :Ei_f_t, :EL_f_t, :Erk_f_t, :ztil_t];
                         [Symbol("rm_tl$i") for i = 1:n_anticipated_shocks(m)]]
    =#

    endogenous_states = [[
        :y_t, :c_t, :i_t, :k_t, :kbar_t, :L_t, :m_t, :mc_t, :pi_t, :R_t, :rk_t,
        :u_t, :w_t, :wtil_t, :xi_t, :xik_t]; [:z_t, :phi_t, :mu_t, :b_t, :g_t, :laf_t];  #6 out 7 shocks are states since r_sh is iid while other shocks are processes
                         [:E_c, :E_pi, :E_rk, :E_w, :E_wtil, :E_xi, :E_xik, :E_i]] #Expectational errors are also states in the model

    # Exogenous shocks
    exogenous_shocks = [:z_sh, :phi_sh, :mu_sh, :b_sh, :g_sh, :laf_sh, :r_sh]

    # Expectations shocks
    expected_shocks = [:Ec_sh, :Epi_sh, :Erk_sh, :Ew_sh, :Ewtil_sh, :Exi_sh, :Exik_sh, :Ei_sh]

    # Equilibrium conditions
    equilibrium_conditions = [[:eq_margcost, :eq_prsett, :eq_capacc, :eq_effcap, :eq_euler, :eq_moneydem, :eq_margut, :eq_invfoc, :eq_rettocap, :eq_utcap, :eq_optwage, :eq_aggwage, :eq_caplabrat, :eq_resources, :eq_prod, :eq_taylor];[:eq_z, :eq_phi, :eq_mu, :eq_b, :eq_g, :eq_laf]; [:eq_Ec, :eq_Epi, :eq_Erk, :eq_Ew, :eq_Ewtil, :eq_Exi, :eq_Exik, :eq_Ei]]

    # Additional states added after solving model
    # Lagged states and observables measurement error
    endogenous_states_augmented = [:y_t1, :c_t1, :i_t1, :w_t1, :m_t1]

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


function DSSW(subspec::String="ss0";
                      custom_settings::Array{S} where S<:Setting = Array{Setting{Bool}}(undef, 0),
                      testing = false)

    # Model-specific specifications
    spec               = split(basename(@__FILE__),'.')[1]
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = DSSW{Float64}(
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
    default_test_settings!(m)
    for custom_setting in custom_settings
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
init_parameters!(m::DSSW)
```

Initializes the model's parameters, as well as empty values for the steady-state
parameters (in preparation for `steadystate!(m)` being called to initialize
those).
"""
function init_parameters!(m::DSSW)
    m <= parameter(:alp, 0.33, (1e-5, 0.99999), (1e-5, 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.33, 0.1), fixed=false,
                   description="α: Capital elasticity in the intermediate goods sector's Cobb-Douglas production function.",
                   tex_label="\\alpha")

    m <= parameter(:zeta_p, 0.75, (1e-5, 0.99999), (1e-5, 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.6, 0.2), fixed=false,
                   description="ζ_p: The Calvo parameter. In every period, intermediate goods producers optimize prices with probability (1-ζ_p). With probability ζ_p, prices are adjusted according to a weighted average of the previous period's inflation (π_t1) and steady-state inflation (π_star).",
                   tex_label="\\zeta_p")

    m <= parameter(:iota_p, 0., (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.28), fixed=true,
                   description="ι_p: The weight attributed to last period's inflation in price indexation. (1-ι_p) is the weight attributed to steady-state inflation.",
                   tex_label="\\iota_p")

    m <= parameter(:del, 0.025, fixed=true,
                   description="δ: The capital depreciation rate.",
                   tex_label="\\delta" )
    #NOT DEFINED IN PAPER
    m <= parameter(:ups, 0., (0., 10.), (0., 10.), ModelConstructors.Exponential(), GammaAlt(1., 0.5), fixed=true, scaling = x -> exp(x/100),
                   description="Υ: The trend evolution of the price of investment goods relative to consumption goods. Set equal to 1.",
                   tex_label="\\Upsilon")
    #NOTE DEFINED IN PAPER
    m <= parameter(:Bigphi, 0., (0., 5.), (0., 5.), ModelConstructors.Exponential(), GammaAlt(0.5, 0.25), fixed=false,
                   description="Φ: Fixed costs.",
                   tex_label="\\Phi")

    m <= parameter(:s2, 4., (0., 20.), (0., 20.), ModelConstructors.Untransformed(), GammaAlt(4.0, 1.5), fixed=false,
                   description="S'': The second derivative of households' cost of adjusting investment.",
                   tex_label="S''")

    m <= parameter(:h, 0.7, (1e-5, 0.99999), (1e-5, 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.7, 0.05), fixed=false,
                   description="h: Consumption habit persistence.",
                   tex_label="h")

    #change desc
    m <= parameter(:a2, 0.2, (0., 0.99999), (0., 0.99999), ModelConstructors.SquareRoot(), GammaAlt(0.2, 0.1), fixed=false,
                   description="ppsi: Utilization costs.",
                   tex_label="\\psi")

    m <= parameter(:nu_l, 2., (1e-5, 10.), (1e-5, 10.), ModelConstructors.Exponential(), GammaAlt(2.0, 0.75), fixed=false,
                   description="ν_l: The coefficient of relative risk aversion on the labor term of households' utility function.", tex_label="\\nu_l")

    #NOT DEFINED IN PAPER
    m <= parameter(:nu_m, 2., (1e-5, 100.), (1e-5, 100.), ModelConstructors.Exponential(), Normal(2.0, 0.75), fixed=true,                                   description="ν_l: The coefficient of relative risk aversion on the labor term of households' utility function.", tex_label="\\nu_m")

    m <= parameter(:zeta_w, 0.8, (1e-5, 0.99999), (1e-5, 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.6, 0.2), fixed=false,
                   description="ζ_w: (1-ζ_w) is the probability with which households can freely choose wages in each period. With probability ζ_w, wages increase at a geometrically weighted average of the steady state rate of wage increases and last period's productivity times last period's inflation.",
                   tex_label="\\zeta_w")

    m <= parameter(:iota_w, 0., (0., 1.), (0., 1.), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.28), fixed=false,
                   description="ι_w: The weight attributed to last period's wage in wage indexation. (1-ι_w) is the weight attributed to steady-state wages.",
                   tex_label="\\iota_w")

    m <= parameter(:law, 0.3, fixed=true,
                   description="λ_w: The wage markup, which affects the elasticity of substitution between differentiated labor services.",
                   tex_label="\\lambda_w")
    #NOT DEFINED IN PAPER
    m <= parameter(:bet, 2., (1e-5, 10.), (1e-5, 10.), ModelConstructors.Exponential(), GammaAlt(2.0, 1.0), fixed=false, scaling = x -> exp(-x/100),#scaling = x -> 1/(1 + x/100),
                   description="β: Discount rate.",
                   tex_label="\\beta")

    m <= parameter(:ps1, 1.7, (1e-5, 10.), (1e-5, 10.), ModelConstructors.Exponential(), GammaAlt(1.5, 0.4), fixed=false,
                   description="ψ₁: Weight on inflation gap in monetary policy rule.",
                   tex_label="\\psi_1")

    m <= parameter(:psi2, 0.125, (1e-5, 10.), (1e-5, 10.), ModelConstructors.Untransformed(), GammaAlt(0.2, 0.1), fixed=false,
                   description="ψ₂: Weight on output gap in monetary policy rule.",
                   tex_label="\\psi_2")
    #change description!
    m <= parameter(:rho_r, 0.8, (1e-5, 0.99999), (1e-5, 0.99999), ModelConstructors.Untransformed(), BetaAlt(0.5, 0.2), fixed=false,
                   description="ψ₃: Weight on rate of change of output gap in the monetary policy rule.",
                   tex_label="\\rho_r")

    m <= parameter(:pistar, 3., (-1., 10.), (-1., 10.), ModelConstructors.Untransformed(), Normal(3.0, 1.5), fixed=false, scaling = x -> exp(x/100),
                   description="ψ₃: Weight on rate of change of output gap in the monetary policy rule.",
                   tex_label="\\pi_{*}")
    # exogenous processes - level

    #change descr!
    m <= parameter(:gam, 2., (1e-6, 10.), (1e-6, 10.), ModelConstructors.Exponential(), GammaAlt(2.0, 1.0), fixed=false, scaling = x -> x/100,
                   tex_label="\\gamma")
    #NOT DEFINED IN PAPER
    #change desc
    m <= parameter(:wadj, 0., (0., 10.), (0., 10.), ModelConstructors.Exponential(), Normal(0.0, 5.0), fixed=false, tex_label="\\w_{adj}")

    #NOT DEFINED IN PAPER
    #change dec
    m <= parameter(:chi, 0.1, (1e-6, 10.), (1e-6, 10.), ModelConstructors.Exponential(), GammaAlt(0.1, 0.1), fixed=false,
                   tex_label="\\chi")

    #change desc
    m <= parameter(:laf, 0.15, (1e-5, 50.), (1e-5, 50.), ModelConstructors.Exponential(), GammaAlt(0.15, 0.1), fixed=false,
                   tex_label="\\lambda_{f}")

    m <= parameter(:gstar, 0.15, (1e-5, 50.), (1e-5, 50.), ModelConstructors.Exponential(), GammaAlt(0.1, 2.), fixed=false, scaling = x -> 1 + x,
                   tex_label="\\g_{*}")

    #change decr
    m <= parameter(:Ladj, 252.,(1e-5, 5000.), (1e-5, 5000.), ModelConstructors.Untransformed(), Normal(252.0, 10.0), fixed=false, scaling = x -> x/100,
                   description="γ: The log of the steady-state growth rate of technology.",
                   tex_label="\\L_{adj}")

    # exogenous processes - autocorrelation
    m <= parameter(:rho_z, .2, (0., 0.99999), (0., 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.2, 0.1), fixed=false,
                   description="ρ_z: AR(1) coefficient in the technology process.",
                   tex_label="\\rho_z")
    #change desc
    m <= parameter(:rho_phi, 0.4, (1e-5, 0.99999), (1e-5, 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.6, 0.2), fixed=false,
                   description="ρ_g: AR(1) coefficient in the government spending process.",
                   tex_label="\\rho_{\\phi}")

    #NOT DEFINED IN PAPER
    #change desc
    m <= parameter(:rho_chi, 0.9, (1e-5, 0.99999), (1e-5, 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.6, 0.2), fixed=false,
                   description="ρ_g: AR(1) coefficient in the government spending process.",
                   tex_label="\\rho_{\\chi}")

    m <= parameter(:rho_laf, 0.9, (1e-5, 0.99999), (1e-5, 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.6, 0.2), fixed=false,
                   description="ρ_λ_f: AR(1) coefficient in the price mark-up shock process.",
                   tex_label="\\rho_{\\lambda_f}")

    m <= parameter(:rho_mu, 0.5, (1e-5, 0.99999), (1e-5, 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.8, 0.05), fixed=false,
                   description="ρ_μ: AR(1) coefficient in capital adjustment cost process.",
                   tex_label="\\rho_{\\mu}")

    m <= parameter(:rho_b, 0.8, (1e-5, 0.99999), (1e-5, 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.6, 0.2), fixed=false,
                   description="ρ_b: AR(1) coefficient in the intertemporal preference shifter process.",
                   tex_label="\\rho_b")


    m <= parameter(:rho_g, 0.9, (1e-5, 0.99999), (1e-5, 0.99999), ModelConstructors.SquareRoot(), BetaAlt(0.8, 0.05), fixed=false,
                   description="ρ_g: AR(1) coefficient in the government spending process.",
                   tex_label="\\rho_g")
    # exogenous processes - standard deviation

    m <= parameter(:sig_z, 0.4, (1e-7, 100.), (1e-7, 100.), ModelConstructors.Exponential(), RootInverseGamma(2., 0.75), fixed=false,
                   description="σ_z: The standard deviation of the process describing the stationary component of productivity.",
                   tex_label="\\sigma_{z}")

    #change desc
    m <= parameter(:sig_phi, 1., (1e-7, 100.), (1e-7, 100.), ModelConstructors.Exponential(), RootInverseGamma(2., 4.), fixed=false,
                   description="σ_z: The standard deviation of the process describing the stationary component of productivity.",
                   tex_label="\\sigma_{\\phi}")

    m <= parameter(:sig_chi, 0., (1e-7, 10000.), (1e-7, 10000.), ModelConstructors.Exponential(), RootInverseGamma(2., 0.75), fixed=false,
                   description="σ_λ_f: The mean of the process that generates the price elasticity of the composite good. Specifically, the elasticity is (1+λ_{f,t})/(λ_{f_t}).",
                   tex_label="\\sigma_{\\chi}")

    m <= parameter(:sig_laf, 1., (1e-7, 10000.), (1e-7, 10000.), ModelConstructors.Exponential(), RootInverseGamma(2., 0.75), fixed=false,
                   description="σ_λ_f: The mean of the process that generates the price elasticity of the composite good. Specifically, the elasticity is (1+λ_{f,t})/(λ_{f_t}).",
                   tex_label="\\sigma_{\\lambda_f}")

    m <= parameter(:sig_mu, 1., (1e-7, 100.), (1e-7, 100.), ModelConstructors.Exponential(), RootInverseGamma(2., 0.75), fixed=false,
                   description="σ_μ: The standard deviation of the exogenous marginal efficiency of investment shock process.",
                   tex_label="\\sigma_{\\mu}")

    m <= parameter(:sig_b, 0.2, (1e-7, 100.), (1e-7, 100.), ModelConstructors.Exponential(), RootInverseGamma(2., 0.75), fixed=false,
                   description="σ_b: The standard deviation of the intertemporal preference shifter process.",
                   tex_label="\\sigma_{b}")


    m <= parameter(:sig_g, 0.3, (1e-7, 100.), (1e-7, 100.), ModelConstructors.Exponential(), RootInverseGamma(2., 0.75), fixed=false,
                   description="σ_g: The standard deviation of the government spending process.",
                   tex_label="\\sigma_{g}")
    #change desc
    m <= parameter(:sig_r, 0.1, (1e-7, 100.), (1e-7, 100.), ModelConstructors.Exponential(), RootInverseGamma(2., 0.20), fixed=false,
                   tex_label="\\sigma_{_r}")


    #end

    # steady states
    m <= SteadyStateParameter(:zstar, NaN, description="Steady-state growth rate of productivity", tex_label="\\z_*")
    m <= SteadyStateParameter(:rstar, NaN, tex_label="\\r_*")
m <= SteadyStateParameter(:rkstar, NaN, tex_label="\\r^k_*")
 m <= SteadyStateParameter(:omegastar, NaN, tex_label="\\omega_*")
    m <= SteadyStateParameter(:Bigphi, NaN, tex_label="\\Phi_*")
    m <= SteadyStateParameter(:kstar, NaN, description="Effective capital that households rent to firms in the steady state.", tex_label="\\k_*")
m <= SteadyStateParameter(:kbarstar, NaN, description="Total capital owned by households in the steady state.", tex_label="\\bar{k}_*")
m <= SteadyStateParameter(:istokbarst, NaN, tex_label="\\istokbarst")
    m <= SteadyStateParameter(:istar, NaN, description="Detrended steady-state investment", tex_label="\\i_*")
    m <= SteadyStateParameter(:ystar, NaN, tex_label="\\y_*")
    m <= SteadyStateParameter(:cstar, NaN, tex_label="\\c_*")
m <= SteadyStateParameter(:xistar, NaN, tex_label="\\xi_*")
m <= SteadyStateParameter(:phi, NaN, tex_label="\\phi")
m <= SteadyStateParameter(:Rstarn, NaN, tex_label="\\R_*_n")
m <= SteadyStateParameter(:wstar, NaN, tex_label="\\w_*")
m <= SteadyStateParameter(:mstar, NaN, tex_label="\\m_*")
m <= SteadyStateParameter(:Lstar, NaN, tex_label="\\L_*")
end

"""
```
steadystate!(m::DSSW)
```

Calculates the model's steady-state values. `steadystate!(m)` must be called whenever the parameters of `m` are updated.
"""
function steadystate!(m::DSSW)
    m[:Lstar]    = 1.
    m[:zstar]    = (m[:gam]/100)+(m[:alp]/(1-m[:alp]))*log(exp(m[:ups]/100))
    m[:rstar]    = (1/exp(-m[:bet]/100))*exp((m[:gam]/100))*exp(m[:ups]/100)^(m[:alp]/(1-m[:alp]))
    m[:rkstar]   = (1/exp(-m[:bet]/100))*exp((m[:gam]/100))*exp(m[:ups]/100)^(1/(1-m[:alp]))-(1-m[:del])
    m[:omegastar]= (m[:alp]^(m[:alp])*(1-m[:alp])^(1-m[:alp])*m[:rkstar]^(-m[:alp])/(1+m[:laf]))^(1/(1-m[:alp]))
    m[:Bigphi]   = m[:laf]*m[:omegastar]*m[:Lstar]/(1-m[:alp])
    m[:kstar]    = (m[:alp]/(1-m[:alp]))*m[:omegastar]*m[:Lstar]/m[:rkstar]
    m[:kbarstar] = m[:kstar]*exp((m[:gam]/100))*exp(m[:ups]/100)^(1/(1-m[:alp]))
    m[:istokbarst] = 1-((1-m[:del])/(exp((m[:gam]/100))*exp(m[:ups]/100)^(1/(1-m[:alp]))))
    m[:istar]    = m[:kbarstar]*m[:istokbarst]
    m[:ystar]    = (m[:kstar]^m[:alp])*(m[:Lstar]^(1-m[:alp]))-m[:Bigphi]
    m[:cstar]    = (m[:ystar]/(m[:gstar]+1))-m[:istar]
    m[:xistar]   = (1/m[:cstar])*((1/(1-m[:h]*exp(-(m[:gam]/100))*exp(m[:ups]/100)^(-m[:alp]/(1-m[:alp]))))-(m[:h]*exp(-m[:bet]/100)/(exp((m[:gam]/100))*exp(m[:ups]/100)^(m[:alp]/(1-m[:alp]))-m[:h])))
    m[:phi]      = m[:Lstar]^(-m[:nu_l])*m[:omegastar]*m[:xistar]/(1+m[:law])
    m[:Rstarn]   = exp(m[:pistar]/100)*m[:rstar]
    m[:wstar]    = (1/(1+m[:laf]) * (m[:alp]^m[:alp]) * (1-m[:alp])^(1-m[:alp]) * m[:rkstar]^(-m[:alp]) )^(1/(1-m[:alp]))
    #m[:mstar]    = (m[:chi] * (m[:Rstarn]/(1-m[:Rstarn])) * (1/m[:xistar]) )^(1/m[:nu_m]) # Not working

#=
    m[:Rstarn]   = m[:pistar]*m[:rstar]
    m[:rkstar]   = m[:rstar]*m[:Upsilon] - (1-m[:δ])
    m[:wstar]    = (m[:α]^m[:α] * (1-m[:α])^(1-m[:α]) * m[:rkstar]^(-m[:α]) / m[:Φ])^(1/(1-m[:α]))
    m[:Lstar]    = 1.
    m[:kstar]    = (m[:α]/(1-m[:α])) * m[:wstar] * m[:Lstar] / m[:rkstar]
    m[:kbarstar] = m[:kstar] * (1+m[:γ]) * m[:Upsilon]^(1 / (1-m[:α]))
    m[:istar]    = m[:kbarstar] * (1-((1-m[:δ])/((1+m[:γ]) * m[:Upsilon]^(1/(1-m[:α])))))
    m[:ystar]    = (m[:kstar]^m[:α]) * (m[:Lstar]^(1-m[:α])) / m[:Φ]
    m[:cstar]    = (1-m[:g_star])*m[:ystar] - m[:istar]
    m[:wl_c]     = (m[:wstar]*m[:Lstar])/(m[:cstar]*m[:λ_w])
=#
    return m
end

function model_settings!(m::DSSW)

    default_settings!(m)

    # Anticipated shocks
    m <= Setting(:n_mon_anticipated_shocks, 0)
    m <= Setting(:n_mon_anticipated_shocks_padding, 20)

    # Estimation
    m <= Setting(:reoptimize, true)
    m <= Setting(:recalculate_hessian, true)

    # Data
    m <= Setting(:data_vintage, "150827")
    m <= Setting(:data_id, 1, "Dataset identifier")
    m <= Setting(:cond_full_names, [:obs_gdp, :obs_nominalrate])
    m <= Setting(:cond_semi_names, [:obs_nominalrate])
end

"""
```
parameter_groupings(m::DSSW)
```

Returns an `OrderedDict{String, Vector{Parameter}}` mapping descriptions of
parameter groupings (e.g. \"Policy Parameters\") to vectors of
`Parameter`s. This dictionary is passed in as a keyword argument to
`prior_table`.
"""
function parameter_groupings(m::DSSW)
    #=
    policy     = [:ψ1, :ψ2, :ψ3, :ρ, :ρ_rm, :σ_rm]
    sticky     = [:ζ_p, :ι_p, :ϵ_p, :ζ_w, :ι_w, :ϵ_w]
    other_endo = [:γ, :α, :β, :σ_c, :h, :ν_l, :δ, :Φ, :S′′, :ppsi,
                  :Lmean, :λ_w, :π_star, :g_star]
    processes  = [:ρ_g, :ρ_b, :ρ_μ, :ρ_z, :ρ_λ_f, :ρ_λ_w, :η_λ_f, :η_λ_w,
                  :σ_g, :σ_b, :σ_μ, :σ_z, :σ_λ_f, :σ_λ_w, :η_gz]

    all_keys     = Vector[policy, sticky, other_endo, processes]
    all_params   = map(keys -> [m[θ]::Parameter for θ in keys], all_keys)
    descriptions = ["Policy", "Nominal Rigidities",
                    "Other Endogenous Propagation and Steady State",
                    "Exogenous Process"]

    groupings = OrderedDict{String, Vector{Parameter}}(zip(descriptions, all_params))

    # Ensure no parameters missing
    incl_params = vcat(collect(values(groupings))...)
    excl_params = [m[θ] for θ in [:Upsilon, :e_y, :e_L, :e_w, :e_π, :e_R, :e_c, :e_i]]
    @assert isempty(setdiff(m.parameters, vcat(incl_params, excl_params)))
    @assert isempty(setdiff(vcat(incl_params, excl_params), m.parameters))
    =#
    groupings = nothing
    return groupings
end

"""
```
shock_groupings(m::DSSW)
```

Returns a `Vector{ShockGroup}`, which must be passed in to
`plot_shock_decomposition`. See `?ShockGroup` for details.
"""
function shock_groupings(m::DSSW)
    #=
    gov = ShockGroup("g", [:g_sh], RGB(0.70, 0.13, 0.13)) # firebrick
    bet = ShockGroup("b", [:b_sh], RGB(0.3, 0.3, 1.0))
    tfp = ShockGroup("z", [:z_sh], RGB(1.0, 0.55, 0.0)) # darkorange
    pmu = ShockGroup("p-mkp", [:λ_f_sh], RGB(0.60, 0.80, 0.20)) # yellowgreen
    wmu = ShockGroup("w-mkp", [:λ_w_sh], RGB(0.0, 0.5, 0.5)) # teal
    pol = ShockGroup("pol", vcat([:rm_sh], [Symbol("rm_shl$i") for i = 1:n_anticipated_shocks(m)]),
                     RGB(1.0, 0.84, 0.0)) # gold
    mei = ShockGroup("mu", [:μ_sh], :cyan)
    det = ShockGroup("dt", [:dettrend], :gray40)
    =#

    return nothing #[gov, bet, tfp, pmu, wmu, pol, mei, det]
end
