using DSGE
using ModelConstructors
import ModelConstructors: get_setting
using OrderedCollections: OrderedDict
using Random: MersenneTwister

if !isdefined(Main, :mBBQ)
mutable struct mBBQ{T} <: AbstractModel{T} #TODO <: AbstractHetModel{T}
    parameters::Vector{AbstractParameter{T}}               # vector of all time-invariant model parameters
    steady_state::Vector{AbstractParameter{T}}             # model steady-state values

    # Temporary to get it to work. Need to
    # figure out a more flexible way to define
    # "grids" that are not necessarily quadrature
    # grids within the model
# TODO: maybe add field/type to hold reduction information e.g. DCTindices (but not coefficient values), copula info
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

    spec::String                                     # Model specification (eg "bayer_born_luetticke")
    subspec::String                                  # Model subspecification (eg "ss1")
    settings::Dict{Symbol,Setting}                   # Settings/flags for computation
    test_settings::Dict{Symbol,Setting}              # Settings/flags for testing mode
    rng::MersenneTwister                             # Random number generator
    testing::Bool                                    # Whether we are in testing mode or not
    observable_mappings::OrderedDict{Symbol, Observable}
    pseudo_observable_mappings::OrderedDict{Symbol, PseudoObservable}
end
end  # end if !isdefined

description(m::mBBQ) = "New York Fed DSGE Model mBBQ"#, $(m.subspec)."

"""
`init_grids!(m::mBBQ; coarse::Bool = true)`

Placeholder grid initialization. Populates m.grids with coarse dimensions when needed.
When full HANK grid setup is implemented, this should build b/a/se grids and update
m.settings[:nx], m.settings[:ns].
"""
function init_grids!(m::mBBQ; coarse::Bool = true)
    # Use coarse defaults from init_model_indices! (nx=12, ns=2).
    # m.grids is already initialized in the constructor.
    # Full grid construction (b, a, se meshes, etc.) to be implemented.
    return nothing
end

"""
`init_model_indices!(m::mBBQ)`

Arguments:
`m:: mBBQ`: a model object

Description:
Initializes indices for all of `m`'s model states, shocks, and equilibrium conditions.
By model states, we mean the states in the model's reduced form state-space representation,
hence these states include both predetermined states and jumps.
"""
function init_model_indices!(m::mBBQ)
    # TODO: maybe want to rename some of the indices to make it explicit
    #       exactly what we're perturbing (e.g. we are perturbing DCT coefficients or CDFs)

    # Predetermined states
    # TODO: delete states that should just be augmented states
    m.state_variables = [:marginal_pdf_m′_t, :marginal_pdf_k′_t, :marginal_pdf_y′_t, :copula′_t,

                         :A′_t, :Z′_t, :Ψ′_t, :RB′_t, :μ_p′_t, :μ_w′_t, :σ′_t, # TODO: rename σ′_t to σ_sq′_t to make it clear it's the variance
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
#=
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

=#
    println("testing new ordering bbl")
    m.jump_variables = [:Vm′_t, :Vk′_t,
                        # Function valued jumps above, Distribution Names
                        :Gini_C′_t, :Gini_X′_t, :I90_share′_t, :I90_share_net′_t,:W90_share′_t, :sd_log_y′_t,
                        # Endogenous scalar-valued jumps
                        :rk′_t, :w′_t, :K′_t, :π′_t, :π_w′_t, :Y′_t, :C′_t, :q′_t, :N′_t, :mc′_t,
                        :mc_w′_t, :u′_t, :Ht′_t, :avg_tax_rate′_t, :T′_t, :I′_t, :B′_t,
                        :BD′_t, :BY′_t, :TY′_t, :mc_w_w′_t, :G′_t, :τ_level′_t, :τ_prog′_t, :Ygrowth′_t,
                        :Bgrowth′_t, :Igrowth′_t, :wgrowth′_t, :Cgrowth′_t,
                        :Tgrowth′_t, :LP′_t, :LP_XA′_t, :union_profits′_t,
                        :profits′_t]


    m.aggregate_jump_variables = m.jump_variables[3:end]

    # Update number of scalar states and jumps
    n_scalar_states = length(m.aggregate_state_variables)
    n_scalar_jumps = length(m.aggregate_jump_variables)
    m.settings[:n_scalar_states] = Setting(:n_scalar_states, n_scalar_states)

    # Grid dimensions for household-level steady state (lstar, cstar, μstar).
    # Must be set before init_parameters! — init_grids! may overwrite with actual values.
    # Defaults: coarse grid nb=4, na=3, nse=2 → nx=nb*na=12, ns=nse=2
    m.settings[:nx] = Setting(:nx, 12)
    m.settings[:ns] = Setting(:ns, 2)
    m.settings[:n_scalar_jumps] = Setting(:n_scalar_jumps, n_scalar_jumps)
    m.settings[:n_scalar_variables] = Setting(:n_scalar_variables, n_scalar_jumps + n_scalar_states)

    m.settings[:PRightAll] = Setting(:PRightAll, Matrix{Float64}(undef, 0, 0))
    m.settings[:State2Control] = Setting(:State2Control, Matrix{Float64}(undef, 0, 0))
    m.settings[:LOMstate] = Setting(:LOMstate, Matrix{Float64}(undef, 0, 0))
    m.settings[:PRightStates] = Setting(:PRightStates, Matrix{Float64}(undef, 0, 0))

    # Exogenous shocks
    exogenous_shocks = collect([:A_sh, :Z_sh, :Ψ_sh, :μ_p_sh, :μ_w_sh, :G_sh, :R_sh, :S_sh, :P_sh])

    shock2state_map = Dict(:A_sh => :A_t, :Z_sh => :Z_t, :Ψ_sh => :Ψ_t, :μ_p_sh => :μ_p_t, :μ_w_sh => :μ_w_t, :G_sh => :G_sh_t, :P_sh => :P_sh_t, :R_sh => :R_sh_t, :S_sh => :S_sh_t)
    m.settings[:shock2state] = Setting(:shock2state, shock2state_map)

    standard_deviation_dictionary = Dict(:A_sh => :σ_A, :Z_sh => :σ_Z, :Ψ_sh => :σ_Ψ, :μ_p_sh => :σ_μ_p,
                                     :μ_w_sh => :σ_μ_w, :G_sh => :σ_G, :R_sh => :σ_R, :S_sh => :σ_S,
                                     :P_sh => :σ_P)

    ## Check These Values Commented Out the 48, 49 etc. values, NOT SURE WHY THEY WERE SET THIS WAY
  #=  standard_deviation_dictionary = Dict(:A_sh => 48, :Z_sh => 49, :Ψ_sh => 50, :μ_p_sh => 51,
                                     :μ_w_sh => 52, :G_sh => 55, :R_sh => 54, :S_sh => 53,
                                     :P_sh => 56)=#

#standard_deviation_dictionary = Dict(:A_sh => 0.00033, :Z_sh => 0.00033, :Ψ_sh => 0.00033, :μ_p_sh => 0.00033,
                                    # :μ_w_sh => 0.00033, :G_sh => 0.00033, :R_sh => 0.00033, :S_sh => 0.511538,
                                    # :P_sh => 0.00033)

    m.settings[:shock_to_deviation_dict] = Setting(:shock_to_deviation_dict, standard_deviation_dictionary)

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
function mBBQ(subspec::String="ss1";
                            custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
                            load_steadystate::Bool = false, load_jacobian::Bool = false,
                            testing = false)

    # Model-specific specifications
    spec               = "mBBQ"
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = mBBQ{Float64}(
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

        #TODO: SETTINGS
    # default_settings!(m)  # TODO: Implement or ensure AbstractHetModel <: AbstractDSGEModel

    # # Set observable transformations
    # init_observable_mappings!(m)

    # # Set settings
    # model_settings!(m)
    # for custom_setting in values(custom_settings)
    #     m <= custom_setting
    # end

    # if get_setting(m, :original_dataset)
    #     _init_original_observable_mappings!(m, m.observable_mappings)
    # end

    # Initialize model indices
    init_model_indices!(m)

    # Init what will keep track of # of states, jumps, and states/jump indices
    # (need model indices first)
    # init_states_and_jumps!(m, states, jumps)

    # Initialize parameters
    init_parameters!(m)

    # Initialize grids
    init_grids!(m; coarse = !load_steadystate) # if steady state has not been computed, we start from a coarse grid
    #println(m[:Σ_n].value)
    # Load the steady state if it has already been computed
    # from the filepath get_setting(m, :steadystate_output_file)
    if load_steadystate
        load_steadystate!(m)
    end
    # steadystate!(m)

    # So that the indices of m.endogenous_states reflect the normalization
    # normalize_model_state_indices!(m)
    #println(m[:Σ_n].value)
    #init_subspec!(m)
    #println(m[:Σ_n].value)
    # Load Jacobian if it has already been computed
    if load_jacobian
        load_jacobian!(m)
    end
    #println(m[:Σ_n].value)
    return m
end


"""
```
init_parameters!(m::mBBQ)
```

Initializes the model's parameters, as well as empty values for the steady-state
parameters (in preparation for `steadystate!(m)` being called to initialize
those).
"""
function init_parameters!(m::mBBQ)
    
    #phillips curve parameters
    m <= parameter(:κ, 0.0215, (1e-5, 5.), (1e-5, 5.), SquareRoot(), GammaAlt(0.1, 0.02), fixed = false,
                   description = "κ: The slope of the Phillips curve", tex_label = "\\kappa")
    m <= parameter(:ρ_w, 0.9085, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_w: AR(1) coefficient in the wage process.",
                   tex_label = "\\rho_w")
    
    #monpol parameters
    m <= parameter(:δ, 0.3205, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.15), fixed = false,
                   description = "δ:",
                   tex_label = "\\delta")
    m <= parameter(:φ, 5.1115, (1e-5, 10.), (1e-5, 10.), ModelConstructors.Exponential(),
                   Normal(3.0, 0.5), fixed = false,
                   description = "φ: Parameter in monetary policy rule.",
                   tex_label = "\\phi")
    m <= parameter(:φ_π, 1.4815, (1e-5, 10.), (1e-5, 10.0), ModelConstructors.Exponential(),
                   Normal(1.7, 0.3), fixed = false,
                   description = "φ_π: Weight on inflation gap in monetary policy rule.",
                   tex_label = "\\varphi_\\pi")
    m <= parameter(:φ_u, 0.3855, (-0.5, 0.5), (-0.5, 0.5), Untransformed(),
                   Normal(0.1, 0.05), fixed = false,
                   description = "φ_u: Weight on unemployment gap in monetary policy rule",
                   tex_label = "\\varphi_u")
    m <= parameter(:ρ_R, 0.8825, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_R: AR(1) coefficient in the monetary policy rule.",
                   tex_label = "\\rho_R")

    #autocorrelation
    m <= parameter(:ρ_BB, 0.9985, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_BB: AR(1) coefficient in the BB process.",
                   tex_label = "\\rho_{BB}")
    m <= parameter(:ρ_B, 0.6135, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_B: AR(1) coefficient in the intertemporal preference shifter process.",
                   tex_label = "\\rho_B")
    m <= parameter(:ρ_D, 0.8335, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_D: AR(1) coefficient in the D process.",
                   tex_label = "\\rho_D")
    m <= parameter(:ρ_G, 0.7315, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_G: AR(1) coefficient in the government spending process.",
                   tex_label = "\\rho_G")
    m <= parameter(:ρ_Z, 0.9515, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_Z: AR(1) coefficient in the technology process.",
                   tex_label = "\\rho_Z")
    m <= parameter(:ρ_η, 0.8885, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_η: AR(1) coefficient in the η process.",
                   tex_label = "\\rho_\\eta")
    m <= parameter(:ρ_ι, 0.9575, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_ι: AR(1) coefficient in the ι process.",
                   tex_label = "\\rho_\\iota")
    m <= parameter(:ρ_B_F, 0.9185, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_B_F: AR(1) coefficient in the B_F process.",
                   tex_label = "\\rho_{B_F}")

    # Exogenous processes - standard deviations
    m <= parameter(:σ_D, 13.0915, (1e-8, 50.), (1e-8, 50.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_D: The standard deviation of the D process.",
                   tex_label = "\\sigma_D")
    m <= parameter(:σ_G, 0.6725, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_G: The standard deviation of the government spending process.",
                   tex_label = "\\sigma_G")
    m <= parameter(:σ_R, 0.1085, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_R: The standard deviation of the monetary policy shock.",
                   tex_label = "\\sigma_R")
    m <= parameter(:σ_Z, 0.7965, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_Z: The standard deviation of the technology process.",
                   tex_label = "\\sigma_Z")
    m <= parameter(:σ_η, 3.6715, (1e-8, 20.), (1e-8, 20.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_η: The standard deviation of the η process.",
                   tex_label = "\\sigma_\\eta")
    m <= parameter(:σ_ι, 0.0915, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_ι: The standard deviation of the ι process.",
                   tex_label = "\\sigma_\\iota")
    m <= parameter(:σ_w, 3.2885, (1e-8, 20.), (1e-8, 20.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_w: The standard deviation of the wage process.",
                   tex_label = "\\sigma_w")
    m <= parameter(:σ_BB, 0.2015, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_BB: The standard deviation of the BB process.",
                   tex_label = "\\sigma_{BB}")
    m <= parameter(:σ_B_F, 1.0145, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_B_F: The standard deviation of the B_F process.",
                   tex_label = "\\sigma_{B_F}")

    # Other parameters
    m <= parameter(:γ, 0.1755, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.15), fixed = false,
                   description = "γ: The log of the steady-state growth rate of technology",
                   tex_label = "\\gamma")
    m <= parameter(:ι, 0.0555, (1e-5, 5.), (1e-5, 5.), SquareRoot(),
                   GammaAlt(0.5, 0.2), fixed = false,
                   description = "ι: Parameter",
                   tex_label = "\\iota")
    
    # Fixed parameters
    m <= parameter(:ε_w, 1.0, fixed = true, description = "ε_w: Parameter", tex_label = "\\varepsilon_w")
    m <= parameter(:φ_x, 0.0, fixed = true, description = "φ_x: Parameter", tex_label = "\\varphi_x")
    m <= parameter(:ρ_ψ_w, 0.0, fixed = true, description = "ρ_ψ_w: Parameter", tex_label = "\\rho_{\\psi_w}")
    m <= parameter(:φ_var, 0.0, fixed = true, description = "φ_var: Parameter", tex_label = "\\varphi")
    m <= parameter(:α_lk, 0.0, fixed = true, description = "α_lk: Parameter", tex_label = "\\alpha_{lk}")
    m <= parameter(:θ_b, 0.97, fixed = true, description = "θ_b: Parameter", tex_label = "\\theta_b")
    m <= parameter(:ρ_X_QE, 0.95, fixed = true, description = "ρ_X_QE: Parameter", tex_label = "\\rho_{X_QE}")
    m <= parameter(:φ_π_QE, 10.0, fixed = true, description = "φ_π_QE: Parameter", tex_label = "\\varphi_{\\pi_QE}")
    m <= parameter(:φ_u_QE, 10.0, fixed = true, description = "φ_u_QE: Parameter", tex_label = "\\varphi_{u_QE}")
    m <= parameter(:γ_B, 0.0, fixed = true, description = "γ_B: Parameter", tex_label = "\\gamma_B")
    m <= parameter(:γ_π, 0.0, fixed = true, description = "γ_π: Parameter", tex_label = "\\gamma_\\pi")
    m <= parameter(:γ_Y, 0.0, fixed = true, description = "γ_Y: Parameter", tex_label = "\\gamma_Y")
    m <= parameter(:ρ_MP, 0.0, fixed = true, description = "ρ_MP: Parameter", tex_label = "\\rho_{MP}")
    m <= parameter(:γ_T, 0.0, fixed = true, description = "γ_T: Parameter", tex_label = "\\gamma_T")
    m <= parameter(:ρ_Ψ_RP, 0.0, fixed = true, description = "ρ_Ψ_RP: Parameter", tex_label = "\\rho_{\\Psi_RP}")
    m <= parameter(:σ_RP, 0.1/100, fixed = true, description = "σ_RP: Parameter", tex_label = "\\sigma_{RP}")
    m <= parameter(:Calvo, 0.7, fixed = true, description = "Calvo: Calvo parameter", tex_label = "Calvo")

    m <= parameter(:π_cb, 1.005, fixed = true, description = "pi star", tex_label = "\\varphi_{\\pi_cb}")
    m <= parameter(:τ_cp, 0.01, fixed = true, description = "extra cost per unit of central bank asset purchases", tex_label = "\\varphi_{\\tau_cb}")
    m <= parameter(:b_a_aux2, 0.01, fixed = true, description = "extra cost per unit of central bank asset purchases", tex_label = "\\varphi_{\\tau_cb}")
    #add :ω

    # Setting steady-state parameters
    nx = get_setting(m, :nx)
    ns = get_setting(m, :ns)

    # Steady state grids for functional/distributional variables and market-clearing discount rate
    m <= SteadyStateParameterGrid(:lstar, fill(NaN, nx*ns), description = "Steady-state expected discounted
                                  marginal utility of consumption", tex_label = "l_*")
    m <= SteadyStateParameterGrid(:cstar, fill(NaN, nx*ns), description = "Steady-state consumption",
                                  tex_label = "c_*")
    m <= SteadyStateParameterGrid(:μstar, fill(NaN, nx*ns), description = "Steady-state cross-sectional density of cash on hand",
                                  tex_label = "\\mu_*")
    m <= SteadyStateParameter(:βstar, NaN, description = "Steady-state discount factor",
                              tex_label = "\\beta_*")
end
