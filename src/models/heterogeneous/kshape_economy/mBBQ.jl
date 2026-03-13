using DSGE
using ModelConstructors
import ModelConstructors: get_setting, description
using OrderedCollections: OrderedDict
using Random: MersenneTwister

"""
```
BayerBornLuetticke{T} <: AbstractHeterogeneousModel{T}
```
### Fields

#### Parameters and Steady-States
*  `parameters::Vector{AbstractParameter{T}}`: Vector of all the time invariant model
parameters.





"""
mutable struct mBBQ{T} <: AbstractHetModel{T} #TODO <: AbstractHetModel{T}
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

description(m::mBBQ) = "New York Fed DSGE Model mBBQ, $(m.subspec)."

"""
`init_grids!(m::mBBQ; coarse::Bool = true)`

Placeholder grid initialization. Populates m.grids with coarse dimensions when needed.
When full HANK grid setup is implemented, this should build b/a/se grids and update
m.settings[:nx], m.settings[:ns].
"""
#=
function init_grids!(m::mBBQ; coarse::Bool = true)
# Use coarse defaults from init_model_indices! (nx=12, ns=2).
# m.grids is already initialized in the constructor.
# Full grid construction (b, a, se meshes, etc.) to be implemented.

#in models/7KShape_Share, need to move here
return nothing
end
=#
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

    m.state_variables = [:marginal_pdf_b′_t, :marginal_pdf_a′_t, :marginal_pdf_se′_t, :copula′_t, 
                         :R_cb′_t, :w′′t, :A_b′_t, :B_b′_t, :A_gaux′_t, :Q′_t, :lev′_t, :NW_b′_t, 
                         :R′_tilde′_t, :x_cb′_t, :π_past′_t, :Y_past′_t, :C_past′_t, :I_past′_t, 
                         :Profit_past′_t, :unemp_past′_t, :G_past′_t, :LT_past′_t, :R_star′_t, :B_F′_t, :Z′_t, 
                         :ψ_rp′_t, :η′_t, :D′_t, :GG′_t, :ι′_t, :BB′_t, :ψ_w′_t, :MP′_t, :p_m′_t]


    m.aggregate_state_variables = m.state_variables[5:end]

    # Jumps
    # TODO: delete jump variables that should just be pseudo-observables
    println("testing new ordering bbl")
    #= m.jump_variables = [:Vm′_t, :Vk′_t,
    # Function valued jumps above, Distribution Names
    :Gini_C′_t, :Gini_X′_t, :I90_share′_t, :I90_share_net′_t,:W90_share′_t, :sd_log_y′_t,
    # Endogenous scalar-valued jumps
    :rk′_t, :w′_t, :K′_t, :π′_t, :π_w′_t, :Y′_t, :C′_t, :q′_t, :N′_t, :mc′_t,
    :mc_w′_t, :u′_t, :Ht′_t, :avg_tax_rate′_t, :T′_t, :I′_t, :B′_t,
    :BD′_t, :BY′_t, :TY′_t, :mc_w_w′_t, :G′_t, :τ_level′_t, :τ_prog′_t, :Ygrowth′_t,
    :Bgrowth′_t, :Igrowth′_t, :wgrowth′_t, :Cgrowth′_t,
    :Tgrowth′_t, :LP′_t, :LP_XA′_t, :union_profits′_t,
    :profits′_t] =#


    m.jump_variables = [:Value, :mutil_cons′_t, :Va′_t, :MRS′_t, :A_hh′_t, :B_hh′_t, :C′_t, 
                        :N′_t, :L′_t, :UB′_t, :K′_t, :B′_t, :B_gov_ncp′_t, :T′_t, :LT′_t, :G′_t, 
                        :λ′_t, :pi′_t, :V′_t, :J′_t, :h′_t, :v′_t, :Y′_t, :Profit′_t, :r_k′_t, 
                        :r_a′_t, :MC′_t, :unemp′_t, :nn′_t, :M′_t, :f′_t, :zz′_t, :xx′_t, :vv′_t, 
                        :ee′_t, :C_b′_t, :Profit_FI′_t, :RRa′_t, :RR′_t, :I′_t, :x_k′_t]
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
    m.settings[:l] = Setting(:l, 12)  # grid dimension for steady-state household variables (lstar, cstar, μstar)
    m.settings[:n_scalar_jumps] = Setting(:n_scalar_jumps, n_scalar_jumps)
    m.settings[:n_scalar_variables] = Setting(:n_scalar_variables, n_scalar_jumps + n_scalar_states)

    m.settings[:PRightAll] = Setting(:PRightAll, Matrix{Float64}(undef, 0, 0))
    m.settings[:State2Control] = Setting(:State2Control, Matrix{Float64}(undef, 0, 0))
    m.settings[:LOMstate] = Setting(:LOMstate, Matrix{Float64}(undef, 0, 0))
    m.settings[:PRightStates] = Setting(:PRightStates, Matrix{Float64}(undef, 0, 0))
    m <= Setting(:n_mon_anticipated_shocks, 0)
    m <= Setting(:n_mon_anticipated_shocks_padding, 0)
    # Exogenous shocks
    exogenous_shocks = collect([:Z_sh, :G_sh, :D_sh, :R_sh, :ι_sh, :η_sh, :w_sh])
    shock2state_map = Dict(:Z_sh => :Z_t, :G_sh => :G_t, :D_sh => :D_t, :R_sh => :R_t, :ι_sh => :ι_t,
                           :η_sh => :η_t, :w_sh => :w_t)
    m.settings[:shock2state] = Setting(:shock2state, shock2state_map)

    standard_deviation_dictionary = Dict(:A_sh => :σ_A, :Z_sh => :σ_Z, :Ψ_sh => :σ_Ψ, :μ_p_sh => :σ_μ_p,
                                         :μ_w_sh => :σ_μ_w, :G_sh => :σ_G, :R_sh => :σ_R, :S_sh => :σ_S,
                                         :P_sh => :σ_P)

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
end 

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
    DSGE.default_settings!(m)  # TODO: Implement or ensure AbstractHetModel <: AbstractDSGEModel

    # # Set observable transformations
    # init_observable_mappings!(m)

    # # Set settings
    model_settings!(m)
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
    #init_grids!(m; coarse=false) 
    #TO-DO: another init_grids after loading the steadystate
    #init_grids!(m; coarse = !load_steadystate) # if steady state has not been computed, we start from a coarse grid
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

function model_settings!(m::mBBQ)
    default_settings!(m)
    #Grid Settings
    m <= Setting(:na, 40, "Number of illiquid asset(equity) points for default grid")
    m <= Setting(:amin, 0, "Minimum grid value for illiquid asset(equity) for default grid")
    m <= Setting(:amax, 1500, "Maximum grid value for illiquid asset(equity) for default grid")

    m <= Setting(:nb, 40, "Number of liquid asset(bond) points for default grid")
    m <= Setting(:bmin, -1.3006141532, "Minimum grid value for liquid asset(bond) for default grid") #placeholder
    m <= Setting(:bmax, 1000, "Maximum grid value for liquid asset(bond) for default grid")

    m <= Setting(:smin, 0.1811542190, "Minimum grid value for idiosyncratic income states for default grid")
    m <= Setting(:smax, 5.4424929457, "Maximum grid value for idiosyncratic income states for default grid")
    m <= Setting(:ns, 5, "Number of idio. income states for default grid")

    m <= Setting(:nse, get_setting(m, :ns)*2+1, "Number of points in income grid") 

    #Labor Supply Settings
    m <= Setting(:n, 1.0, "Labor supply") 
    m <= Setting(:in, 0.0005, "Prob. of becoming an enterpreneur")
    m <= Setting(:out,  0.2061564501, "Prob. of entrepreneur -> worker")
    m <= Setting(:in_s, 0.005, "Prob. of becoming a high-skilled worker")
    m <= Setting(:out_s, 0.1581289398, "Prob. of high skilled ->  worker")
    m <= Setting(:in_l, 0.01, "Prob. of becoming a low skilled worker")
    m <= Setting(:out_l, 0.0937744842, "Prob. of low skilled -> worker")
    m <= Setting(:b_ratio, 0.4, "Replacement ratio for unemployment benefit")

    #Copula Settings
    m <= Setting(:reduc_copula, 100)
    #=    m <= Setting(:na_copula, 10)
    m <= Setting(:nb_copula, 10)
    m <= Setting(:nse_copula, 10) =#
    m <= Setting(:dct_compression_indices, Dict{Symbol, Vector{Int}}()) #make not a dict
    m <= Setting(:n_copula_dct_coefficients, 98, "Number of coefficients in the DCT compression of the "*
                 "distribution over idiosyncratic states to approximate a perturbation in the copula")


    # Temp settings
    m <= Setting(:data_vintage, "210504")
    m <= Setting(:cond_vintage, "210504")
    m <= Setting(:data_id, 1793)
    m <= Setting(:cond_id, 1793)
    m <= Setting(:date_zlb_start, quartertodate("2009-Q1")) # ZLB measured using Wu and Xia (2016) shadow FFR, which starts in 2009:Q1
    m <= Setting(:date_mainsample_start, quartertodate("1954-Q4"))
    m <= Setting(:date_presample_start, quartertodate("1954-Q4"))
    m <= Setting(:date_forecast_start, quartertodate("2020-Q1"))
    m <= Setting(:date_conditional_end, quartertodate("2020-Q1"))


    #Regime Settings
    m <= Setting(:regime_switching, true)
    m <= Setting(:n_regimes, 2)
    m <= Setting(:n_hist_regimes, 1)   # how many regimes appear in estimation sample
    m <= Setting(:regime_dates, Dict{Int, Date}())  # populated by user at runtime

    ## Monetary Policy
    m <= Setting(:n_mon_anticipated_shocks_padding, 0) # anticipated shocks are not used

end
function setup_indices!(m::mBBQ)
    #Abbreviations for ease
    state_vars = m.state_variables
    jump_vars = m.jump_variables
    endo = m.endogenous_states
    aggr_end = m.aggregate_endogenous_states
    eqconds = m.equilibrium_conditions
    aggr_eqconds = m.aggregate_equilibrium_conditions

    #Retrieve size of the grid
    nb, na, nse = get_idiosyncratic_dims(m)
    #=
    na = get_setting(m, :na)
    nb = get_setting(m, :nb)
    ns = get_setting(m, :ns)
    nse = get_setting(m, :nse)
    =#      
    #compute idiosyncratic dims
    n_idio_states = na + nb + nse - 3 #nse more accurately 
    dof_to_remove = 3

    #Marginals
    endo[:marginal_b′_t] = 1:nb-1
    endo[:marginal_a′_t] = nb+1:nb+na-1
    endo[:marginal_se′_t] = nb+na+1:nse-1

    #Copula-related indexation
    n_dct_copula = length(get_setting(m, :dct_compression_indices)[:copula])
    n_distr_states = n_idio_states + n_dct_copula
    #n_idio_states + get_setting(m, :n_copula_dct_coefficients) 
    ##TO-DO: modify state_reduc_tvcopula.jl to save all those grid items into model obj

    endo[:copula′_t] = (1 + n_idio_states):n_distr_states
    for (i, k) ∈ enumerate(get_aggregate_state_variables(m))
        endo[k] = (n_distr_states + i):(n_distr_states+i)
        aggr_endo[k] = i
    end

    # Update n_states to be consistent with number of
    # idiosyncratic states and jumps after reduction
    n_states = first(endo[state_vars[end]])
    n_aggr_states = n_states - n_distr_states
    m <= Setting(:n_backward_looking_states, n_states)

    # Jump indices
    # URGENT TO-DO: FIGURE OUT HOW TO DEAL WITH THE BEGINNING JUMP VARS

    for (i, k) ∈ enumerate(get_aggregate_jump_variables(m))
        endo[k] = (n_states_idio_jumps + i):(n_stapes_idio_jumps+i)
        aggr_endo[k] = i + n_aggr_states
    end



end


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

# TODO: maybe add coarse as a kwarg

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
    m <= parameter(:κ, 0.05247664549755819, (1e-5, 5.), (1e-5, 5.), SquareRoot(), GammaAlt(0.1, 0.02), fixed = false,
                   description = "κ: The slope of the Phillips curve", tex_label = "\\kappa")
    m <= parameter(:ρ_w, 0.7982120589417719, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_w: AR(1) coefficient in the wage process.",
                   tex_label = "\\rho_w")

    #monpol parameters
    m <= parameter(:δ, 0.3267222276512135, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.15), fixed = false,
                   description = "δ:",
                   tex_label = "\\delta") #TODO: check if this or delta is fixed
    m <= parameter(:d, 0.8164683082325113, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.15), fixed = true,
                   description = "δ:",
                   tex_label = "\\delta")
    m <= parameter(:φ, 5.1115, (1e-5, 10.), (1e-5, 10.), ModelConstructors.Exponential(),
                   Normal(3.0, 0.5), fixed = false,
                   description = "φ: Parameter in monetary policy rule.",
                   tex_label = "\\phi")
    m <= parameter(:φ_π, 1.3101032779695438, (1e-5, 10.), (1e-5, 10.0), ModelConstructors.Exponential(),
                   Normal(1.7, 0.3), fixed = false,
                   description = "φ_π: Weight on inflation gap in monetary policy rule.",
                   tex_label = "\\varphi_\\pi")
    m <= parameter(:φ_u, 0.37475243192750335, (-0.5, 0.5), (-0.5, 0.5), Untransformed(),
                   Normal(0.1, 0.05), fixed = false,
                   description = "φ_u: Weight on unemployment gap in monetary policy rule",
                   tex_label = "\\varphi_u")
    m <= parameter(:ρ_R, 0.792722753458172, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_R: AR(1) coefficient in the monetary policy rule.",
                   tex_label = "\\rho_R")

    #autocorrelation
    m <= parameter(:ρ_BB, 0.9886677881949328, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_BB: AR(1) coefficient in the BB process.",
                   tex_label = "\\rho_{BB}")
    m <= parameter(:ρ_B, 0.5057547780246887, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_B: AR(1) coefficient in the intertemporal preference shifter process.",
                   tex_label = "\\rho_B")
    m <= parameter(:ρ_D, 0.9996850817482046, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_D: AR(1) coefficient in the D process.",
                   tex_label = "\\rho_D")
    m <= parameter(:ρ_G, 0.9985898293160633, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_G: AR(1) coefficient in the government spending process.",
                   tex_label = "\\rho_G")
    m <= parameter(:ρ_Z, 0.9952045106146552, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_Z: AR(1) coefficient in the technology process.",
                   tex_label = "\\rho_Z")
    m <= parameter(:ρ_η, 0.9608468601131817, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_η: AR(1) coefficient in the η process.",
                   tex_label = "\\rho_\\eta")
    m <= parameter(:ρ_ι, 0.9783872396306657, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_ι: AR(1) coefficient in the ι process.",
                   tex_label = "\\rho_\\iota")
    m <= parameter(:ρ_B_F, 0.9504808782484778, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
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
    m <= parameter(:ι, 0.03168618124805551, (1e-5, 5.), (1e-5, 5.), SquareRoot(),
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
    m <= parameter(:ρ_mp, 0.0, fixed = true, description = "ρ_MP: Parameter", tex_label = "\\rho_{MP}")
    m <= parameter(:γ_T, 0.0, fixed = true, description = "γ_T: Parameter", tex_label = "\\gamma_T")
    m <= parameter(:ρ_ψ_rp, 0.0, fixed = true, description = "ρ_Ψ_RP: Parameter", tex_label = "\\rho_{\\Psi_RP}")
    m <= parameter(:σ_RP, 0.1/100, fixed = true, description = "σ_RP: Parameter", tex_label = "\\sigma_{RP}")
    m <= parameter(:Calvo, 0.7, fixed = true, description = "Calvo: Calvo parameter", tex_label = "Calvo")
    m <= parameter(:η, 3.0, fixed = true, description = "η", tex_label = "\\η")
    m <= parameter(:ρ_S, 0.93, fixed = true, description = "Persistence of productivity shocks",
                   tex_label = "\\rho_S")
    m <= parameter(:σ_S, 0.03, fixed = true, description = "Standard deviation of productivity shocks", 
                   tex_label = "\\sigma_S")

    m <= parameter(:π_cb, 1.005, fixed = true, description = "pi star", tex_label = "\\varphi_{\\pi_cb}")
    m <= parameter(:τ_cp, 0.0, fixed = true, description = "extra cost per unit of central bank asset purchases", tex_label = "\\varphi_{\\tau_cb}")
    m <= parameter(:b_a_aux2, 1.0, fixed = true, description = "extra cost per unit of central bank asset purchases", tex_label = "\\varphi_{\\tau_cb}")
    m <= parameter(:ω, 0.007975547917397408, fixed=true, description="")
    m <= parameter(:σ_s, 0.03, fixed=true, description="Standard deviation of productivity shock")
    m <= parameter(:b_aux, 1.6335, fixed=true, description="Exponent on bonus component of worker income")
    m <= parameter(:ξ, 3.0, fixed = true, description="Inverse frisch elasticity", tex_label = "\\xi")
    m <= parameter(:ϕ, 50.01728408708132, fixed = true, description="", tex_label = "\\phi")
    m <= parameter(:fix2, 0.03981570985425617, fixed = true, description="", tex_label = "\\phi")
    m <= parameter(:τ_a, 0.3517469674818686, fixed = true, description="tax rate assets", tex_label = "\\phi")
    m <= parameter(:τ_w, 0.3517469674818686, fixed = true, description="tax rate wages", tex_label = "\\phi")
    m <= parameter(:λ_aux, 0.994966466098559, fixed = true, description="", tex_label = "\\lamdba_aux")
    m <= parameter(:λ_b_aux, 0.996816728468132, fixed = true, description = " ", tex_label = "\\lambda_aux")
    m <= parameter(:fix_L, 0.00763002310934482, fixed = true, description = " ", tex_label = "\\")
    m <= parameter(:dr, 0.005555555555555556, fixed = true, description = " ", tex_label = "\\")
    m <= parameter(:in, 0.0005, fixed = true, description = " ", tex_label = "\\")
    m <= parameter(:π_bar, 1.005, fixed = true, description = "Steady state inflation rate",
                   tex_label ="\\bar{\\pi}")
    m <= parameter(:θ_2, 0.73, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:θ, 0.27, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:α_ll, 0., fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:δ_1, 1.0025, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:δ_0, 0.014851493304229206, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:fix, 0.5237134756666655, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:δ_0, 0.014851493304229206, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:γ , 0.12186571534563953, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:α_kk , 0., fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:Eratio, 0.2380054186, fixed = true, tex_label = "\\mathrm{Eratio}")
    m <= parameter(:b_share, 0.0, fixed = true, tex_label = "b_{\\mathrm{share}}")

    # Setting steady-state parameters
    nx = get_setting(m, :nx)
    ns = get_setting(m, :ns)
    l = get_setting(m, :l)
    # Steady state grids for functional/distributional variables and market-clearing discount rate
    m <= SteadyStateParameterGrid(:lstar, fill(NaN, nx*ns), description = "Steady-state expected discounted
                                  marginal utility of consumption", tex_label = "l_*")
    m <= SteadyStateParameterGrid(:cstar, fill(NaN, nx*ns), description = "Steady-state consumption",
                                  tex_label = "c_*")
    m <= SteadyStateParameterGrid(:μstar, fill(NaN, nx*ns), description = "Steady-state cross-sectional density of cash on hand",
                                  tex_label = "\\mu_*")
    m <= SteadyStateParameter(:βstar, NaN, description = "Steady-state discount factor",
                              tex_label = "\\beta_*")


    #Steady state grids
    m <= SteadyStateParameterGrid(:marginal_cdf_b_star, Vector{Float64}(undef, 0), 
                                  description = "Marginal CDF of liquid bonds (steady-state)", 
                                  tex_label = " ")
    m <= SteadyStateParameterGrid(:marginal_cdf_a_star, Vector{Float64}(undef, 0),
                                  description = "Marginal CDF of illiquid capital (steady-state)", 
                                  tex_label = " ")
    m <= SteadyStateParameterGrid(:marginal_cdf_se_star, Vector{Float64}(undef, 0),
                                  description = "Marginal CDF of income (steady-state)", 
                                  tex_label = " ")
    m <= SteadyStateParameterGrid(:marginal_pdf_b_star, Vector{Float64}(undef, 0), 
                                  description = "Marginal PDF of liquid bonds (steady-state)", 
                                  tex_label = " ")
    m <= SteadyStateParameterGrid(:marginal_pdf_a_star, Vector{Float64}(undef, 0),
                                  description = "Marginal PDF of illiquid capital (steady-state)", 
                                  tex_label = " ")
    m <= SteadyStateParameterGrid(:marginal_pdf_se_star, Vector{Float64}(undef, 0),
                                  description = "Marginal PDF of income (steady-state)", 
                                  tex_label = " ")



end
