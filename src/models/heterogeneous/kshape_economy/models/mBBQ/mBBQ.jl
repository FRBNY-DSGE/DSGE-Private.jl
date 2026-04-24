using DSGE
using ModelConstructors
import ModelConstructors: get_setting, description
using OrderedCollections: OrderedDict
using Random: MersenneTwister

"""
```
mBBQ{T} <: AbstractHeterogeneousModel{T}
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
    dicts::Dict{Symbol,Any}                        # container for large SS/intermediate dictionaries
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


    m.jump_variables = [:Value′_t, :mutil_cons′_t, :Va′_t, :MRS′_t, :A_hh′_t, :B_hh′_t, :C′_t, 
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
#    m.settings[:ns] = Setting(:ns, 2)
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



 regime_dates = Dict(
    1  => Dates.Date("1987-09-30"),
    2  => Dates.Date("2001-03-31"),
    3  => Dates.Date("2004-09-30"),
    4  => Dates.Date("2007-09-30"),
    5  => Dates.Date("2008-12-31"),
    6  => Dates.Date("2014-03-31"),
    7  => Dates.Date("2020-03-31"),
    8  => Dates.Date("2022-03-31"),
    )

    regime_monpol = Dict(
    1  => :taylor,
    2  => :taylor,
    3  => :taylor,
    4  => :qe,
    5  => :qe,
    6  => :taylor,
    7  => :qe,
    8  => :taylor,
    )
    m <= Setting(:regime_dates, regime_dates)
    m <= Setting(:regime_monpol, regime_monpol)





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
                      Vector{AbstractParameter{Float64}}(), Vector{AbstractParameter{Float64}}(),
                      # grids and keys
                      OrderedDict{Symbol,Union{Grid, Array, Float64}}(), Dict{Symbol,Any}(), OrderedDict{Symbol,Int}(),

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

    # Set observable transformations (must precede init_model_indices!, which reads observable keys)
    init_observable_mappings!(m)

    # Set settings
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
    init_grids!(m; coarse=false) 
    #TO-DO: another init_grids after loading the steadystate
    #init_grids!(m; coarse = !load_steadystate) # if steady state has not been computed, we start from a coarse grid
    #println(m[:Σ_n].value)
    # Load the steady state if it has already been computed
    # from the filepath get_setting(m, :steadystate_output_file)
   #= if load_steadystate
        load_steadystate!(m)
    end
    steadystate!(m)
=#
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

    m <= Setting(:smin, 0.181165219, "Minimum grid value for idiosyncratic income states for default grid")
    m <= Setting(:smax, 5.44249294570, "Maximum grid value for idiosyncratic income states for default grid")
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
    m <= Setting(:compute_full_steadystate, true, "Flag to avoid re-computing the full steady-state.")


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

    # Indices, declared just for initialization
    m <= Setting(:n_scalar_jumps, 1, "number of scalar jumps")
    m <= Setting(:n_scalar_states, 1, "number of scalar states")
    m <= Setting(:n_scalar_variables, get_setting(m, :n_scalar_jumps)+get_setting(m, :n_scalar_states),
                 "Num Scalars (jumps and states)")
    m <= Setting(:n_backward_looking_states, 1, "Total number of states after reduction steps") # just initializing, will count later
    m <= Setting(:n_jumps, 1, "Total number of jumps after reduction steps")
    m <= Setting(:n_model_states, get_setting(m, :n_backward_looking_states) + get_setting(m, :n_jumps),
                 "Number of model states (predetermined states and jump variables)")




end
function setup_indices!(m::mBBQ)
    #Abbreviations for ease
    state_vars = m.state_variables
    jump_vars = m.jump_variables
    endo = m.endogenous_states
    aggr_endo = m.aggregate_endogenous_states
    eqconds = m.equilibrium_conditions
    aggr_eqconds = m.aggregate_equilibrium_conditions

    # Retrieve size of the grid
    nb, na, nse = get_idiosyncratic_dims(m)
    # Compute idiosyncratic dims
    n_idio_states = get_setting(m, :nRedMarg)
    # Store compression indices
    #Marginals
    endo[:marginal_pdf_b′_t] = 1:(nb-1)
    endo[:marginal_pdf_a′_t] = nb:(nb+na-2)
    endo[:marginal_pdf_se′_t] = (nb+na-1):(nb+na+nse-3)

    #Copula-related indexation
    n_dct_copula = get_setting(m, :nCOP)
    n_distr_states = n_idio_states + n_dct_copula
    endo[:copula′_t] = (1 + n_idio_states):n_distr_states

    for (i, k) ∈ enumerate(get_aggregate_state_variables(m))
        endo[k] = (n_distr_states + i):(n_distr_states+i)
        aggr_endo[k] = i
    end

    # Update n_states to be consistent with number of
    # idiosyncratic states and jumps after reduction
    n_states = Int64(first(endo[state_vars[end]]))
    n_aggr_states = n_states - n_distr_states
    m <= Setting(:n_backward_looking_states, n_states)

    # Jump indices
    #not sure this is right
    nPoly = get_setting(m, :nPoly)
    n_idio_jumps  = 3 * nPoly
    n_states_idio_jumps  = n_states + n_idio_jumps
    endo[:Value′_t] = n_states .+ (1:(nPoly-1))
    endo[:mutil_cons′_t] = n_states .+ (nPoly:(2*nPoly-1))
    endo[:Va′_t] = n_states .+ (2*nPoly:3*nPoly-1)

    #=
    n_compression_indices = get_setting(m, :nPoly)
    endo[:Value′_t] = 1:n_compression_indices
    endo[:mutil_cons′_t] = n_compression_indices+1:2*n_compression_indices
    endo[:Va′_t] = 2*n_compression_indices+1:3*n_compression_indices =#
    for (i, k) ∈ enumerate(get_aggregate_jump_variables(m))
        endo[k] = (n_states_idio_jumps + i):(n_states_idio_jumps+i)
        aggr_endo[k] = i + n_aggr_states
    end
    m <= Setting(:n_model_states, first(endo[jump_vars[end]])) #test equiv. to :numstates
    m <= Setting(:n_jumps, get_setting(m, :n_model_states) - n_states) #compare w :numRedCtrl

    # Populating equation indices:
    eqconds[:eq_marginal_pdf_b]             =   endo[:marginal_pdf_b′_t]
    eqconds[:eq_marginal_pdf_a]             =   endo[:marginal_pdf_a′_t]
    eqconds[:eq_marginal_pdf_se]            =   endo[:marginal_pdf_se′_t]
    eqconds[:eq_copula]                     =   endo[:copula′_t]
    eqconds[:eq_value]                      =   endo[:Value′_t]
    eqconds[:eq_marginal_util_cons]         =   endo[:mutil_cons′_t]
    eqconds[:eq_marginal_value_illiquid]    =   endo[:Va′_t]

    ### DOESN'T MATTER BC bbl CODE DOES IT DIF
    #=
    exogenous_shocks = collect([:Z_sh, :G_sh, :D_sh, :R_sh, :ι_sh, :η_sh, :w_sh])
    aggr_eqn_names = [
    :eq_Z, :eq_A, :eq_G, :eq_D, :eq_R, :eq_ι, :eq_η, :eq_w]
    =#
    for (i, name) ∈ enumerate([
                               :eq_rate_monetary_policy, :eq_wage, :eq_liquid_assets,
                               :eq_central_bank_assets, :eq_tobins_q, :eq_leverage,
                               :eq_net_worth_bank, :eq_houshold_bond_rate, :eq_quantitative_easing,
                               :eq_inflation_lag, :eq_output_lag, :eq_consumption_lag,
                               :eq_investment_lag, :eq_profit_lag, :eq_unemployment_lag,
                               :eq_government_spending_lag, :eq_lump_sum_transfers,
                               :eq_fiscal_liability, :eq_z, :eq_ψ, :eq_η, :eq_D, :eq_GG,
                               :eq_ι, :eq_BB, :eq_ψ_w, :eq_MP, :eq_pm, :eq_control_capital,
                               :eq_control_bond, :eq_control_government_bond, :eq_control_tax,
                               :eq_control_transfer, :eq_control_government_spending,
                               :eq_control_lambda, :eq_control_inflation, :eq_control_vacancies,
                               :eq_control_j, :eq_control_mpl, :eq_control_elasticity_labor,
                               :eq_control_output, :eq_control_profit, :eq_control_mpk,
                               :eq_control_mpa, :eq_control_marginal_cost,
                               :eq_control_unemployment_rate, :eq_control_nn, :eq_control_M,
                               :eq_control_f, :eq_control_zz, :eq_control_xx, :eq_control_vv,
                               :eq_control_ee, :eq_control_cb, :eq_control_profit_fi,
                               :eq_control_rra, :eq_control_rr, :eq_control_investment,
                               :eq_control_x_k, :eq_control_a_g_obs, :eq_control_y_obs,
                               :eq_control_c_obs, :eq_control_i_obs, :eq_control_w_obs,
                               :eq_control_profit_obs,
                              ])
        eqconds[name] = (n_states_idio_jumps + i):(n_states_idio_jumps+i)
        aggr_eqconds[name] = i + n_aggr_states
    end
    m <= Setting(:n_model_states, first(endo[jump_vars[end]]))
    m <= Setting(:n_jumps, get_setting(m, :n_model_states) - n_states)


    m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states))
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
    # TO-DO: fix these parameter defs
    m <= parameter(:ν, 1.0, fixed = true)
    #phillips curve parameters
     m <= parameter(:κ, 0.0276503, (1e-5, 0.2), (1e-5, 0.2), SquareRoot(), GammaAlt(0.1, 0.02), fixed = false,
                   description = "κ: The slope of the Phillips curve", tex_label = "\\kappa")
    m <= parameter(:ρ_w, 0.8916942990519234, (1e-5, 5.), (1e-5, 5.), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_w: AR(1) coefficient in the wage process.",
                   tex_label = "\\rho_w")
    
    #monpol parameters
    m <= parameter(:δ, 0.3267222276512135, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.15), fixed = false,
                   description = "δ:",
                   tex_label = "\\delta") #TODO: check if this or delta is fixed
    m <= parameter(:d, 0.6738, (1e-5, 2.), (1e-5, 2.), SquareRoot(),
                   BetaAlt(0.5, 0.15), fixed = true,
                   description = "δ:",
                   tex_label = "\\delta")
    m <= parameter(:ϕ, 49.7917, (1e-5, 270.), (1e-5, 270.), ModelConstructors.Exponential(),
                   Normal(3.0, 0.5), fixed = false,
                   description = "ϕ: Parameter in monetary policy rule.",
                   tex_label = "\\phi")
    m <= parameter(:ϕ_π, 1.532296337149513, (1e-5, 10.), (1e-5, 10.0), ModelConstructors.Exponential(),
                   Normal(1.7, 0.3), fixed = false,
                   description = "ϕ_π: Weight on inflation gap in monetary policy rule.",
                   tex_label = "\\varphi_\\pi")
    m <= parameter(:ϕ_u, 0.37475243192750335, (-0.5, 2.), (-0.5, 2.), Untransformed(),
                   Normal(0.1, 0.05), fixed = false,
                   description = "ϕ_u: Weight on unemployment gap in monetary policy rule",
                   tex_label = "\\varphi_u")
    m <= parameter(:ρ_R, 0.8818054913182859, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_R: AR(1) coefficient in the monetary policy rule.",
                   tex_label = "\\rho_R")

    #autocorrelation
    m <= parameter(:ρ_BB, 0.9983489940166279, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_BB: AR(1) coefficient in the BB process.",
                   tex_label = "\\rho_{BB}")
    m <= parameter(:ρ_B, 0.5057547780246887, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_B: AR(1) coefficient in the intertemporal preference shifter process.",
                   tex_label = "\\rho_B")
    m <= parameter(:ρ_D, 0.8440051981700979, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_D: AR(1) coefficient in the D process.",
                   tex_label = "\\rho_D")
    m <= parameter(:ρ_G, 0.7402477481386883, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_G: AR(1) coefficient in the government spending process.",
                   tex_label = "\\rho_G")
    m <= parameter(:ρ_Z, 0.9516176031961177, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_Z: AR(1) coefficient in the technology process.",
                   tex_label = "\\rho_Z")
    m <= parameter(:ρ_η, 0.8944669355937231, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_η: AR(1) coefficient in the η process.",
                   tex_label = "\\rho_\\eta")
    m <= parameter(:ρ_ι, 0.9598287691285154, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_ι: AR(1) coefficient in the ι process.",
                   tex_label = "\\rho_\\iota")
    m <= parameter(:ρ_B_F, 0.9175878755308129, (1e-5, 1 - 1e-5), (1e-5, 1-1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_B_F: AR(1) coefficient in the B_F process.",
                   tex_label = "\\rho_{B_F}")

    # Exogenous processes - standard deviations
    m <= parameter(:σ_D, 0.12371, (1e-8, 1.), (1e-8, 1.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_D: The standard deviation of the D process.",
                   tex_label = "\\sigma_D")
    m <= parameter(:σ_G, 0.0068107, (1e-8, 0.05), (1e-8, 0.05), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_G: The standard deviation of the government spending process.",
                   tex_label = "\\sigma_G")
    m <= parameter(:σ_R, 0.0010768, (1e-8, 0.005), (1e-8, 0.005), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_R: The standard deviation of the monetary policy shock.",
                   tex_label = "\\sigma_R")
    m <= parameter(:σ_Z, 0.0079954, (1e-8, 0.05), (1e-8, 0.05), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_Z: The standard deviation of the technology process.",
                   tex_label = "\\sigma_Z")
    m <= parameter(:σ_η, 0.039461, (1e-8, 0.5), (1e-8, 0.5), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_η: The standard deviation of the η process.",
                   tex_label = "\\sigma_\\eta")
    m <= parameter(:σ_ι, 0.00090722, (1e-8, 0.005), (1e-8, 0.005), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_ι: The standard deviation of the ι process.",
                   tex_label = "\\sigma_\\iota")
    m <= parameter(:σ_w, 0.031033, (1e-8, 3.), (1e-8, 3.), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_w: The standard deviation of the wage process.",
                   tex_label = "\\sigma_w")
    m <= parameter(:σ_BB, 0.0019753, (1e-8, 0.01), (1e-8, 0.01), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_BB: The standard deviation of the BB process.",
                   tex_label = "\\sigma_{BB}")
    m <= parameter(:σ_B_F, 0.010128, (1e-8, 0.05), (1e-8, 0.05), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_B_F: The standard deviation of the B_F process.",
                   tex_label = "\\sigma_{B_F}")

    # Other parameters
    m <= parameter(:γ, 0.18919, (1e-5, 1.), (1e-5, 1.), SquareRoot(),
                   BetaAlt(0.5, 0.15), fixed = false,
                   description = "γ: The log of the steady-state growth rate of technology",
                   tex_label = "\\gamma")
    m <= parameter(:ι, 0.0542, (1e-5, 0.5), (1e-5, 0.5), SquareRoot(),
                   GammaAlt(0.5, 0.2), fixed = false,
                   description = "ι: Parameter",
                   tex_label = "\\iota")
     
    # Fixed parameters
    m <= parameter(:ε_w, 1.0, fixed = true, description = "ε_w: Parameter", tex_label = "\\varepsilon_w")
    m <= parameter(:ϕ_x, 0.0, fixed = true, description = "ϕ_x: Parameter", tex_label = "\\varphi_x")
    m <= parameter(:ρ_ψ_w, 0.0, fixed = true, description = "ρ_ψ_w: Parameter", tex_label = "\\rho_{\\psi_w}")
    m <= parameter(:ϕ_var, 0.0, fixed = true, description = "ϕ_var: Parameter", tex_label = "\\varphi")
    m <= parameter(:α_lk, 0.0, fixed = true, description = "α_lk: Parameter", tex_label = "\\alpha_{lk}")
    m <= parameter(:θ_b, 0.97, fixed = true, description = "θ_b: Parameter", tex_label = "\\theta_b")
    m <= parameter(:ρ_X_QE, 0.95, fixed = true, description = "ρ_X_QE: Parameter", tex_label = "\\rho_{X_QE}")
    m <= parameter(:ϕ_π_QE, 10.0, fixed = true, description = "ϕ_π_QE: Parameter", tex_label = "\\varphi_{\\pi_QE}")
    m <= parameter(:ϕ_u_QE, 10.0, fixed = true, description = "ϕ_u_QE: Parameter", tex_label = "\\varphi_{u_QE}")
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
    m <= parameter(:τ_cp,   0.0, fixed = true, description = "extra cost per unit of central bank asset purchases", tex_label = "\\tau_{cp}")
    m <= parameter(:psi_cp, 0.0, fixed = true, description = "equity payout share of financial intermediaries", tex_label = "\\psi_{cp}")
    m <= parameter(:frac_b, 0.0, fixed = true, description = "fraction of banker households", tex_label = "\\text{frac}_b")
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
    m <= parameter(:fix_L,  0.0043555, fixed = true, description = " ", tex_label = "\\")
    #m <= parameter(:fix_L,  0.00763002310934482, fixed = true, description = " ", tex_label = "\\")
    m <= parameter(:w_bar, 1.2111925176640317, fixed = true, description ="")
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
    m <= parameter(:δ_0, 0.014851, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:fix, 0.524055, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:γ , 0.12186571534563953, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:α_kk , 0., fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:Eratio, 0.2380054186, fixed = true, tex_label = "\\mathrm{Eratio}")
    m <= parameter(:b_share, 0.0, fixed = true, tex_label = "b_{\\mathrm{share}}")
    m <= parameter(:Eshare, 0.0024194744454305124, fixed = true, tex_label = "b_{\\mathrm{share}}")
    m <= parameter(:ψ, 0.785159222538979, fixed = true, tex_label = "b_{\\mathrm{share}}")
    m <= parameter(:α , 1.7127143830661984, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:β_b , 0.9950218012776563, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:ϕ_b , 0., fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:σ_2 , 1.5, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:ρ_passive_QE , 0.95, fixed = true, description = "",
                   tex_label ="")
    m <= parameter(:MMF_ratio_1, 10., fixed = true, description = "")
    m <= parameter(:Rprem, 0.038285657, fixed = true, description = "")
    m <= parameter(:μ_χ, 9.049040346313916, fixed = true, description = "")
    m <= parameter(:σ_χ, 3.4204542765, fixed = true, description = "")
    m <= parameter(:β, 0.9931748706, fixed = true, description = "")
    m <= parameter(:ρ_A_g, 0.95, fixed = true, description = "")
    m <= parameter(:b_a_aux, 0.9944444444444445, fixed=true, description = "", tex_label = "b_{\\text{aux}}")
    m <= parameter(:v_util, 0.75, fixed = true, description = "Steady-state utilization rate")
    m <= parameter(:u_target, 0.058, fixed = true, description = "Target unemployment rate")
    m <= parameter(:v_f_target, 0.7, fixed = true, description = "Target vacancy fill rate")
    m <= parameter(:λ_sep, 0.03, fixed = true, description = "Separation rate")
    m <= parameter(:LT_ratio, 0.0, fixed = true, description = "Transfers-to-output ratio")
    m <= parameter(:wmarkdown, 1.0, fixed = true, description = "Wage markdown")
    m <= parameter(:β_L, m[:β].value, fixed = true, description = "Labor-side discount factor")
    m <= parameter(:q_ss, 1.0, fixed = true, description = "Steady-state Tobin''s q")
    m <= parameter(:R_cb_ss, 1.0069, fixed = true, description = "Steady-state policy rate")
    m <= parameter(:B_gov_ratio, 0.6, fixed = true, description = "Government debt to output ratio")
    m <= parameter(:B_F_ratio, 0.5, fixed = true, description = "Fiscal liability share of government debt")
    m <= parameter(:Ab_ratio_init, 0.1, fixed = true, description = "Initial bank equity share")
    m <= parameter(:THETA_ss, 10.0, fixed = true, description = "Steady-state leverage ratio")
    m <= parameter(:Bb_ratio, 0.5, fixed = true, description = "Family deposit share")
    m <= parameter(:guessadj, 0.5, fixed = true, description = "Initial portfolio adjustment probability")
    m <= parameter(:A_F_ratio, 0.0, fixed = true, description = "FI asset share")
    m <= parameter(:A_g_ratio, 0.0, fixed = true, description = "Central bank asset share")
    m <= parameter(:K_ss_init, 30.0, fixed = true, description = "Initial steady-state capital guess")
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

    m.grids[:StateSS] = zeros(17732)
    m.grids[:ControlSS] = zeros(52869)
    m.dicts[:SS_stats] = Dict{String, Any}()
    m.dicts[:grid] = Dict{String, Any}()
    m.dicts[:param] = Dict{String, Any}()

    m.dicts[:Jacob_base] = Dict{Symbol, Any}()


    m.grids[:mu_dist] = zeros(1)
    m.grids[:Value] = zeros(1)
    m.grids[:mutil_c] = zeros(1)
    m.grids[:Va] = zeros(1)

    m.grids[:Va] = zeros(1)


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
    m <= SteadyStateParameter(:Output_star, NaN, description = "Steady-state output", tex_label = " ")
    m <= SteadyStateParameter(:w_bar_star, NaN, description = "Steady-state wage", tex_label = " ")
    m <= SteadyStateParameter(:R_a_star, NaN, description = "Steady-state return on equity", tex_label = " ")
    m <= SteadyStateParameter(:r_l_star, NaN, description = "Steady-state labor rental rate", tex_label = " ")
    m <= SteadyStateParameter(:r_k_star, NaN, description = "Steady-state capital rental rate", tex_label = " ")
    m <= SteadyStateParameter(:mc_star, NaN, description = "Steady-state marginal cost", tex_label = " ")
    m <= SteadyStateParameter(:u_star, NaN, description = "Steady-state unemployment", tex_label = " ")
    m <= SteadyStateParameter(:n_star, NaN, description = "Steady-state employment", tex_label = " ")
    m <= SteadyStateParameter(:v_star, NaN, description = "Steady-state vacancies", tex_label = " ")
    m <= SteadyStateParameter(:Profit_star, NaN, description = "Steady-state profits", tex_label = " ")
    m <= SteadyStateParameter(:THETA_star, NaN, description = "Steady-state leverage", tex_label = " ")
    m <= SteadyStateParameter(:NWb_star, NaN, description = "Steady-state bank net worth", tex_label = " ")
    m <= SteadyStateParameter(:Profit_FI_star, NaN, description = "Steady-state FI profits", tex_label = " ")
    m <= SteadyStateParameter(:Lambda_star, NaN, description = "Steady-state discount factor", tex_label = " ")
    m <= SteadyStateParameter(:f_star, NaN, description = "Steady-state job finding rate", tex_label = " ")
    m <= SteadyStateParameter(:M_star, NaN, description = "Steady-state matches", tex_label = " ")
    m <= SteadyStateParameter(:Ab_star, NaN, description = "Steady-state bank equity holdings", tex_label = " ")
    m <= SteadyStateParameter(:Bb_star, NaN, description = "Steady-state family deposits", tex_label = " ")
    m <= SteadyStateParameter(:Cb_star, NaN, description = "Steady-state family consumption", tex_label = " ")
    m <= SteadyStateParameter(:V_star, NaN, description = "Steady-state vacancies stock", tex_label = " ")
    m <= SteadyStateParameterGrid(:mu_dist_star, Vector{Float64}(undef, 0), description = "Steady-state household distribution", tex_label = " ")
    m <= SteadyStateParameterGrid(:Value_star, Vector{Float64}(undef, 0), description = "Steady-state value function", tex_label = " ")
    m <= SteadyStateParameterGrid(:Vb_star, Vector{Float64}(undef, 0), description = "Steady-state bond marginal value", tex_label = " ")
    m <= SteadyStateParameterGrid(:Va_star, Vector{Float64}(undef, 0), description = "Steady-state capital marginal value", tex_label = " ")
    m <= SteadyStateParameterGrid(:mutil_c_star, Vector{Float64}(undef, 0), description = "Steady-state marginal utility of consumption", tex_label = " ")



end
