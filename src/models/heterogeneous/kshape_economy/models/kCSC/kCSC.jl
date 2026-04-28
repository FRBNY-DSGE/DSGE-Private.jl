using DSGE
using ModelConstructors
import ModelConstructors: get_setting, description
using OrderedCollections: OrderedDict
using Random: MersenneTwister

"""
```
kCSC{T} <: AbstractHeterogeneousModel{T}
```
### Fields

#### Parameters and Steady-States
*  `parameters::Vector{AbstractParameter{T}}`: Vector of all the time invariant model
parameters.





"""
mutable struct kCSC{T} <: AbstractHetModel{T} #TODO <: AbstractHetModel{T}
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

description(m::kCSC) = "New York Fed DSGE Model kCSC, $(m.subspec)."

"""
`init_grids!(m::kCSC; coarse::Bool = true)`

Placeholder grid initialization. Populates m.grids with coarse dimensions when needed.
When full HANK grid setup is implemented, this should build b/a/se grids and update
m.settings[:nx], m.settings[:ns].
"""
#=
function init_grids!(m::kCSC; coarse::Bool = true)
# Use coarse defaults from init_model_indices! (nx=12, ns=2).
# m.grids is already initialized in the constructor.
# Full grid construction (b, a, se meshes, etc.) to be implemented.

#in models/7KShape_Share, need to move here
return nothing
end
=#
"""
`init_model_indices!(m::kCSC)`

Arguments:
`m:: kCSC`: a model object

Description:
Initializes indices for all of `m`'s model states, shocks, and equilibrium conditions.
By model states, we mean the states in the model's reduced form state-space representation,
hence these states include both predetermined states and jumps.
"""
function init_model_indices!(m::kCSC)
    # TODO: maybe want to rename some of the indices to make it explicit
    #       exactly what we're perturbing (e.g. we are perturbing DCT coefficients or CDFs)

    # Predetermined states
    # TODO: delete states that should just be augmented states
    m.state_variables = [
        :marginal_pdf_b′_t, :marginal_pdf_a′_t, :marginal_pdf_se′_t, :copula′_t,
        # aggregate states
        :R_cb′_t, :w_1′_t, :w_2′_t, :w′_t,
        :A_b′_t, :B_b′_t, :A_gaux′_t, :Q′_t, :lev′_t, :NW_b′_t,
        :R_tilde′_t, :x_cb′_t, :π_past′_t, :Y_past′_t, :C_past′_t, :I_past′_t,
        :Profit_past′_t, :unemp_past_1′_t, :unemp_past_2′_t,
        :G_past′_t, :LT_past′_t, :R_star′_t,
        :ZZ_1′_t, :ZZ_2′_t, :ZZ_3′_t, :ZZ_4′_t,
        :B_F′_t, :Z′_t, :ψ_rp′_t, :η′_t, :D′_t, :GG′_t,
        :ι′_t, :BB′_t, :ψ_w′_t, :MP′_t, :p_m′_t,
        # shock states
        :eps_1′_t, :eps_2′_t, :eps_3′_t, :eps_4′_t,
        :eps_QE′_t, :eps_RP′_t, :eps_B_F′_t, :eps_BB′_t,
        :eps_Z′_t, :eps_G′_t, :eps_D′_t, :eps_R′_t,
        :eps_iota′_t, :eps_eta′_t, :eps_w′_t,
    ]

    m.aggregate_state_variables = m.state_variables[5:end]

    # Jumps
    # TODO: delete jump variables that should just be pseudo-observables
    m.jump_variables = [
        :Value′_t, :mutil_cons′_t, :Va′_t,
        :MRS′_t, :A_hh′_t, :B_hh′_t, :C′_t,
        :N_1′_t, :N_2′_t, :L_1′_t, :L_2′_t,
        :UB′_t, :unemp_1′_t, :unemp_2′_t, :U_1′_t, :U_2′_t,
        :K′_t, :B′_t, :B_gov_ncp′_t, :T′_t, :LT′_t, :G′_t,
        :λ′_t, :pi′_t, :V_1′_t, :V_2′_t, :J′_t,
        :r_l_1′_t, :r_l_2′_t, :v′_t, :Y′_t, :Profit′_t, :r_k′_t,
        :r_a′_t, :MC′_t, :unemp′_t, :n_1′_t, :n_2′_t,
        :M_1′_t, :M_2′_t, :f_1′_t, :f_2′_t,
        :zz′_t, :xx′_t, :vv′_t, :ee′_t, :C_b′_t, :Profit_FI′_t,
        :RRa′_t, :RR′_t, :I′_t, :x_k′_t,
        :A_g_obs′_t, :Y_obs′_t, :C_obs′_t, :I_obs′_t,
        :w_1_obs′_t, :w_2_obs′_t, :Profit_obs′_t,
        :unemp_1_obs′_t, :unemp_2_obs′_t, :unemp_obs′_t,
        :pi_obs′_t, :R_obs′_t,
        :w_1_lag′_t, :w_2_lag′_t, :w_lag′_t,
        :G_obs′_t, :YY_lag′_t, :CC_lag′_t, :II_lag′_t,
        :PPROFIT_lag′_t, :uu_1_lag′_t, :uu_2_lag′_t,
        :GG_lag′_t, :A_g_lag′_t,
        :l_λ_1′_t, :l_λ_2′_t,
        :Q_lag′_t, :x_I′_t, :η2′_t, :ι2′_t,
        :LT2_lag′_t, :G2_lag′_t,
        :LT_obs′_t, :B_gov_ncp2′_t,
    ]
    m.aggregate_jump_variables = m.jump_variables[4:end]

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
    exogenous_shocks = [
        :Z_sh, :G_sh, :D_sh, :R_sh, :ι_sh, :η_sh, :w_sh,
        :ZZ_1_sh, :ZZ_2_sh, :ZZ_3_sh, :ZZ_4_sh,
        :QE_sh, :RP_sh, :BB_sh, :B_F_sh,
    ]
    shock2state_map = Dict(
        :Z_sh => :Z_t, :G_sh => :GG_t, :D_sh => :D_t, :R_sh => :MP_t,
        :ι_sh => :ι_t, :η_sh => :η_t, :w_sh => :ψ_w_t,
        :ZZ_1_sh => :ZZ_1_t, :ZZ_2_sh => :ZZ_2_t, :ZZ_3_sh => :ZZ_3_t, :ZZ_4_sh => :ZZ_4_t,
        :QE_sh => :x_cb_t, :RP_sh => :ψ_rp_t, :BB_sh => :BB_t, :B_F_sh => :B_F_t,
    )
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

function kCSC(subspec::String="ss1";
        custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
        load_steadystate::Bool = false, load_jacobian::Bool = false,
        testing = false)

    # Model-specific specifications
    spec               = "kCSC"
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = kCSC{Float64}(
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
    #init_observable_mappings!(m)

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
   # init_grids!(m; coarse=false) 
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

function model_settings!(m::kCSC)
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
    m <= Setting(:n_regimes, 8)
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
function setup_indices!(m::kCSC)
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
                               :eq_control_unemp_1_obs, :eq_control_unemp_2_obs, :eq_control_unemp_obs,
                               :eq_control_pi_obs, :eq_control_r_obs,
                               :eq_control_w_1_lag, :eq_control_w_2_lag, :eq_control_w_lag,
                               :eq_control_g_obs,
                               :eq_control_yy_lag, :eq_control_cc_lag, :eq_control_ii_lag,
                               :eq_control_pprofit_lag, :eq_control_uu_1_lag, :eq_control_uu_2_lag,
                               :eq_control_gg_lag, :eq_control_a_g_lag,
                               :eq_control_l_lambda_1, :eq_control_l_lambda_2,
                               :eq_control_q_lag, :eq_control_x_i,
                               :eq_control_eta2, :eq_control_iota2,
                               :eq_control_lt2_lag, :eq_control_g2_lag,
                               :eq_control_lt_obs, :eq_control_b_gov_ncp2,
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
init_parameters!(m::kCSC)
```

Initializes the model's parameters, as well as empty values for the steady-state
parameters (in preparation for `steadystate!(m)` being called to initialize
those).
"""
function init_parameters!(m::kCSC)
    # Fixed parameters aligned with steadystate/parameters.jl.

    # 1) Households
    m <= parameter(:σ, 1.5, fixed = true, description = "CRRA coefficient", tex_label = "\\sigma")
    m <= parameter(:ξ, 3.0, fixed = true, description = "Inverse Frisch elasticity", tex_label = "\\xi")
    m <= parameter(:σ_χ, 3.4204542765, fixed = true, description = "σ_χ: Logit scale parameter", tex_label = "\\sigma_\\chi")
    m <= parameter(:logitratio, 2.6455668209, fixed = true, description = "Logit ratio for entrepreneur share")
    m <= parameter(:μ_χ, m[:logitratio].value * m[:σ_χ].value, fixed = true, description = "μ_χ: Logit location parameter", tex_label = "\\mu_\\chi")
    m <= parameter(:ν, 1.0, fixed = true, description = "ν: Logit numerator", tex_label = "\\nu")
    m <= parameter(:guessadj, m[:ν].value / (1 + exp(m[:μ_χ].value / m[:σ_χ].value)), fixed = true,
                   description = "Initial portfolio adjustment probability")
    m <= parameter(:n_1, 1.0, fixed = true, description = "Labor endowment type 1")
    m <= parameter(:n_2, 1.0, fixed = true, description = "Labor endowment type 2")
    m <= parameter(:β, 0.9875, fixed = true, description = "β: Household discount factor", tex_label = "\\beta")
    m <= parameter(:dr, 1.0 / 180.0, fixed = true, description = "Death rate")
    m <= parameter(:reinvest, 0.0, fixed = true, description = "Reinvestment fraction")
    m <= parameter(:GHH, (1 + m[:ξ].value) / m[:ξ].value, fixed = true, description = "GHH labor aggregator = (1+ξ)/ξ")

    # 2) Idiosyncratic productivity
    m <= parameter(:ρ_S, 0.99, fixed = true, description = "Persistence of productivity shocks", tex_label = "\\rho_S")
    m <= parameter(:σ_S, 0.0282, fixed = true, description = "Standard deviation of productivity shocks", tex_label = "\\sigma_S")
    m <= parameter(:in, 0.001, fixed = true, description = "Prob. worker -> entrepreneur")
    m <= parameter(:out, 0.20, fixed = true, description = "Prob. entrepreneur -> worker")
    m <= parameter(:Eshare, m[:in].value / (m[:in].value + m[:out].value), fixed = true,
                   description = "Entrepreneur share")
    m <= parameter(:in_s, 0.05, fixed = true, description = "Prob. becoming high-skilled")
    m <= parameter(:out_s, 0.10, fixed = true, description = "Prob. high-skilled -> worker")
    m <= parameter(:in_l, 0.05, fixed = true, description = "Prob. becoming low-skilled")
    m <= parameter(:out_l, 0.10, fixed = true, description = "Prob. low-skilled -> worker")
    m <= parameter(:super_low, 0.1811652190, fixed = true, description = "Lowest income grid point")
    m <= parameter(:super_high, 5.4424929457, fixed = true, description = "Highest income grid point")

    # 3) Intermediate goods firms
    m <= parameter(:η, 3.0, fixed = true, description = "η", tex_label = "\\eta")
    m <= parameter(:π_bar, 1.005, fixed = true, description = "Steady state inflation rate", tex_label = "\\bar{\\pi}")
    m <= parameter(:Eratio, 0.3, fixed = true, description = "Entrepreneur wealth ratio", tex_label = "\\mathrm{Eratio}")
    m <= parameter(:fix_ratio, 0.201538837260750, fixed = true, description = "Fixed cost ratio")
    m <= parameter(:fix, 0.0, fixed = true, description = "Fixed cost")
    m <= parameter(:b_share, 0.0, fixed = true, tex_label = "b_{\\mathrm{share}}")
    m <= parameter(:b_aux, 1.6335, fixed = true, description = "Exponent on bonus component of worker income")
    m <= parameter(:ζ, 0.401, fixed = true, description = "ζ: CES weight on capital", tex_label = "\\zeta")
    m <= parameter(:ρ_ces, -0.495, fixed = true, description = "ρ: CES elasticity parameter", tex_label = "\\rho_{\\mathrm{CES}}")
    m <= parameter(:a_share, 0.4, fixed = true, description = "Capital share in production")
    m <= parameter(:w_share, 0.35, fixed = true, description = "Labor weight in production")
    m <= parameter(:Z_ss, 3.0, fixed = true, description = "Steady-state TFP level", tex_label = "Z")

    # 4) Capital service
    m <= parameter(:v_util, 1.0, fixed = true, description = "Steady-state utilization rate")
    m <= parameter(:δ_ss, 0.01, fixed = true, description = "Steady-state depreciation rate", tex_label = "\\delta_{ss}")

    # 5) Labor agencies
    m <= parameter(:u_1, 0.0215, fixed = true, description = "Unemployment rate market 1")
    m <= parameter(:v_f_1, 0.710, fixed = true, description = "Vacancy fill rate market 1")
    m <= parameter(:f_1, 0.85, fixed = true, description = "Job finding rate market 1")
    m <= parameter(:ι_1, 0.05, fixed = true, description = "ι_1: Matching elasticity market 1", tex_label = "\\iota_1")
    m <= parameter(:λ_1, 0.022, fixed = true, description = "λ_1: Separation rate market 1", tex_label = "\\lambda_1")
    m <= parameter(:fix_L_1, 4.643476652503947e-4, fixed = true, description = "Fixed cost vacancy market 1")
    m <= parameter(:u_2, 0.054, fixed = true, description = "Unemployment rate market 2")
    m <= parameter(:v_f_2, 0.312, fixed = true, description = "Vacancy fill rate market 2")
    m <= parameter(:f_2, 0.664, fixed = true, description = "Job finding rate market 2")
    m <= parameter(:ι_2, 0.3, fixed = true, description = "ι_2: Matching elasticity market 2", tex_label = "\\iota_2")
    m <= parameter(:λ_2, 0.055, fixed = true, description = "λ_2: Separation rate market 2", tex_label = "\\lambda_2")
    m <= parameter(:fix_L_2, 4.643476652503947e-4, fixed = true, description = "Fixed cost vacancy market 2")
    m <= parameter(:earning_ratio, 2.016, fixed = true, description = "Earnings ratio market 1 / market 2")

    # 6) Fiscal
    m <= parameter(:τ_w, 0.30, fixed = true, description = "τ_w: Tax rate on wages", tex_label = "\\tau_w")
    m <= parameter(:τ_a, 0.30, fixed = true, description = "τ_a: Tax rate on assets", tex_label = "\\tau_a")
    m <= parameter(:τ_b, 0.0, fixed = true, description = "τ_b: Tax rate on bonds", tex_label = "\\tau_b")
    m <= parameter(:b_ratio, 0.4, fixed = true, description = "Replacement ratio for UB")
    m <= parameter(:LT, 0.260227733085066, fixed = true, description = "Transfers-to-output ratio")
    m <= parameter(:B_gov_ratio, 0.6, fixed = true, description = "Government debt to output ratio")

    # 7) Monetary / financial
    m <= parameter(:R_cb_ss, 1.0075, fixed = true, description = "Steady-state policy rate")
    m <= parameter(:π_cb, m[:π_bar].value, fixed = true, description = "Central bank inflation target", tex_label = "\\pi_{cb}")
    m <= parameter(:Rprem, 0.038285657, fixed = true, description = "Risk premium")
    m <= parameter(:b_min, -1.3006141532, fixed = true, description = "Borrowing constraint", tex_label = "b_{\\min}")
    m <= parameter(:A_g_ratio, 0.05, fixed = true, description = "Central bank asset share")
    m <= parameter(:θ_b, 0.97, fixed = true, description = "θ_b: Parameter", tex_label = "\\theta_b")
    m <= parameter(:Θ, 3.0, fixed = true, description = "Steady-state leverage ratio", tex_label = "\\Theta")
    m <= parameter(:Ab_ratio, 0.271426003778368, fixed = true, description = "Bank equity share")
    m <= parameter(:b_a_aux, 0.994444444444444, fixed = true, description = "", tex_label = "b_{\\text{aux}}")
    m <= parameter(:b_a_aux2, 1.0, fixed = true, description = "Extra cost per unit of central bank asset purchases", tex_label = "\\varphi_{\\tau_cb}")
    m <= parameter(:β_b, m[:π_cb].value / m[:R_cb_ss].value * (1 / (1 - m[:dr].value)) * m[:b_a_aux].value,
                   fixed = true, description = "Banker discount factor", tex_label = "\\beta_b")
    m <= parameter(:B_MMMF_ratio, 0.783074980718057, fixed = true, description = "MMMF deposit share")
    m <= parameter(:q_ss, 1.0, fixed = true, description = "Steady-state Tobin''s q")
    m <= parameter(:τ_cp, 0.0, fixed = true, description = "Extra cost per unit of central bank asset purchases", tex_label = "\\tau_{cp}")
    m <= parameter(:β_L, m[:β].value, fixed = true, description = "Labor-side discount factor")
    m <= parameter(:frac_b, 0.0, fixed = true, description = "Fraction of banker households", tex_label = "\\text{frac}_b")

    # Other fixed parameters retained for existing kCSC code paths.
    m <= parameter(:ε_w, 1.0, fixed = true, description = "ε_w: Parameter", tex_label = "\\varepsilon_w")
    m <= parameter(:ϕ_x, 0.0, fixed = true, description = "ϕ_x: Parameter", tex_label = "\\varphi_x")
    m <= parameter(:ρ_ψ_w, 0.0, fixed = true, description = "ρ_ψ_w: Parameter", tex_label = "\\rho_{\\psi_w}")
    m <= parameter(:ϕ_var, 0.0, fixed = true, description = "ϕ_var: Parameter", tex_label = "\\varphi")
    m <= parameter(:α_lk, 0.0, fixed = true, description = "α_lk: Parameter", tex_label = "\\alpha_{lk}")
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
    m <= parameter(:psi_cp, 0.0, fixed = true, description = "Equity payout share of financial intermediaries", tex_label = "\\psi_{cp}")
    m <= parameter(:ω, 0.007975547917397408, fixed = true, description = "")
    m <= parameter(:σ_s, 0.03, fixed = true, description = "Standard deviation of productivity shock")
    m <= parameter(:ϕ, 50.01728408708132, fixed = true, description = "", tex_label = "\\phi")
    m <= parameter(:λ_aux, 0.994966466098559, fixed = true, description = "", tex_label = "\\lambda_aux")
    m <= parameter(:λ_b_aux, 0.996816728468132, fixed = true, description = " ", tex_label = "\\lambda_aux")
    m <= parameter(:w_bar, 1.2111925176640317, fixed = true, description = "")
    m <= parameter(:θ_2, 0.73, fixed = true, description = "", tex_label = "")
    m <= parameter(:θ, 0.27, fixed = true, description = "", tex_label = "")
    m <= parameter(:α_ll, 0.0, fixed = true, description = "", tex_label = "")
    m <= parameter(:δ_1, 1.0025, fixed = true, description = "", tex_label = "")
    m <= parameter(:α_kk, 0.0, fixed = true, description = "", tex_label = "")
    m <= parameter(:ψ, 0.785159222538979, fixed = true, tex_label = "b_{\\mathrm{share}}")
    m <= parameter(:α, 1.7127143830661984, fixed = true, description = "", tex_label = "")
    m <= parameter(:ϕ_b, 0.0, fixed = true, description = "", tex_label = "")
    m <= parameter(:ρ_passive_QE, 0.95, fixed = true, description = "", tex_label = "")
    m <= parameter(:MMF_ratio_1, 10.0, fixed = true, description = "")
    m <= parameter(:ρ_A_g, 0.95, fixed = true, description = "")
    m <= parameter(:u_target, 0.058, fixed = true, description = "Target unemployment rate")
    m <= parameter(:v_f_target, 0.7, fixed = true, description = "Target vacancy fill rate")
    m <= parameter(:λ_sep, 0.03, fixed = true, description = "Separation rate")
    m <= parameter(:wmarkdown, 1.0, fixed = true, description = "Wage markdown")
    m <= parameter(:B_F_ratio, 0.5, fixed = true, description = "Fiscal liability share of government debt")
    m <= parameter(:A_F_ratio, 0.0, fixed = true, description = "FI asset share")
    m <= parameter(:K_ss_init, 30.0, fixed = true, description = "Initial steady-state capital guess")
    m <= parameter(:d, 0.6738, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.15), fixed = true, description = "d:", tex_label = "d")
    m <= parameter(:d_1, m[:d].value, fixed = true, description = "Wage indexation market 1")
    m <= parameter(:d_2, m[:d].value, fixed = true, description = "Wage indexation market 2")
    m <= parameter(:alpha_1, 1.0, fixed = true, description = "Matching function curvature market 1")
    m <= parameter(:alpha_2, 1.0, fixed = true, description = "Matching function curvature market 2")
    m <= parameter(:GAMMA, 0.0, fixed = true, description = "Inflation indexation parameter")
    m <= parameter(:w_bar_1, m[:w_bar].value, fixed = true, description = "Steady-state wage market 1")
    m <= parameter(:w_bar_2, m[:w_bar].value, fixed = true, description = "Steady-state wage market 2")
    m <= parameter(:psi_1, 1.0, fixed = true, description = "Labor disutility scale market 1")
    m <= parameter(:psi_2, 1.0, fixed = true, description = "Labor disutility scale market 2")
    m <= parameter(:ρ_ZZ_1, 0.9, fixed = true, description = "AR(1) coefficient for ZZ_1", tex_label = "\\rho_{ZZ,1}")
    m <= parameter(:ρ_ZZ_2, 0.9, fixed = true, description = "AR(1) coefficient for ZZ_2", tex_label = "\\rho_{ZZ,2}")
    m <= parameter(:ρ_ZZ_3, 0.9, fixed = true, description = "AR(1) coefficient for ZZ_3", tex_label = "\\rho_{ZZ,3}")
    m <= parameter(:ρ_ZZ_4, 0.9, fixed = true, description = "AR(1) coefficient for ZZ_4", tex_label = "\\rho_{ZZ,4}")
    m <= parameter(:ρ_w_1, 0.8916942990519234, fixed = true, description = "AR(1) coefficient for wage process market 1", tex_label = "\\rho_{w,1}")
    m <= parameter(:ρ_w_2, 0.8916942990519234, fixed = true, description = "AR(1) coefficient for wage process market 2", tex_label = "\\rho_{w,2}")
    m <= parameter(:σ_ZZ_1, 0.0, fixed = true, description = "Standard deviation for ZZ_1 shock", tex_label = "\\sigma_{ZZ,1}")
    m <= parameter(:σ_ZZ_2, 0.0, fixed = true, description = "Standard deviation for ZZ_2 shock", tex_label = "\\sigma_{ZZ,2}")
    m <= parameter(:σ_ZZ_3, 0.0, fixed = true, description = "Standard deviation for ZZ_3 shock", tex_label = "\\sigma_{ZZ,3}")
    m <= parameter(:σ_ZZ_4, 0.0, fixed = true, description = "Standard deviation for ZZ_4 shock", tex_label = "\\sigma_{ZZ,4}")

    # Estimated parameters
    m <= parameter(:κ, 0.0276503, (1e-5, 5.0), (1e-5, 5.0), SquareRoot(), GammaAlt(0.1, 0.02), fixed = false,
                   description = "κ: The slope of the Phillips curve", tex_label = "\\kappa")
    # m <= parameter(:κ, 0.05247664549755819, (1e-5, 5.0), (1e-5, 5.0), SquareRoot(), GammaAlt(0.1, 0.02), fixed = false,
    #                description = "κ: The slope of the Phillips curve", tex_label = "\\kappa")
    m <= parameter(:ρ_w, 0.8916942990519234, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_w: AR(1) coefficient in the wage process.",
                   tex_label = "\\rho_w")
    m <= parameter(:δ, 0.3267222276512135, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.15), fixed = false,
                   description = "δ:",
                   tex_label = "\\delta")
    #=
    m <= parameter(:ϕ, 49.7917, (1e-5, 10.0), (1e-5, 10.0), ModelConstructors.Exponential(),
                   Normal(3.0, 0.5), fixed = false,
                   description = "ϕ: Parameter in monetary policy rule.",
                   tex_label = "\\phi")
    =#
    m <= parameter(:ϕ_π, 1.532296337149513, (1e-5, 10.0), (1e-5, 10.0), ModelConstructors.Exponential(),
                   Normal(1.7, 0.3), fixed = false,
                   description = "ϕ_π: Weight on inflation gap in monetary policy rule.",
                   tex_label = "\\varphi_\\pi")
    m <= parameter(:ϕ_u, 0.37475243192750335, (-0.5, 0.5), (-0.5, 0.5), Untransformed(),
                   Normal(0.1, 0.05), fixed = false,
                   description = "ϕ_u: Weight on unemployment gap in monetary policy rule",
                   tex_label = "\\varphi_u")
    m <= parameter(:ρ_R, 0.8818054913182859, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_R: AR(1) coefficient in the monetary policy rule.",
                   tex_label = "\\rho_R")

    m <= parameter(:ρ_BB, 0.9983489940166279, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_BB: AR(1) coefficient in the BB process.",
                   tex_label = "\\rho_{BB}")
    m <= parameter(:ρ_B, 0.5057547780246887, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_B: AR(1) coefficient in the intertemporal preference shifter process.",
                   tex_label = "\\rho_B")
    m <= parameter(:ρ_D, 0.8440051981700979, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_D: AR(1) coefficient in the D process.",
                   tex_label = "\\rho_D")
    m <= parameter(:ρ_G, 0.7402477481386883, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_G: AR(1) coefficient in the government spending process.",
                   tex_label = "\\rho_G")
    m <= parameter(:ρ_Z, 0.9516176031961177, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_Z: AR(1) coefficient in the technology process.",
                   tex_label = "\\rho_Z")
    m <= parameter(:ρ_η, 0.8944669355937231, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_η: AR(1) coefficient in the η process.",
                   tex_label = "\\rho_\\eta")
    m <= parameter(:ρ_ι, 0.9598287691285154, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_ι: AR(1) coefficient in the ι process.",
                   tex_label = "\\rho_\\iota")
    m <= parameter(:ρ_B_F, 0.9175878755308129, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.2), fixed = false,
                   description = "ρ_B_F: AR(1) coefficient in the B_F process.",
                   tex_label = "\\rho_{B_F}")

    m <= parameter(:σ_D, 0.12371, (1e-8, 50.0), (1e-8, 50.0), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_D: The standard deviation of the D process.",
                   tex_label = "\\sigma_D")
    m <= parameter(:σ_G, 0.0068107, (1e-8, 5.0), (1e-8, 5.0), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_G: The standard deviation of the government spending process.",
                   tex_label = "\\sigma_G")
    m <= parameter(:σ_R, 0.0010768, (1e-8, 5.0), (1e-8, 5.0), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_R: The standard deviation of the monetary policy shock.",
                   tex_label = "\\sigma_R")
    m <= parameter(:σ_Z, 0.0079954, (1e-8, 5.0), (1e-8, 5.0), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_Z: The standard deviation of the technology process.",
                   tex_label = "\\sigma_Z")
    m <= parameter(:σ_η, 0.039461, (1e-8, 20.0), (1e-8, 20.0), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_η: The standard deviation of the η process.",
                   tex_label = "\\sigma_\\eta")
    m <= parameter(:σ_ι, 0.00090722, (1e-8, 5.0), (1e-8, 5.0), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_ι: The standard deviation of the ι process.",
                   tex_label = "\\sigma_\\iota")
    m <= parameter(:σ_w, 0.031033, (1e-8, 20.0), (1e-8, 20.0), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_w: The standard deviation of the wage process.",
                   tex_label = "\\sigma_w")
    m <= parameter(:σ_BB, 0.0019753, (1e-8, 5.0), (1e-8, 5.0), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_BB: The standard deviation of the BB process.",
                   tex_label = "\\sigma_{BB}")
    m <= parameter(:σ_B_F, 0.010128, (1e-8, 5.0), (1e-8, 5.0), ModelConstructors.Exponential(),
                   RootInverseGamma(2, 0.02), fixed = false,
                   description = "σ_B_F: The standard deviation of the B_F process.",
                   tex_label = "\\sigma_{B_F}")

    m <= parameter(:γ, 0.18919, (1e-5, 1 - 1e-5), (1e-5, 1 - 1e-5), SquareRoot(),
                   BetaAlt(0.5, 0.15), fixed = false,
                   description = "γ: The log of the steady-state growth rate of technology",
                   tex_label = "\\gamma")
    m <= parameter(:ι, 0.0542, (1e-5, 5.0), (1e-5, 5.0), SquareRoot(),
                   GammaAlt(0.5, 0.2), fixed = false,
                   description = "ι: Parameter",
                   tex_label = "\\iota")
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
