mutable struct OnionModel{T} <: AbstractRepModel{T}
    parameters::Vector{AbstractParameter{T}}
    steady_state::ParameterVector{T}
    keys::OrderedDict{Symbol, Int}

    endogenous_states::OrderedDict{Symbol, Int}
    exogenous_shocks::OrderedDict{Symbol, Int}
    expected_shocks::OrderedDict{Symbol, Int}
    equilibrium_conditions::OrderedDict{Symbol, Int}
    endogenous_states_augmented::OrderedDict{Symbol, Int}
    observables::OrderedDict{Symbol, Int}
    pseudo_observables::OrderedDict{Symbol, Int}

    spec::String
    subspec::String
    settings::Dict{Symbol, Setting}
    test_settings::Dict{Symbol, Setting}
    rng::MersenneTwister
    testing::Bool

    observable_mappings::OrderedDict{Symbol, Observable}
    pseudo_observable_mappings::OrderedDict{Symbol, PseudoObservable}
end


description(m::OnionModel) = "December 2024 Briefing Model, Inflation Networks, subspec: $(m.subspec)"

function Base.show(io::IO, m::OnionModel)
    @printf io "OnionModel, spec %s\n" subspec(m)
    @printf io "description: %s\n" description(m)
end

function OnionModel(subspec::String = "ss1";
                    custom_settings::Array{S} where S<:Setting = Array{Setting{Bool}}(undef, 0),
                    testing = false)

    spec             = split(basename(@__FILE__), '.')[1]
    subspec          = subspec
    settings         = Dict{Symbol, Setting}()
    test_settings    = Dict{Symbol, Setting}()
    rng              = MersenneTwister(0)

    m = OnionModel{Float64}(
            # model parameters and steady state values
        Vector{AbstractParameter{Float64}}(),
        Vector{Float64}(),
        OrderedDict{Symbol,Int}(),

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

    for setting in custom_settings
        m <= setting
    end

    init_settings!(m)
    init_observable_mappings!(m)
    init_pseudo_observable_mappings!(m)
    init_model_indices!(m)
    init_parameters!(m)
    init_subspec!(m)
    steadystate!(m)

    return m
end



function init_settings!(m::OnionModel)
    # For parsing model subspec to Int
    subspec_int = parse(Int, subspec(m)[3:end])

    if subspec_int == 0
        m <= Setting(:n_sectors, 396)
    elseif subspec_int >= 1
        m <= Setting(:n_sectors, 69)

        energy_sectors = [3,4,5,8,26]
        food_sectors   = [1,21]
        core_services = 42:69
        core_goods = setdiff(1:69, vcat(energy_sectors, core_services, food_sectors))
        core = vcat(core_services, core_goods)

        # Settings previously in spec files
        m <= Setting(:irf_type, "oil")
        m <= Setting(:expected_ffr, [])
        m <= Setting(:cond_semi_names, [])
        m <= Setting(:cond_full_names, [])
        m <= Setting(:solution_method, :gensys)
        m <= Setting(:T, 52)
        m <= Setting(:forecast_horizons, 40)
        m <= Setting(:impulse_response_horizons, 40)
        m <= Setting(:use_parallel_workers, true)
        m <= Setting(:sectoral_nan, true)
        m <= Setting(:n_smc_blocks, 1)
        m <= Setting(:sampling_method, :SMC)
        m <= Setting(:subgroup_names,
                     OrderedDict{String, Symbol}("core_goods" => :CUSR0000SACL1E,
                                                 "core_services" => :CUSR0000SASLE,
                                                 "cpi_energy" => :CPIENGSL,
                                                 "core_foods" => :NOTHING))
        m <= Setting(:subgroup_to_sector,
                     OrderedDict{String, Array{Int64,1}}("core_goods" => core_goods,
                                                         "core_services" => core_services,
                                                         "cpi_energy" =>  energy_sectors,
                                                         "core_foods" => food_sectors))
    end

    # Sectoral Discrimination
    m <= Setting(:energy_sectors, energy_sectors)
    m <= Setting(:food_sectors, food_sectors)
    m <= Setting(:core_service_sectors, core_services)
    m <= Setting(:core_goods_sectors, core_goods)
    m <= Setting(:core_sectors, core)
    m <= Setting(:no_data_sectors, [2, 7, 9, 25, 38, 39, 41, 47, 49, 51, 54, 56, 65])


    # Relevant for Model Building (sizes of matrices and such)
    m <= Setting(:n_back_states, get_setting(m, :n_sectors)+3)
    m <= Setting(:n_jump_states, get_setting(m, :n_sectors)+3)
    m <= Setting(:n_exo_states, 1)
    #m <= Setting(:n_model_states, get_setting(m, :n_back_states) + get_setting(m, :n_jump_states))


    # Relevant for filepaths to read data and save figures
    m <= Setting(:param_data_root, "/data/dsge_data_dir/proc/dsge/briefings/202412/Model_Data/")
    m <= Setting(:dataroot, "/data/dsge_data_dir/proc/dsge/briefings/202412/Model_Data/input_data/")
    m <= Setting(:saveroot, "/data/dsge_data_dir/proc/dsge/briefings/202412/output_data/")
    m <= Setting(:forecast_input_file_overrides, Dict{Symbol, String}())


    # Relevant for reading and formatting data
    m <= Setting(:cond_id, 2)
    m <= Setting(:data_id, 3)
    m <= Setting(:use_population_forecast, true)
    m <= Setting(:hpfilter_population, true)
    m <= Setting(:date_presample_start, quartertodate("1959-Q3"))
    m <= Setting(:date_mainsample_start, quartertodate("1960-Q1"))
    m <= Setting(:population_mnemonic, Nullable(:CNP16OV__FRED),
        "Mnemonic of FRED data series for computing per-capita values (a Nullable{Symbol})")
    m <= Setting(:data_quarter_or_month, :quarter)
    sectoral_inflation_path = get_setting(m, :dataroot) * "sector_inflation_" * (get_setting(m, :data_quarter_or_month) == :quarter ? "quarterly" : "monthly") * ".csv"
    m <= Setting(:sector_names, (names(CSV.read(sectoral_inflation_path, DataFrame))[3:end])[Not([69,70,72,73])])

    # Relevant for other things such as IRFs, smoothing, and forecasting
    m <= Setting(:n_mon_anticipated_shocks, 0)
    m <= Setting(:forecast_smoother, :durbin_koopman)
    m <= Setting(:forecast_uncertainty_override, Nullable{Bool}())
    m <= Setting(:shockdec_startdate, Nullable{Bool}())
    m <= Setting(:shockdec_enddate, Nullable{Bool}())
end



function init_parameters!(m::OnionModel)
    # For parsing model subspec to Int
    subspec_int = parse(Int, subspec(m)[3:end])

    if subspec_int == 0
        m <= parameter(:bet, 0.96^(1/12), fixed = true,
                       description="β: temporal discount",
                       tex_label="\\beta")

        m <= parameter(:ρ_τ, 1.22, fixed = true,
                       description="ρ_τ: ",
                       tex_label="\\rho_\\tau")

        m <= parameter(:ρ_τ2, -0.2475, fixed = true,
                       description="ρ_τ2: ",
                       tex_label="\\rho_\\tau2")

        m <= parameter(:ρ_i, 0.85^(1/3), fixed = true,
                       description="ρ_i: inertia",
                       tex_label="\\rho_i")

        m <= parameter(:η, 0.6, fixed = true,
                       description="η: ",
                       tex_label="\\eta")

        m <= parameter(:ζ, 2.0, fixed = true,
                       description="ζ:",
                       tex_label="\\zeta")

        m <= parameter(:ν, 0.2, fixed = true,
                       description="ν: ",
                       tex_label="\\nu")

        m <= parameter(:ξ, 0.1, fixed = true,
                       description="ξ: ",
                       tex_label="\\xi")

        m <= parameter(:mp_cpi_infl, 1.01, fixed = true,
                       description="weight on cpi inflation in mp rule",
                       tex_label="mp_cpi_infl")

        m <= parameter(:mp_cons, 0.0, fixed = true,
                       description="weight on consumption in mp rule",
                       tex_label="mp_cons")

        m <= parameter(:mp_cstar, 0.0, fixed = true,
                       description="weight on potential consumption",
                       tex_label="mp_cstar")

        m <= parameter(:invkapw,740.3277, fixed = true,
                       description="inverse kappaw",
                       tex_label="invkawp")

        m <= parameter(:oil, 14., fixed = true,
                       description = "Index of oil",
                       tex_label="oil")

        m <= parameter(:gas, 15., fixed = true,
                       description = "Index of gas",
                       tex_label="gas")

        m <= parameter(:coal, 16., fixed = true,
                       description = "Index of coal",
                       tex_label="coal")

        #Setting all of these to 1 just to have values down -- to be filled later

        #placeholder = fill(1.0, 396)
        placeholder_dist = Product(fill(Uniform(1.0, 1.1), get_setting(m, :n_sectors)))

        if irf_type == "oil"
            param_path = get_setting(m, :param_data_root) * "matlab_params_oil.jld2"
        elseif irf_type == "taylor"
            param_path = get_setting(m, :param_data_root) * "matlab_params.jld2"
        end

        fl = JLD2.jldopen(param_path, "r")
        InOut = fl["IO"]
        InOut2 = fl["IO2"]
        inpshare = fl["inpshare"]
        #Let the input output matrix be a setting given this won't get estimated, and creating a parameter which holds a matrix isn't worthwhile
        m <= Setting(:IO, InOut)
        #m[:IO] = InOut #Feels dubious but maybe?
        m <= Setting(:IO2, InOut2)
        #m[:IO2] = IO2 #Feels dubious but maybe?

        #Another matrix, just setting here for convenience. Will fix later? -BP
        m <= Setting(:ω_tilde, fl["omega_tilde"])
        m <= Setting(:ωE_tilde, fl["omegaE_tilde"])
        m <= Setting(:ωN_tilde, fl["omegaN_tilde"])
        m <= Setting(:inpshare, fl["inpshare"])

        m <= Setting(:labshare, vec(fl["labshare"]))
        m <= Setting(:taxshare, vec(fl["taxshare"]))
        m <= Setting(:tointshare, vec(fl["tointshare"]))
        m <= Setting(:int_totout, vec(fl["ind_totout"]))
        m <= Setting(:invkap, vec(fl["invkap"]))
        m <= Setting(:food, vec(fl["food"]))
        m <= Setting(:core, vec(fl["core"]))
        m <= Setting(:energy, vec(fl["energy"]))
        m <= Setting(:gam, vec(fl["gam"])) #Expenditure shares
        m <= Setting(:gam_core, vec(fl["gam_core"]))
        m <= Setting(:gam_food, vec(fl["gam_food"]))
        m <= Setting(:gam_energy, vec(fl["gam_energy"]))
        m <= Setting(:gam_services, vec(fl["gam_services"]))
        m <= Setting(:gam_coreservices, vec(fl["gam_coreservices"]))
        m <= Setting(:gam_goods, vec(fl["gam_goods"]))
        m <= Setting(:gam_coregoods, vec(fl["gam_coregoods"]))
        m <= Setting(:pc_px, vec(fl["pc_px"]))
        m <= Setting(:to_mx, vec(fl["to_mx"]))
        m <= Setting(:ei, vec(fl["ei"]))
        m <= Setting(:ς_tilde, vec(fl["varsig_tilde"]))


    elseif subspec_int >= 1

        paras = InOutData()

        #=
        m <= parameter(:bet, paras["β"], fixed = true,
                       description="β: temporal discount",
                       tex_label="\\beta")
        =#

        m <= parameter(:bet, paras["β"], (1e-5, 10.), (1e-5, 10.), ModelConstructors.Exponential(), GammaAlt(0.25, 0.1), fixed = true, scaling = x -> 1/(1 + x/100),
                       description="β: temporal discount",
                       tex_label="\\beta")

        m <= parameter(:ρ_τ, 1.22, fixed = true,
                       description="ρ_τ: ",
                       tex_label="\\rho_\\tau")

        m <= parameter(:ρ_τ2, -0.2475, fixed = true,
                       description="ρ_τ2: ",
                       tex_label="\\rho_\\tau2")
#=
m <= parameter(:ρ_i, 0.85^(1/3), fixed = true,
               description="ρ_i: policy inertia",
               tex_label="\\rho_i")
=#

m <= parameter(:ρ_i, 0.85^(1/3), (1e-5, 0.999), (1e-5, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.75, 0.10), fixed = true,
               description="ρ_i: policy inertia",
               tex_label="\\rho_i")

m <= parameter(:η, paras["η"], fixed = true,
               description="η: Elasticity of substitution between labor and intermediate inputs.",
               tex_label="\\eta")

m <= parameter(:ζ, paras["ζ"], fixed = true,
               description="ζ: Elasticity of substitution in CES consumption. ",
               tex_label="\\zeta")

m <= parameter(:ν, paras["ν"], fixed = true,
               description="ν: Elasticity of substitution between energy and non-energy inputs.",
               tex_label="\\nu")

m <= parameter(:ξ, paras["ξ"], fixed = true,
               description="ξ: Elasticity of substitution between intermediate inputs.",
               tex_label="\\xi")

m <= parameter(:mp_cpi_infl, 1.01, (1e-5, 10.), (1e-5, 10.00), ModelConstructors.Exponential(), Normal(1.5, 0.25), fixed=false, #paras["mp_cpi_infl"] NEEDS TO BE 1.01 to AVOID EIGENVALUE ISSUE
               description="weight on cpi inflation in mp rule",
               tex_label="mp_cpi_infl")

m <= parameter(:mp_cons, 0., (-0.5, 0.5), (-0.5, 0.5), ModelConstructors.Untransformed(), Normal(0.12, 0.05), fixed = false,
               description="weight on consumption in mp rule",
               tex_label="mp_cons")

m <= parameter(:mp_cstar, 0.,  (-0.5, 0.5), (-0.5, 0.5), ModelConstructors.Untransformed(), Normal(0.12, 0.05), fixed = false,
               description="weight on potential consumption",
               tex_label="mp_cstar")

m <= parameter(:invkapw,paras["invkapw"], fixed = true,
               description="inverse kappaw",
               tex_label="invkawp")

m <= Setting(:oil, Int(paras["oil"]))
m <= Setting(:gas, Int(paras["gas"]))
m <= Setting(:coal, Int(paras["coal"]))

m <= parameter(:oil, paras["oil"], fixed = true,
               description = "Index of oil",
                   tex_label="oil")

m <= parameter(:gas, paras["gas"], fixed = true,
               description = "Index of gas",
               tex_label="gas")

m <= Setting(:coal, paras["coal"])


#Setting all of these to 1 just to have values down -- to be filled later

    #placeholder = fill(1.0, 396)
    placeholder_dist = Product(fill(Uniform(1.0, 1.1), 396))


    #fl = JLD2.jldopen("/data/dsge_data_dir/proc/dsge/briefings/202412/Model_Data/matlab_params_oil.jld2", "r")
    InOut = paras["IO"]
    InOut2 = paras["IO2"]
    inpshare = paras["inpshare"]
    #Let the input output matrix be a setting given this won't get estimated, and creating a parameter which holds a matrix isn't worthwhile
    m <= Setting(:IO, InOut)
    m <= Setting(:IO2, InOut2)


#Another matrix, just setting here for convenience. Will fix later? -BP
m <= Setting(:ω_tilde, paras["ω_tilde"])
m <= Setting(:ωE_tilde, paras["ωE_tilde"])
m <= Setting(:ωN_tilde, paras["ωN_tilde"])

m <= Setting(:inpshare, paras["inpshare"])
m <= Setting(:labshare, vec(paras["labshare"]))

m <= Setting(:taxshare, vec(paras["taxshare"]))
m <= Setting(:totintshare, vec(paras["totintshare"]))
m <= Setting(:int_totout, vec(paras["ind_totout"]))
m <= Setting(:invkap, vec(paras["invkap"]))
m <= Setting(:food, float(vec(paras["food"])))
m <= Setting(:core, float(vec(paras["core"])))
m <= Setting(:energy, float(vec(paras["energy"])))
m <= Setting(:gam, vec(paras["gam"]))
m <= Setting(:gam_core, vec(paras["gam_core"]))
m <= Setting(:gam_food, vec(paras["gam_food"]))
m <= Setting(:gam_energy, vec(paras["gam_energy"]))
m <= Setting(:pc_px, vec(paras["pc_px"]))
m <= Setting(:to_mx, vec(paras["to_mx"]))
m <= Setting(:ei, vec(paras["ei"]))
m <= Setting(:ς_tilde, vec(paras["varsig_tilde"]))


## Adding model parameters for standard deviation of shocks

### Shared across all subspecs
m <= parameter(:σ_c, 0.8719, (1e-5, 10.), (1e-5, 10.), ModelConstructors.Exponential(), Normal(1.5, 0.37), fixed=false,
               description = "σ_c: Coefficient of relative risk aversion",
               tex_label = "\\sigma_c")
m <= parameter(:σ_b_t, 0.0292, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
               description = "σ_b: Standard deviation of the discount rate process",
               tex_label = "\\sigma_{b_t}")
m <= parameter(:ρ_b_t, 0.941, (1e-5, 0.999), (1e-5, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
               description = "ρ_b: AR(1) coefficient of the discount rate process",
               tex_label = "\\rho_{b_t}")

m <= parameter(:σ_μw, 0.1314, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
               description = "σ_wμ: standard deviation of wage mark up shock process",
               tex_label = "\\sigma_{\\mu w}")
m <= parameter(:ρ_μw, 0.3884, (1e-5, 0.999), (1e-5, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
               description = "ρ_μw: AR(1) coefficient in the wage mark up shock process",
               tex_label = "\\rho_{\\mu w}")

m <= parameter(:σ_πstar, 0.0269, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(6, 0.03), fixed=false,
               description = "σ_πstar: standard deviation of the process describing the time varying inflation target",
               tex_label = "\\sigma_{\\pi^\\star}")
m <= parameter(:ρ_πstar, 0.99, (1e-5, 0.999), (1e-5, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=true,
               description = "ρ_πstar: AR(1) coefficient of process describing the time varying inflation target",
               tex_label = "\\rho_{\\pi^\\star}")

m <= parameter(:σ_a_t, 0.6742, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false, #Taken from std dev of stationary comp of prod
               description = "σ_a_t: standard deviation of the process describing productivity",
               tex_label = "\\sigma_{a_t}")

m <= parameter(:ρ_a_t, 0.9446,  (0., 0.999), (1e-5, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,#Taken from std dev of stationary comp of prod 0.9446
               description = "ρ_a_t: AR(1) coefficient of the process describing productivity",
               tex_label = "\\rho_{a_t}")

m <= parameter(:h, 0.5347,  (1e-5, 0.999), (1e-5, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.7, 0.1), fixed=false,
               description = "h: consumption habit persistence",
               tex_label="h")

m <= parameter(:γ, 0.0, (-5.0, 5.0), (-5., 5.), ModelConstructors.Untransformed(), Normal(0.4, 0.1), fixed=false,
               scaling = x -> x/100, #Growth rate of economy
               description = "γ: Log of the steady-state growth rate of technology",
               tex_label="\\gamma")
#=
m <= parameter(:mp_habit, 0.0, fixed = true,
               description = ":mp_habit: weight of MP rule on habit formation",
               tex_label="mp-habit")
=#

m <= parameter(:mp_habit, 0.0, (-0.5, 0.5), (-0.5, 0.5), ModelConstructors.Untransformed(), Normal(0.12, 0.05), fixed = false,
               description = ":mp_habit: weight of MP rule on habit formation",
               tex_label="mp-habit")


m <= parameter(:σ_r_m, 0.2380, (0.0, 5.), (0.0, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
               description = "Standard deviation of process describing iid monetary policy shock",
               tex_label="\\sigma_{r^m}")

#= we don't use for now (measurement errors)
m <= parameter(:ρ_meas_πc, 0.0, fixed = true,
               description = "AR(1) coefficient for CPI inflation measurement error process")

m <= parameter(:σ_meas_πc, 0.0999, fixed = true,
               description = "AR(1) coefficient for CPI inflation measurement error process")
=#

#==============
Subspec specific parameter additions
==============#

if subspec_int ∈ [0, 1]

    #1) Add common trend persistence for briefing model
    m <= parameter(:ρ_μ_trend, 0.8827)

    #2) Add common markup shock std and AR(1) coefficient
    m <= parameter(:σ_μ, 0.1314, fixed = true,
                   description = "σ_μ: standard deviation of mark up shock process")
    m <= parameter(:ρ_μ, 0.8827, fixed = true,
                   description = "ρ_μ: AR(1) coefficient of the mark up shock process")
    m <= parameter(:π_star, 0.5, fixed = true,
                   description = "Steady state rate of inflation")
else # ss ∈ [2, 3, ...]

    for i in collect(keys(get_setting(m, :subgroup_names)))
        #1) Differentiate markup shock std and persistence
        m <= parameter(Symbol("σ_μ_$i"), 0.1314, (1e-8, 5.), (1e-8, 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       description = "σ_μ: standard deviation of mark up shock process",
                       tex_label = "\\sigma_{\\mu_$i}")
        m <= parameter(Symbol("ρ_μ_$i"), 0.8827, (1e-5, 0.999), (1e-5, 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       description = "ρ_μ: AR(1) coefficient of the mark up shock process",
                       tex_label = "\\rho_{\\mu_$i}")
    end

    if subspec_int ∈ [3, 4, 5]
        m <= parameter(:π_star, 0.5, fixed = true,
                   description = "Steady state rate of inflation")
    end

end

# Keshav's inflation shares
Kgam = DataFrame(CSV.File("/data/dsge_data_dir/proc/dsge/briefings/202412/Model_Data/gamma_vs_true_gamma.csv"))
Kgam_vec = vec(Kgam[!, :true_gamma])

m <= Setting(:Kgam, Kgam_vec)

end

end



function init_model_indices!(m::OnionModel)
    # For parsing model subspec to Int
    subspec_int = parse(Int, subspec(m)[3:end])

    n = get_setting(m, :n_sectors)


    exogenous_shocks            =
        [[Symbol("μ_$(i)_sh") for i in collect(keys(get_setting(m, :subgroup_names)))];
         [:μw_sh, :πstar_sh, :mp_sh, :b_sh, :a_sh]
         [:τ_sh, :μ_com_sh]]

    observables                 = keys(m.observable_mappings)

    pseudo_observables = keys(m.pseudo_observable_mappings)

    endogenous_states = [[Symbol("s_$(i)") for i in 1:n]; #(log deviation of) real sectoral prices
                         [Symbol("π_$i") for i in 1:n]; #sectoral inflation
                         [:r_t, :c_t, :πc_t, :πw_t, :w_t, :πKc_t] ; #interest rate, cons, CPI, wage Infl, wages
                         [:a_t, :b_t, :μw, :lτ, :τ, :πstar, :mp_t];
                         [:μ_com];
                         [Symbol("Eπ_$i") for i in 1:n];
                         [:Ec_t, :Eπc_t, :Eπw_t];
                         [Symbol("μ_$(i)") for i in collect(keys(get_setting(m, :subgroup_names)))]]

    endogenous_states_augmented = [:w_t1, :c_t1, :r_t1, :πc_t1] #, :e_meas_πc_t

    expected_shocks =[[Symbol("Eπ_$(i)_sh") for i in 1:n];
                      [:Ec_sh, :Eπc_sh, :Eπw_sh]]

    equilibrium_conditions = [[Symbol("eq_pc_$i") for i in 1:n];
                              [Symbol("eq_srec_$i") for i in 1:n];
                              [:eq_cpi, :eq_wpc, :eq_wrec, :eq_monpol, :eq_euler];
                              [:eq_a_t,:eq_b_t,:eq_μw, :eq_τ, :eq_lτdef, :eq_πstar, :eq_mp_t];
                              [:eq_Ect, :eq_Eπct, :eq_Eπwt, :eq_Kcpi];
                              [:eq_μ_com];
                              [Symbol("eq_Eπ_$i") for i in 1:n];
                              [Symbol("eq_μ_$(i)") for i in collect(keys(get_setting(m, :subgroup_names)))]]


    for (i,k) in enumerate(observables); m.observables[k] = i end
    for (i,k) in enumerate(pseudo_observables); m.pseudo_observables[k] = i end
    for (i,k) in enumerate(exogenous_shocks); m.exogenous_shocks[k] = i end
    for (i,k) in enumerate(expected_shocks); m.expected_shocks[k] = i end
    for (i,k) in enumerate(equilibrium_conditions); m.equilibrium_conditions[k] = i end
    for (i,k) in enumerate(endogenous_states); m.endogenous_states[k] = i end
    for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i + length(endogenous_states) end

    m <= Setting(:n_model_states, length(m.endogenous_states))
end

function steadystate!(m::OnionModel)
    return m
end


function shock_groupings(m::OnionModel)

    subspec_int = parse(Int, subspec(m)[3:end])

    #1) Create shock groups

    core_goods_mkp   = ShockGroup("mkp_core_goods", [:μ_core_goods_sh] , RGB(0.0, 0.6, 0.1))
    core_services_mkp   = ShockGroup("mkp_core_services", [:μ_core_services_sh], RGB(0.6,0.6,0.0))
    energy_mkp = ShockGroup("mkp_energy", [:μ_cpi_energy_sh], RGB(0.0, 0.6, 0.6))
    wage_pmu = ShockGroup("wage_mkp", [:μw_sh], RGB(0.5,0.0, 0.5))
    #tax = ShockGroup("tax", [:τ_sh], RGB(0.29, 0.0, 0.51))
    pis = ShockGroup("pi-LR", [:πstar_sh], RGB(1.0, 0.75, 0.793))
    pol = ShockGroup("pol", [:mp_sh], RGB(1.0,0.84,0.0))
    tfp = ShockGroup("tfp", [:a_sh], RGB(1.0,0.55,0.0))
    bet = ShockGroup("b", [:b_sh], RGB(0.3, 0.3, 1.0))


    #2) Return shock groups based on subspec

    if subspec_int ∈ [1, 2]
        return [core_goods_mkp, core_services_mkp, energy_mkp, tfp, wage_pmu, pol, bet]
    elseif subspec_int ∈ [3, 4, 5]
        return [core_goods_mkp, core_services_mkp, energy_mkp, tfp, pis, wage_pmu, pol, bet]
    else
        return [core_goods_mkp, core_services_mkp, energy_mkp,  wage_pmu, pol,  bet]
    end
end
