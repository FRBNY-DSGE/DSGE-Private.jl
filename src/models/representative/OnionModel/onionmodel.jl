mutable struct OnionModel{T} <: AbstractRepModel{T}
    parameters::Vector{Union{AbstractParameter{T}, AbstractVectorParameter{Vector,T}}}
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
        Vector{Union{AbstractParameter{Float64}, VectorParameter{Vector,Float64,Transform}}}(),
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
        OrderedDict{Symbol,PseudoObservable}())

    for setting in custom_settings
        m <= setting
    end

    init_settings!(m)
    init_observable_mappings!(m)
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
    end

    # Relevant for Model Building (sizes of matrices and such)
    m <= Setting(:n_back_states, get_setting(m, :n_sectors)+3)
    m <= Setting(:n_jump_states, get_setting(m, :n_sectors)+3)
    m <= Setting(:n_exo_states, 1)
    #m <= Setting(:n_model_states, get_setting(m, :n_back_states) + get_setting(m, :n_jump_states))


    # Relevant for filepaths to read data and save figures
    m <= Setting(:param_data_root, "/data/dsge_data_dir/proc/dsge/briefings/202412/Model_Data/")
    m <= Setting(:dataroot, "/data/dsge_data_dir/proc/dsge/briefings/202412/Model_Data/input_data/")
    m <= Setting(:saveroot, "/data/dsge_data_dir/proc/dsge/briefings/202412/output_data/")


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
    m <= Setting(:sector_names, names(CSV.read(sectoral_inflation_path, DataFrame))[3:end])


    # Relevant for other things such as IRFs, smoothing, and forecasting
    m <= Setting(:n_mon_anticipated_shocks, 0)
    m <= Setting(:forecast_smoother, :durbin_koopman)

    m <= Setting(:forecast_uncertainty_override, Nullable{Bool}())
    #m <= Setting(:shockdec_startdate, Nullable{Bool}())
    #m <= Setting(:shockdec_enddate, Nullable{Bool}())
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


        m <= parameter(:labshare, vec(fl["labshare"]))
        m <= parameter(:taxshare, vec(fl["taxshare"]))
        m <= parameter(:tointshare, vec(fl["tointshare"]))
        m <= parameter(:int_totout, vec(fl["ind_totout"]))
        m <= parameter(:invkap, vec(fl["invkap"]))
        m <= parameter(:food, vec(fl["food"]))
        m <= parameter(:core, vec(fl["core"]))
        m <= parameter(:energy, vec(fl["energy"]))
        m <= parameter(:gam, vec(fl["gam"])) #Expenditure shares
        m <= paramater(:gam_core, vec(fl["gam_core"]))
        m <= parameter(:gam_food, vec(fl["gam_food"]))
        m <= parameter(:gam_energy, vec(fl["gam_energy"]))
        m <= parameter(:gam_services, vec(fl["gam_services"]))
        m <= parameter(:gam_coreservices, vec(fl["gam_coreservices"]))
        m <= parameter(:gam_goods, vec(fl["gam_goods"]))
        m <= parameter(:gam_coregoods, vec(fl["gam_coregoods"]))
        m <= parameter(:pc_px, vec(fl["pc_px"]))
        m <= parameter(:to_mx, vec(fl["to_mx"]))
        m <= parameter(:ei, vec(fl["ei"]))
        m <= parameter(:ς_tilde, vec(fl["varsig_tilde"]))


    elseif subspec_int == 1

        paras = InOutData()



        m <= parameter(:bet, paras["β"], fixed = true,
                       description="β: temporal discount",
                       tex_label="\\beta")

        m <= parameter(:ρ_τ, 1.22, fixed = true, #paras["ρ_τ"]
                       description="ρ_τ: ",
                       tex_label="\\rho_\\tau")

        m <= parameter(:ρ_τ2, -0.2475, fixed = true,
                       description="ρ_τ2: ",
                       tex_label="\\rho_\\tau2")

m <= parameter(:ρ_i, 0.85^(1/3), fixed = true,
               description="ρ_i: policy inertia",
               tex_label="\\rho_i")

m <= parameter(:η, paras["η"], fixed = true,
               description="η: ",
               tex_label="\\eta")

m <= parameter(:ζ, paras["ζ"], fixed = true,
               description="ζ:",
               tex_label="\\zeta")

m <= parameter(:ν, paras["ν"], fixed = true,
               description="ν: ",
               tex_label="\\nu")

m <= parameter(:ξ, paras["ξ"], fixed = true,
               description="ξ: ",
               tex_label="\\xi")

m <= parameter(:mp_cpi_infl, 1.01, fixed = true, #paras["mp_cpi_infl"] NEEDS TO BE 1.01 to AVOID EIGENVALUE ISSUE
               description="weight on cpi inflation in mp rule",
               tex_label="mp_cpi_infl")

m <= parameter(:mp_cons, 0., fixed = true, #paras["mp_cons"]
               description="weight on consumption in mp rule",
               tex_label="mp_cons")

m <= parameter(:mp_cstar, 0., fixed = true, #paras["mp_cstar"]
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

    m <= parameter(:coal, paras["coal"], fixed = true,
                   description = "Index of coal",
                   tex_label="coal")


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

m <= parameter(:labshare, vec(paras["labshare"]))
m <= parameter(:taxshare, vec(paras["taxshare"]))
m <= parameter(:totintshare, vec(paras["totintshare"]))
m <= parameter(:int_totout, vec(paras["ind_totout"]))
m <= parameter(:invkap, vec(paras["invkap"]))
m <= parameter(:food, float(vec(paras["food"])))
m <= parameter(:core, float(vec(paras["core"])))
m <= parameter(:energy, float(vec(paras["energy"])))
m <= parameter(:gam, vec(paras["gam"]))
m <= parameter(:gam_core, vec(paras["gam_core"]))
m <= parameter(:gam_food, vec(paras["gam_food"]))
m <= parameter(:gam_energy, vec(paras["gam_energy"]))
m <= parameter(:pc_px, vec(paras["pc_px"]))
m <= parameter(:to_mx, vec(paras["to_mx"]))
m <= parameter(:ei, vec(paras["ei"]))
m <= parameter(:ς_tilde, vec(paras["varsig_tilde"]))
m <= parameter(:ρ_μ_trend, vec(0.999 * ones(get_setting(m, :n_sectors))))



## Adding model parameters for standard deviation of shocks
m <= parameter(:σ_c, 0.8719, fixed = true,
               description = "σ_c: Coefficient of relative risk aversion")
m <= parameter(:σ_b_t, 0.0292,fixed = true,
               description = "σ_b: Standard deviation of the discount rate process")
m <= parameter(:ρ_b_t, 0.941, fixed = true,
               description = "ρ_b: AR(1) coefficient of the discount rate process")
m <= parameter(:σ_μ, 0.1314, fixed = true,
               description = "σ_μ: standard deviation of mark up shock process")
m <= parameter(:ρ_μ, 0.8827, fixed = true,
               description = "ρ_μ: AR(1) coefficient of the mark up shock process")
m <= parameter(:σ_μw, 0.1314, fixed = true,
               description = "σ_wμ: standard deviation of wage mark up shock process")
m <= parameter(:ρ_μw, 0.3884, fixed = true,
               description = "ρ_μw: AR(1) coefficient in the wage mark up shock process")
m <= parameter(:σ_πstar, 0.0269, fixed = true,
               description = "σ_πstar: standard deviation of the process describing the time varying inflation target")
m <= parameter(:ρ_πstar, 0.99, fixed = true,
               description = "ρ_πstar: AR(1) coefficient of process describing the time varying inflation target")
m <= parameter(:σ_a_t, 0.6742, fixed = true, #Taken from std dev of stationary comp of prod
               description = "σ_a_t: standard deviation of the process describing productivity")
m <= parameter(:ρ_a_t, 0.6742, fixed = true,#Taken from std dev of stationary comp of prod
               description = "ρ_a_t: AR(1) coefficient of the process describing productivity")

m <= parameter(:h, 0.5347, fixed = true,
               description = "h: consumption habit persistence")
m <= parameter(:γ, 0.0, fixed=true, #0.3673, fixed = true,#Growth rate of economy
               description = "γ: Log of the steady-state growth rate of technology")

m <= parameter(:mp_habit, 0.0, fixed = true,
               description = ":mp_habit: weight of MP rule on habit formation")
m <= parameter(:σ_r_m, 0.2380, fixed = true,
               description = "Standard deviation of process describing iid monetary policy shock")


end

end



function init_model_indices!(m::OnionModel)
    # For parsing model subspec to Int
    subspec_int = parse(Int, subspec(m)[3:end])

    n = get_setting(m, :n_sectors)


    exogenous_shocks            = [[Symbol("μ_trend_$(i)_sh") for i in 1:n];
                                   [Symbol("μ_iid_$(i)_sh") for i in 1:n];
                                   [:μw_sh, :πstar_sh, :mp_sh, :b_sh, :a_sh]
                                   [:τ_sh]]

    observables                 = keys(m.observable_mappings)

    endogenous_states = [[Symbol("s_$(i)") for i in 1:n]; #(log deviation of) real sectoral prices
                         [Symbol("π_$i") for i in 1:n]; #sectoral inflation
                         [:r_t, :c_t, :πc_t, :πw_t, :w_t] ; #interest rate, cons, CPI, wage Infl, wages
                         [:a_t, :b_t, :μw, :lτ, :τ, :πstar];
                         [Symbol("Eπ_$i") for i in 1:n];
                         [:Ec_t, :Eπc_t, :Eπw_t];
                         [Symbol("mkup_trend_$(i)") for i in 1:n]]

    endogenous_states_augmented = [:w_t1, :c_t1, :r_t1, :πc_t1]

    expected_shocks =[[Symbol("Eπ_$(i)_sh") for i in 1:n];
                      [:Ec_sh, :Eπc_sh, :Eπw_sh]]

    equilibrium_conditions = [[Symbol("eq_pc_$i") for i in 1:n];
                              [Symbol("eq_srec_$i") for i in 1:n];
                              [:eq_cpi, :eq_wpc, :eq_wrec, :eq_monpol, :eq_euler];
                              [:eq_a_t,:eq_b_t,:eq_μw, :eq_τ, :eq_lτdef, :eq_πstar];
                              [:eq_Ect, :eq_Eπct, :eq_Eπwt];
                              [Symbol("eq_Eπ_$i") for i in 1:n];
                              [Symbol("eq_mkup_trend_$(i)") for i in 1:n]]


    for (i,k) in enumerate(observables); m.observables[k] = i end
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
    #Ignore subspecs for now:

    pmu_trend = ShockGroup("mkp_trend", [Symbol("μ_trend_$(i)_sh") for i in 1:get_setting(m, :n_sectors)], RGB(0.0, 0.8, 0.0))
    pmu_iid = ShockGroup("mkp_iid", [Symbol("μ_iid_$(i)_sh") for i in 1:get_setting(m, :n_sectors)], RGB(0.5, 0.5, 0.0))
    wage_pmu = ShockGroup("wage_mkp", [:μw_sh], RGB(0.5,0.0, 0.5))
    tax = ShockGroup("tax", [:τ_sh], RGB(0.29, 0.0, 0.51))
    pis = ShockGroup("pi-LR", [:πstar_sh], RGB(1.0, 0.75, 0.793))
    pol = ShockGroup("pol", [:mp_sh], RGB(1.0,0.84,0.0))
    tfp = ShockGroup("tfp", [:a_sh], RGB(1.0,0.55,0.0))
    bet = ShockGroup("b", [:b_sh], RGB(0.3, 0.3, 1.0))

    #[:μw_sh, :πstar_sh, :mp_sh, :b_sh, :a_sh]
    return [pmu_trend, pmu_iid,wage_pmu, tax, pis, pol, tfp, bet]
end
