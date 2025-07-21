#=
Role of subspecs within this function is to set parameter values and designate any necessary settings.

=#
function init_subspec!(m::OnionModel)
    if subspec(m) == "ss0"
        return ss0!(m)
    elseif subspec(m) == "ss1"
        return ss1!(m)
    elseif subspec(m) == "ss2"
        return ss2!(m)
    elseif subspec(m) == "ss3"
        return ss3!(m)
    elseif subspec(m) == "ss4"
        return ss4!(m)
    elseif subspec(m) == "ss5"
        return ss5!(m)
    elseif subspec(m) == "ss6"
        return ss6!(m)
    elseif subspec(m) == "ss7"
        return ss7!(m)
    else
        error("This subspec has not been defined.")
    end
end

# This is the version of the model with all ~400 sectors
function ss0!(m::OnionModel)
    return m
end

# This is the briefing model. The analog for this subspec is marco_test_num = 68.
function ss1!(m::OnionModel)
    return m
end

# This is the model where we estimate the std of differentiated inflation shocks.
function ss2!(m::OnionModel)

    #1) Shut down inflation targeting process
    m[:σ_πstar].value = 0
    m[:ρ_πstar].value = 0

    #2) Specify estimated parameters and fix all others: saves having to add this to forecast and
    # estimation spec files
    estimated_params = [:σ_μ_core_goods, :σ_μ_core_services, :σ_μ_cpi_energy, :σ_μw]

    for param in m.parameters
        if param.key ∈ estimated_params
            m[param.key].fixed = false
        else
            m[param.key].fixed = true
        end
    end

    #3) Order of parameters in cloud is same as order of parameters in m.parameters. Since parameter
    # ordering has changed between model used to estimate and this model, we reorder params accordingly (temp)

    cloud_dict = Dict(
        :bet               => 1,  :ρ_τ               => 2,
        :ρ_τ2              => 3,  :ρ_i               => 4,
        :η                 => 5,
        :ζ                 => 6,  :ν                 => 7,
        :ξ                 => 8,  :mp_cpi_infl       => 9,
        :mp_cons           => 10, :mp_cstar          => 11,
        :invkapw           => 12, :oil               => 13,
        :gas               => 14, :σ_c               => 15,
        :σ_b_t             => 16, :ρ_b_t             => 17,
        :σ_μ_core_goods    => 18, :ρ_μ_core_goods    => 19,
        :σ_μ_core_services => 20, :ρ_μ_core_services => 21,
        :σ_μ_cpi_energy    => 22, :ρ_μ_cpi_energy    => 23,
        :σ_μw              => 24, :ρ_μw              => 25,
        :σ_πstar           => 26, :ρ_πstar           => 27,
        :σ_a_t             => 28, :ρ_a_t             => 29,
        :h                 => 30, :γ                 => 31,
        :mp_habit          => 32, :σ_r_m             => 33)

    m <= Setting(:model2cloud_dict, cloud_dict)

    return m
end

# This is the model where we estimate the shock process. Analog is marco_test_num = 100.
function ss3!(m::OnionModel)

    #=
    1) We fix all parameters other than
       a) σ and ρ for core goods, core services, and cpi energy
       b) σ and ρ for a, b, and μw
       c) σ_c and σ_πstar
    =#
    fixed_params = [:h, :mp_cpi_infl, :mp_cons, :mp_cstar, :mp_habit, :γ]

    for param in m.parameters
        if param.key ∈ fixed_params
            m[param.key].fixed = true
        end
    end

    #2) Order of parameters in this model are different than the model used to run estimation.
    # We adust index of modal draws to match the position of parameters in this model.
    cloud_dict = Dict(
        :bet               => 1,  :ρ_τ               => 2,
        :ρ_τ2              => 3,  :ρ_i               => 4,
        :η                 => 5,
        :ζ                 => 6,  :ν                 => 7,
        :ξ                 => 8,  :mp_cpi_infl       => 9,
        :mp_cons           => 10, :mp_cstar          => 11,
        :invkapw           => 12, :oil               => 13,
        :gas               => 14, :σ_c               => 15,
        :σ_b_t             => 16, :ρ_b_t             => 17,
        :σ_μ_core_goods    => 18, :ρ_μ_core_goods    => 19,
        :σ_μ_core_services => 20, :ρ_μ_core_services => 21,
        :σ_μ_cpi_energy    => 22, :ρ_μ_cpi_energy    => 23,
        :σ_μw              => 24, :ρ_μw              => 25,
        :σ_πstar           => 26, :ρ_πstar           => 27,
        :σ_a_t             => 28, :ρ_a_t             => 29,
        :h                 => 30, :γ                 => 31,
        :mp_habit          => 32, :σ_r_m             => 33,
        :π_star            => 34)

     m <= Setting(:model2cloud_dict, cloud_dict)

    return m
end

function ss4!(m::OnionModel)

    #=
    1) We fix all parameters other than
       a) σ and ρ for core goods, core services, and cpi energy
       b) σ and ρ for a, b, and μw
       c) σ_c
    and we keep σ_πstar fixed at = 0.01

    2) Estimated using cleaned spec ss4 so no need for model2cloud dictionary
    =#
    fixed_params = [:h, :mp_cpi_infl, :mp_cons, :mp_cstar, :σ_πstar, :mp_habit, :γ]

    m[:σ_πstar].value = 0.01

    for param in m.parameters
        if param.key ∈ fixed_params
            m[param.key].fixed = true
        end
    end

    return m
end


function ss5!(m::OnionModel)

    #=
    1) We estimate most parameters except
       a) Elasticity parameters (ν, ζ, η, ξ)
       b) σ_πstar fixed to 0.01
       c) Indexes for oil and gas (should be settings in retrospect)
       d) Primitives like π_star (already detrend lr inflation), invkapw
       e) mp_cpi_infl fixed to 1.01 for eigenvalue issue? (estimate for now)
       f) taus (we do not draw any tau shocks as their std set to 0)

    2) Estimated using cleaned spec ss4 so no need for model2cloud dictionary
    =#
    fixed_params = [:ν, :ζ, :η, :ξ,
                    :σ_πstar,
                    :oil, :gas,
                    :invkapw, :π_star,
                    :ρ_τ, :ρ_τ2]

    m[:σ_πstar].value = 0.01

    for param in m.parameters
        if param.key ∈ fixed_params
            m[param.key].fixed = true
        else
            m[param.key].fixed = false
        end
    end

    return m
end

function ss6!(m::OnionModel)

    #=
    1) We estimate most parameters except
       a) Elasticity parameters (ν, ζ, η, ξ)
       b) σ_πstar fixed to 0.01
       c) Indexes for oil and gas (should be settings in retrospect)
       d) Primitives like π_star (already detrend lr inflation), invkapw
       e) mp_cpi_infl fixed to 1.01 for eigenvalue issue? (estimate for now)
       f) taus (we do not draw any tau shocks as their std set to 0)

    2) Estimated using cleaned spec ss4 so no need for model2cloud dictionary
    =#
    fixed_params = [:ν, :ζ, :η, :ξ,
                    :σ_πstar,
                    :oil, :gas,
                    :invkapw, :π_star,
                    :ρ_τ, :ρ_τ2]

    m[:σ_πstar].value = 0.01

    for param in m.parameters
        if param.key ∈ fixed_params
            m[param.key].fixed = true
        else
            m[param.key].fixed = false
        end
    end

    return m
end

function ss7!(m::OnionModel)

    #=
    In this subspec, we
    (1) Observables: Remove tfp growth observable and consumption growth observables.
        We add cpi inflation obs and cpi_inflation shock
    (2) Shocks: Add a common mark-up shock process while keeping categorical shocks.
    (3) Add mark-up shock for food w/o specifying observable.
    =#

    # (1.A) Set std and persistance to 0 to get rid of tfp shock process
    m[:σ_a_t].value = 0.
    m[:ρ_a_t].value = 0.

    m[:σ_a_t].fixed = true
    m[:ρ_a_t].fixed = true


    # (1.B) Set std and persistance to 0 to get rid of discount factor shock process
    m[:σ_b_t].value = 0.
    m[:ρ_b_t].value = 0.

    m[:σ_b_t].fixed = true
    m[:ρ_b_t].fixed = true

    # (2) Add common mark-up

    # (1e-8, 5.) (1e-8, 5.)
    m <= parameter(:σ_μ_com, 0.1314, (0., 5.), (0., 5.), ModelConstructors.Exponential(), RootInverseGamma(2, 0.10), fixed=false,
                       description = "σ_μ_com: standard deviation of common mark up shock process",
                       tex_label = "\\sigma_{\\mu_com}")
    m <= parameter(:ρ_μ_com, 0.8827, (0., 0.999), (0., 0.999), ModelConstructors.SquareRoot(), BetaAlt(0.5, 0.2), fixed=false,
                       description = "ρ_μ: AR(1) coefficient of the mark up shock process",
                       tex_label = "\\rho_{\\mu_com}")

    # Set this to zero (in model but not estimated?)
    m[:σ_μ_com].value = 0.
    m[:ρ_μ_com].value = 0.

    # (3.A) Add core_foods to subgroup names
    m <= Setting(:subgroup_names,
                 OrderedDict{String, Symbol}("core_goods" => :CUSR0000SACL1E,
                                             "core_services" => :CUSR0000SASLE,
                                             "cpi_energy" => :CPIENGSL,
                                             "core_foods" => :NOTHING)) # Is it in this dataset?
    ### Estimation changes ###

    return m
end

function ss8!(m::OnionModel)
    #=
    In this subspec, we
    (1) Observables: Remove just tfp growth
    (2) Shocks: Add a common shock process while keeping categorical shocks.
    (3) Add mark-up shock for food w/o specifying observable
    =#

    return m
end
