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
    elseif subspec(m) == "ss8"
        return ss8!(m)
    elseif subspec(m) == "ss9"
        return ss9!(m)
    elseif subspec(m) == "ss10"
        return ss10!(m)
    elseif subspec(m) == "ss12"
        return ss12!(m)
    elseif subspec(m) == "ss13"
        return ss13!(m)
    elseif subspec(m) == "ss20"
        return ss20!(m)
    elseif subspec(m) == "ss21"
        return ss21!(m)
    elseif subspec(m) == "ss113"
        return ss113!(m)
    elseif subspec(m) == "ss114"
        return ss114!(m)
    elseif subspec(m) == "ss115"
        return ss115!(m)
    elseif subspec(m) == "ss116"
        return ss116!(m)
    elseif subspec(m) == "ss117"
        return ss117!(m)
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
       b) σ_πstar fixed to 0.01; ρ_πstar fixed
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
                    :ρ_τ, :ρ_τ2, :ρ_πstar]

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
       g) Here, we test the CPI inflation observable

    2) Estimated using cleaned spec ss4 so no need for model2cloud dictionary
    =#
    fixed_params = [:ν, :ζ, :η, :ξ,
                    :σ_πstar,
                    :oil, :gas,
                    :invkapw, :π_star,
                    :ρ_τ, :ρ_τ2, :ρ_πstar]

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
    ss6!(m)
    # Estimate all variables except for fixed_params from ss6.
    # In ss7:
    # (1) We get rid of the TFP growth process and consumption growth process (including both observables)
    # (2) We add the CPI inflation observable (no need to add CPI inflation shock)
    # (3) We add food markup shock process (include an unobserved cpi_foods subgroup) and estimate rho/sigma
    # (4) We add an iid common markup shock and estimate sigma

    # Shut down tfp shock process
    m[:σ_a_t].value = 0.
    m[:ρ_a_t].value = 0.
    m[:σ_a_t].valuebounds = (0., 5.)
    m[:ρ_a_t].valuebounds = (0., 5.)

    # Shut down discout factor shock process
    m[:σ_b_t].value = 0.
    m[:ρ_b_t].value = 0.
    m[:σ_b_t].valuebounds = (0., 5.)
    m[:ρ_b_t].valuebounds = (0., 5.)

    ss6_to_7_fixed = [:σ_a_t, :ρ_a_t,
                      :σ_b_t, :ρ_b_t]

    for param in m.parameters
        if param.key ∈ ss6_to_7_fixed
            m[param.key].fixed = true
        end
    end

    return m
end

function ss8!(m::OnionModel)
    ss6!(m)

    # Estimate all variables except for fixed_params from ss6.
    # In ss7:
    # (1) We get rid of the TFP growth process but keep consumption growth process (and obs)
    # (2) We add the CPI inflation observable (no need to add CPI inflation shock)
    # (3) We add food markup shock process (include an unobserved cpi_foods subgroup) and estimate rho/sigma
    # (4) We add an iid common markup shock and estimate sigma

    # Shut down tfp shock process
    m[:σ_a_t].value = 0.
    m[:ρ_a_t].value = 0.
    m[:σ_a_t].valuebounds = (0., 5.)
    m[:ρ_a_t].valuebounds = (0., 5.)


    ss6_to_8_fixed = [:σ_a_t, :ρ_a_t]

    for param in m.parameters
        if param.key ∈ ss6_to_8_fixed
            m[param.key].fixed = true
        end
    end

    return m
end

function ss9!(m::OnionModel)
    ss7!(m)

    # In ss9:
    # (1) Using ss7, we set gamma to 0 so model is propertly demeaned

    m[:γ].value = 0
    m[:γ].fixed = true

    return m
end

function ss10!(m::OnionModel)
    ss8!(m)

    # In ss10:
    # (1) Using ss8, we set gamma to 0 so model is propertly demeaned

    m[:γ].value = 0
    m[:γ].fixed = true

    return m
end


function ss12!(m::OnionModel)
    # Subspec for DSGEVAR estimation of onion model where:
    ss5!(m)

    m[:γ].value = 0
    m[:γ].fixed = true

#=
    if haskey(get_settings(m), :fix_ρ_πstar) && get_setting(m, :fix_ρ_πstar)
        m[:ρ_πstar].fixed = true
        m[:ρ_πstar].value  = 0.99
    end
=#

end

function ss13!(m::OnionModel)
    # Subspec for DSGEVAR estimation of onion model where:
    ss12!(m)

    # Addition:
    # Add cpi_food as an observable
    # Add meas err process to CPI inflation

    # Fix rho_meas_cpi to 0
    m[:ρ_meas_cpi].fixed = true
    m[:ρ_meas_cpi].value = 0.0

    m[:σ_meas_cpi].fixed = false
end

function ss113!(m::OnionModel)
    # Subspec for DSGEVAR estimation of onion model where:
    ss13!(m)

    # Addition:
    # We keep the four main subgroups (goods, services, energy, food)
    # We have sector specific markup shock persistence and std (no longer subgroup specific)
end

function ss114!(m::OnionModel)
    # Subspec for DSGEVAR estimation of onion model where:
    ss113!(m)

    # Addition:
    # We keep the four main subgroups (goods, services, energy, food) but try decomposing energy
    # We have sector specific markup shock persistence and std (no longer subgroup specific)
    # In raw data, renamed cpiquarter_241023 to be cpiquarter_241122 to match data vints

    # Create setting for which subgroup to decompose into sectoral observables
    decomp_subgroup = ["cpi_energy"]
    m <= Setting(:decomp_subgroup, decomp_subgroup)

end

function ss115!(m::OnionModel)
    # Subspec for DSGEVAR estimation of onion model where:
    ss113!(m)

    # Addition:
    # We keep the four main subgroups (goods, services, energy, food) but try decomposing energy
    # We have sector specific markup shock persistence and std (no longer subgroup specific)
    # In raw data, renamed cpiquarter_241023 to be cpiquarter_241122 to match data vints

    # Create setting for which subgroup to decompose into sectoral observables
    decomp_subgroup = ["cpi_energy", "cpi_food"]
    m <= Setting(:decomp_subgroup, decomp_subgroup)

end

function ss116!(m::OnionModel)
    # Subspec for DSGEVAR estimation of onion model where:
    ss113!(m)

    # Addition:
    # We keep the four main subgroups (goods, services, energy, food) but try decomposing energy
    # We have sector specific markup shock persistence and std (no longer subgroup specific)
    # In raw data, renamed cpiquarter_241023 to be cpiquarter_241122 to match data vints

    # Create setting for which subgroup to decompose into sectoral observables
    decomp_subgroup = ["cpi_energy", "cpi_food", "core_goods"]
    m <= Setting(:decomp_subgroup, decomp_subgroup)

end

function ss117!(m::OnionModel)
    # Subspec for DSGEVAR estimation of onion model where:
    ss113!(m)

    # Addition:
    # We keep the four main subgroups (goods, services, energy, food) but try decomposing energy
    # We have sector specific markup shock persistence and std (no longer subgroup specific)
    # In raw data, renamed cpiquarter_241023 to be cpiquarter_241122 to match data vints

    # Create setting for which subgroup to decompose into sectoral observables
    decomp_subgroup = ["cpi_energy", "cpi_food", "core_goods", "core_services"]
    m <= Setting(:decomp_subgroup, decomp_subgroup)

end




#=
function ss20!(m::OnionModel)
    # Only observe cpi inflation and PCE expectations and no other variables
    ss5!(m)
    m[:γ].value = 0
    m[:γ].fixed = true

end

function ss21!(m::OnionModel)
    # Previously ss20

    # CPI inflation & expectations model (shutting down everything not in sectoral pc)
    ss9!(m)

    # Shut down discount factor shock process
    m[:σ_b_t].value = 0.
    m[:ρ_b_t].value = 0.

    m[:σ_b_t].valuebounds = (0., 0.)
    m[:ρ_b_t].valuebounds = (0., 0.)

    m[:σ_b_t].fixed = true
    m[:ρ_b_t].fixed = true

    # Shut down wage growth process
    m[:σ_μw] = 0.
    m[:ρ_μw] = 0.

    m[:σ_μw].valuebounds = (0., 0.)
    m[:ρ_μw].valuebounds = (0., 0.)

    m[:σ_μw].fixed = true
    m[:ρ_μw].fixed = true

    # Remove tfp process
    m[:σ_a_t].value = 0.
    m[:ρ_a_t].value = 0.

    m[:σ_a_t].valuebounds = (0., 0.)
    m[:ρ_a_t].valuebounds = (0., 0.)

    m[:σ_a_t].fixed = true
    m[:ρ_a_t].fixed = true

    # Remove mp_shock
    m[:σ_r_m].value = 0.
    m[:σ_r_m].valuebounds = (0., 0.)
    m[:σ_r_m].fixed = true

    #rho pi_star (fix!)
    m[:ρ_πstar].fixed = true
end
=#
