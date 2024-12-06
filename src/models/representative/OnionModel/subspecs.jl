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
    elseif subspec(m) == "ss11"
        return ss11!(m)
    elseif subspec(m) == "ss12"
        return ss12!(m)
    elseif subspec(m) == "ss13"
        return ss13!(m)
    elseif subspec(m) == "ss14"
        return ss14!(m)
    elseif subspec(m) == "ss15"
        return ss15!(m)
    elseif subspec(m) == "ss16"
        return ss16!(m)
    else
        error("This subspec has not been defined.")
    end
end

function ss0!(m::OnionModel)
    #This is our base case, as laid out in onionmodel.jl.
    #=
    Observables:
    1. Nominal FFR
    2. Aggregate CPI Inflation
    3. Real Wage Growth
    4. Consumption Growth
    With 4 associated shocks:
    1. Monetary Policy (:mp_sh)
    2. TFP shock (a_sh)
    3. Wage Markup (μw_sh)
    4. Discount Rate (b_sh)
    =#
    return m
end

function ss1!(m::OnionModel)
    #We include one sector's iid markup shocks wihtout observing that sector
    #final_shock_ind = length(m.exogenous_shocks)
    #m.exogenous_shocks[final_shock_ind + 1] = Symbol("μ_iid_1_sh")
    return m
end

function ss2!(m::OnionModel)

    return m
end

function ss3!(m::OnionModel)
    return m
end

function ss4!(m::OnionModel)

    #final_shock_ind = length(m.exogenous_shocks)
    #for i in 1:get_setting(m, :n_sectors)
    #    m.exogenous_shocks[final_shock_ind + i] = Symbol("μ_iid_$(i)_sh")
    #end


    return m
end

function ss5!(m::OnionModel)
    #same as ss4, but we make the markup trend set to the iid
    m[:ρ_μ_trend].value = zeros(get_setting(m, :n_sectors))
    m[:σ_μ_trend].value = m[:σ_μ_iid].value

    return m
end

function ss6!(m::OnionModel)
    return m
end

function ss7!(m::OnionModel)
    return m
end

function ss8!(m::OnionModel)
    return m
end

function ss9!(m::OnionModel)
    return m
end

function ss10!(m::OnionModel)
    return m
end

function ss11!(m::OnionModel)
    return m
end


#This is where we start cooking. BMP + PG

function ss12!(m::OnionModel)

    return m
end

function ss13!(m::OnionModel)
    m[:ρ_μ_trend].value = ones(get_setting(m, :n_sectors)) * m[:ρ_trend].value
    return m
end

function ss14!(m::OnionModel)
    return m
end

function ss15!(m::OnionModel)
    return m
end

function ss16!(m::OnionModel)
    return m
end
