#=
Role of subspecs within this function is to set parameter values and designate any necessary settings.

=#
function init_subspec!(m::SectoralOnionModel)
    if subspec(m) == "ss0"
        return ss0!(m)
    elseif subspec(m) == "ss1"
        return ss1!(m)
    else
        error("This subspec has not been defined.")
    end
end

# This is the version of the model with all ~400 sectors
function ss0!(m::SectoralOnionModel)
    return m
end

# This is the briefing model
function ss1!(m::SectoralOnionModel)
    return m
end
