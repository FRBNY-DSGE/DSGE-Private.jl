function init_subspec!(m::OnionModel)
    if subspec(m) == "ss1"
        return ss1!(m)
    else
        error("This subspec has not been defined.")
    end
end

function ss1!(m::OnionModel)
    return m
end
