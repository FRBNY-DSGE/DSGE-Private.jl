"""
`init_subspec!(m::SmetsWouters)`

Initializes a model subspecification by overwriting parameters from
the original model object with new parameter objects. This function is
called from within the model constructor.
"""
function init_subspec!(m::DSSW)
    if subspec(m) == "ss0"
        return m
    elseif subspec(m) == "ss1"
        return ss1!(m)
    end
end

function ss1!(m::DSSW)
    # This model corresponds to the m106 model in the DSSW

    # Calibrated parameter values
    m[:alp].value = 0.33
    m[:iota_p].value = 0.5
    m[:iota_w].value = 0.5
    m[:bet].value = 2.0
    m[:pistar].value = 3.0
    m[:gam].value = 2.0
    m[:wadj].value = 0.0
    m[:laf].value = 0.15
    m[:gstar].value = 0.15
    m[:Ladj].value = 252.

    # Valuebounds
    m[:ups].valuebounds = (0., 10.)
    m[:bigphi].valuebounds = (0., 5.)

    # Transformation
    m[:wadj].transform_parameterization = (0., 0.)

    # Scaling
    m[:ups].scaling = x -> exp(x/400)
    m[:bet].scaling = x -> exp(-x/400)
    m[:pistar].scaling = x -> exp(x/400)
    m[:gam].scaling = x -> x/400
    m[:gstar].scaling = x -> 1 + x

    # Fixing parameters (from matlab: para_mask) - Lazy, I'm sorry
    fixed_list = [0;0;0;1;1;1;0;0;0;0;1;0;0;1;0;0;0;0;0;0;1;1;0;0;0;0;0;1;0;0;0;0;0;0;1;0;0;0;0;0]
    for (i, f_val) in enumerate(fixed_list)
        f_val == 0 ? m.parameters[i].fixed = false : m.parameters[i].fixed = true
    end

end


#function ss1!(m::DSSW)
    #=
    This model object corresponds to the code run in Michael Pham's
    coint_dsgevar/euro/michael_get_matrices.m file to obtain matlab matrices
    and test vecm functions (vecm_likelihood, vecm_population_moments, vecm_irfs)
    in Julia. This version of the model isn't correct (it is neither m106 nor m101),
    but was used in a validation exercise.
    =#

#end
