"""
`init_subspec!(m::SmetsWouters)`

Initializes a model subspecification by overwriting parameters from
the original model object with new parameter objects. This function is
called from within the model constructor.
"""
function init_subspec!(m::DSSW)
    if subspec(m) == "ss0"
        return m
    elseif subpect(m) == "ss1"
        return ss1!(m)
    end
end

function ss1!(m::DSSW)
    #=
    This model object corresponds to the code run in Michael Pham's
    coint_dsgevar/euro/michael_get_matrices.m file to obtain matlab matrices
    and test vecm functions (vecm_likelihood, vecm_population_moments, vecm_irfs)
    in Julia. This version of the model isn't correct (it is neither m106 nor m101),
    but was used in a validation exercise.
    =#

end
