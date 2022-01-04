"""
`init_subspec!(m::AnSchorfheide)`

Initializes a model subspecification by overwriting parameters from
the original model object with new parameter objects. This function is
called from within the model constructor.
In this case, subspecs used in eqcond to define policy equation.
See notes on plt for more info
"""
function init_subspec!(m::PLT) # unless you want to add subspecifications, do not edit this function definition.
    if subspec(m) in ["ss0", "ss1", "ss2", "ss3", "ss4", "ss5", "ss6"]
        return
    else
        error("This subspec should be a 0")
    end
end
