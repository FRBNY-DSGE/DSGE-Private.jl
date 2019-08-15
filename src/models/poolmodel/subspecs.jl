"""
`init_subspec!(m::PoolModel)`

Initializes a model subspecification by overwriting parameters from
the original model object with new parameter objects. This function is
called from within the model constructor.
"""
function init_subspec!(m::PoolModel)
    if subspec(m) == "ss0"
        return
    elseif subspec(m) == "ss1"
        return ss1!(m)
    elseif subspec(m) == "ss2"
        return ss2!(m)
    else
        error("This subspec is not defined")
    end
end

"""
```
ss1!(m::PoolModel)
```

Initialize subspec 1 of `PoolModel` which uses the original Matlab predictive densities.
"""
function ss1!(m::PoolModel)
end

"""
```
ss2!(m::PoolModel)
```

Initialize subspec 2 of `PoolModel` which uses the corrected `Model805` predictive density
and the original `Model904` predictive density from Matlab.
"""
function ss2!(m::PoolModel)
end
