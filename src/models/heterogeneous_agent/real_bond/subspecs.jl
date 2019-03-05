"""
`init_subspec!(m::RealBond)`

Initializes a model subspecification by overwriting parameters from
the original model object with new parameter objects. This function is
called from within the model constructor.
"""
function init_subspec!(m::RealBond)
    if subspec(m) == "ss0"
        return
    elseif subspec(m) == "ss1"
        return ss1!(m)
    else
        error("This subspec should be a 0")
    end
end

"""
```
ss1!(m::RealBond)
```

Initializes subspec 1 of `RealBond`. In this subspecification,
Every the monetary shock process is shut down with σ_mon = 0., and
ρ_z is fixed so the only parameter that is free is σ_z.
"""
function ss1!(m::RealBond)
    m <= parameter(:ρ_z, 0.0, fixed = true,
                   description="ρ_z: AR(1) coefficient in the technology process.",
                   tex_label="\\rho_z")
    m <= parameter(:σ_mon, 0.0, fixed = true,
                   description="σ_mon: The standard deviation of the monetary policy shock.",
                   tex_label="\\sigma_{mon}")
end
