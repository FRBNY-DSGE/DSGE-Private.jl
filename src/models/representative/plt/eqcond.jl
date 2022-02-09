"""
```
eqcond(m::PLT)
```

Expresses the equilibrium conditions in canonical form using Γ0, Γ1, C, Ψ, and Π matrices.
Using the mappings of states/equations to integers defined in an_schorfheide.jl, coefficients are
specified in their proper positions.

### Outputs

* `Γ0` (`n_states` x `n_states`) holds coefficients of current time states.
* `Γ1` (`n_states` x `n_states`) holds coefficients of lagged states.
* `C`  (`n_states` x `1`) is a vector of constants
* `Ψ`  (`n_states` x `n_shocks_exogenous`) holds coefficients of iid shocks.
* `Π`  (`n_states` x `n_states_expectational`) holds coefficients of expectational states.
"""
function eqcond(m::PLT) # do not edit this function definition
    return eqcond(m, 1)
end

function eqcond(m::PLT, reg::Int) # do not edit these inputs
    endo = m.endogenous_states # for convenience, make references to these fields
    exo  = m.exogenous_shocks  # which hold indices, e.g. endo[:y_t] will return an Int
    ex   = m.expected_shocks   # for the index of the `y_t` variable.
    eq   = m.equilibrium_conditions # you can rename these abbreviations if you want

    Γ0 = zeros(n_states(m), n_states(m)) # initialize containers (do not edit)
    Γ1 = zeros(n_states(m), n_states(m))
    C  = zeros(n_states(m))
    Ψ  = zeros(n_states(m), n_shocks_exogenous(m))
    Π  = zeros(n_states(m), n_shocks_expectational(m))

    # Regime-switching parameters (you should probably not edit this call)
    for para in m.parameters      # if you're new to DSGE.jl, then revisit this block of code later.
        if !isempty(para.regimes) # if you don't need regime-switching, then you can comment out this block of code
            if (haskey(get_settings(m), :model2para_regime) ? haskey(get_setting(m, :model2para_regime), para.key) : false)
                toggle_regime!(para, reg, get_setting(m, :model2para_regime)[para.key])
            else
                toggle_regime!(para, reg)
            end
        end
    end

    # You should definitely make edits to the following code!

    ### ENDOGENOUS STATES ###

    ### 1. IS Curve

    Γ0[eq[:eq_is], endo[:x_t]]    = 1.
    Γ0[eq[:eq_is], endo[:Ex_t1]] = -1
    Γ0[eq[:eq_is], endo[:i_t]]   = 1 / m[:σ]
    Γ0[eq[:eq_is], endo[:Eπ_t1]] = - 1 / m[:σ]
    Γ0[eq[:eq_is], endo[:r_t]]  = - 1 / m[:σ]

    ### 2. NK Phillips Curve

    Γ0[eq[:eq_phillips], endo[:x_t]]   = -m[:κ]
    Γ0[eq[:eq_phillips], endo[:π_t]]   = 1
    Γ0[eq[:eq_phillips], endo[:Eπ_t1]] = -m[:β]
    Γ0[eq[:eq_phillips], endo[:u_t]]   = -1

    ### 3. Price level

    Γ1[eq[:eq_p], endo[:p_t]] = 1.0
    Γ0[eq[:eq_p], endo[:π_t]] = -1
    Γ0[eq[:eq_p], endo[:p_t]] = 1

    ### u
    Γ0[eq[:eq_u], endo[:u_t]] = 1
    Γ1[eq[:eq_u], endo[:u_t]] = m[:ρ_u]
    Ψ[eq[:eq_u], exo[:u_sh]]  = 1

    ### r
    Γ0[eq[:eq_r], endo[:r_t]] = 1.0
    Γ1[eq[:eq_r], endo[:r_t]] = m[:ρ_r]
    Ψ[eq[:eq_r], exo[:r_sh]]  = 1.0

    ### policy equation depends on subspec

    if subspec(m) in ["ss0"]
        # optimal policy
        ψ_i = 2.163#m[:κ] / (m[:β] * m[:σ])
        ψ_di = 1.0101#1 / m[:β]
        ψ_pi = 0.6408
        ψ_x  = 0.0813 #m[:λ_x] / (m[:λ_i] * m[:σ])

        Γ0[eq[:eq_pol], endo[:i_t]]  = 1
        Γ1[eq[:eq_pol], endo[:i_t]]  = ψ_i #(1 + ψ_i + ψ_di)
        Γ1[eq[:eq_pol], endo[:i_t1]] = - ψ_di
        Γ0[eq[:eq_pol], endo[:π_t]]  = - ψ_pi
        Γ0[eq[:eq_pol], endo[:x_t]]  = - ψ_x #/ 4
        Γ1[eq[:eq_pol], endo[:x_t]]  = - ψ_x #/ 4

        # Γ0[eq[:eq_pol], endo[:i_t]]  = 1
        # Γ1[eq[:eq_pol], endo[:i_t]]  = 2.163
        # Γ1[eq[:eq_pol], endo[:i_t1]] = -1.010
        # Γ0[eq[:eq_pol], endo[:π_t]]  = - 0.641
        # Γ0[eq[:eq_pol], endo[:x_t]]  = - 0.325
        # Γ1[eq[:eq_pol], endo[:x_t]]  = - 0.325
    elseif subspec(m) in ["ss1"]
        # optimal under commitment with lam_i = 0
        Γ0[eq[:eq_pol], endo[:π_t]] = 1
        Γ0[eq[:eq_pol], endo[:x_t]] = m[:λ_x] / m[:κ]
        Γ1[eq[:eq_pol], endo[:x_t]] = m[:λ_x] / m[:κ]
    elseif subspec(m) in ["ss2"]
        # optimal under discretion with lam_i = 0
        Γ0[eq[:eq_pol], endo[:π_t]] = 1
        Γ0[eq[:eq_pol], endo[:x_t]] = m[:λ_x] / m[:κ]
    elseif subspec(m) in ["ss3"]
        # Taylor rule ρ_r = 0 (coeff come from table 2 of paper

        Γ0[eq[:eq_pol], endo[:i_t]] = 1
        Γ0[eq[:eq_pol], endo[:π_t]] = -m[:ψ_piT]
        Γ0[eq[:eq_pol], endo[:x_t]] = -m[:ψ_xT]
    elseif subspec(m) in ["ss4"]
        # Taylor rule ρ_u = 0
        ψ_piT = 0.888
        ψ_xT  = 0.694
        Γ0[eq[:eq_pol], endo[:i_t]] = 1
        Γ0[eq[:eq_pol], endo[:π_t]] = -ψ_piT
        Γ0[eq[:eq_pol], endo[:x_t]] = -ψ_xT
    elseif subspec(m) in ["ss5"]
        # wick rule
        Γ0[eq[:eq_pol], endo[:i_t]] = 1
        Γ0[eq[:eq_pol], endo[:p_t]] = -m[:ψ_pW]
        Γ0[eq[:eq_pol], endo[:x_t]] = -m[:ψ_xW]
    elseif subspec(m) in ["ss6"]
        ψ_pW = 1.997
        ψ_xW = 0.182
        Γ0[eq[:eq_pol], endo[:i_t]] = 1
        Γ0[eq[:eq_pol], endo[:p_t]] = -ψ_pW
        Γ0[eq[:eq_pol], endo[:x_t]] = -ψ_xW
    end
    ### lagged i_t
    Γ0[eq[:eq_it1], endo[:i_t1]] = 1
    Γ1[eq[:eq_it1], endo[:i_t]]  = 1

    ### expected x_t
    Γ0[eq[:eq_Ex_t1], endo[:x_t]]   = 1
    Γ1[eq[:eq_Ex_t1], endo[:Ex_t1]] = 1
    Π[eq[:eq_Ex_t1], ex[:Ex_sh]] = 1
    ### expected pi_t
    Γ0[eq[:eq_Eπ_t1], endo[:π_t]]   = 1
    Γ1[eq[:eq_Eπ_t1], endo[:Eπ_t1]] = 1
    Π[eq[:eq_Eπ_t1], ex[:Eπ_sh]] = 1


    # Ensure parameter regimes are in 1 at the end (you should probably not edit)
    for para in m.parameters      # if you're new to DSGE.jl, then revisit this block of code later.
        if !isempty(para.regimes) # if you don't need regime-switching, then you can comment out this block of code
            toggle_regime!(para, 1)
        end
    end

    return Γ0, Γ1, C, Ψ, Π # do not edit this return
end
