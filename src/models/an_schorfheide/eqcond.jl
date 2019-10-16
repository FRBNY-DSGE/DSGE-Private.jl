"""
```
eqcond(m::Schorf)
```

Expresses the equilibrium conditions in canonical form using Γ0, Γ1, C, Ψ, and Π matrices.
Using the mappings of states/equations to integers defined in schorf.jl, coefficients are
specified in their proper positions.

### Outputs

* `Γ0` (`n_states` x `n_states`) holds coefficients of current time states.
* `Γ1` (`n_states` x `n_states`) holds coefficients of lagged states.
* `C`  (`n_states` x `1`) is a vector of constants
* `Ψ`  (`n_states` x `n_shocks_exogenous`) holds coefficients of iid shocks.
* `Π`  (`n_states` x `n_states_expectational`) holds coefficients of expectational states.
"""
function eqcond(m::AnSchorfheide; method::Symbol = :gensys,
                standard_normal_exog_shocks::Bool = false)

    if method == :gensys
        endo = m.endogenous_states
        exo  = m.exogenous_shocks
        ex   = m.expected_shocks
        eq   = m.equilibrium_conditions

        Γ0 = zeros(n_states(m), n_states(m))
        Γ1 = zeros(n_states(m), n_states(m))
        C  = zeros(n_states(m))
        Ψ  = zeros(n_states(m), n_shocks_exogenous(m))
        Π  = zeros(n_states(m), n_shocks_expectational(m))

        ### ENDOGENOUS STATES ###


        ### 1. Consumption Euler Equation

        Γ0[eq[:eq_euler], endo[:y_t]]  = 1.
        Γ0[eq[:eq_euler], endo[:R_t]] = 1/m[:τ]
        Γ0[eq[:eq_euler], endo[:g_t]] = -(1-m[:ρ_g])
        Γ0[eq[:eq_euler], endo[:z_t]] = -m[:ρ_z]/m[:τ]
        Γ0[eq[:eq_euler], endo[:Ey_t1]] = -1
        Γ0[eq[:eq_euler], endo[:Eπ_t1]] = -1/m[:τ]

        ### 2. NK Phillips Curve

        Γ0[eq[:eq_phillips], endo[:y_t]] = -m[:κ]
        Γ0[eq[:eq_phillips], endo[:π_t]] = 1
        Γ0[eq[:eq_phillips], endo[:g_t]] = m[:κ]
        Γ0[eq[:eq_phillips], endo[:Eπ_t1]] = -1/(1+m[:rA]/400)

        ### 3. Monetary Policy Rule

        Γ0[eq[:eq_mp], endo[:y_t]] = -(1-m[:ρ_R])*m[:ψ_2]
        Γ0[eq[:eq_mp], endo[:π_t]] = -(1-m[:ρ_R])*m[:ψ_1]
        Γ0[eq[:eq_mp], endo[:R_t]] = 1
        Γ0[eq[:eq_mp], endo[:g_t]] = (1-m[:ρ_R])*m[:ψ_2]
        Γ1[eq[:eq_mp], endo[:R_t]] = m[:ρ_R]
        if standard_normal_exog_shocks
            Ψ[eq[:eq_mp], exo[:rm_sh]] = m[:σ_R]
        else
            Ψ[eq[:eq_mp], exo[:rm_sh]] = 1
        end

        ### 4. Output lag

        Γ0[eq[:eq_y_t1], endo[:y_t1]] = 1
        Γ1[eq[:eq_y_t1], endo[:y_t]] = 1

        ### 5. Government spending

        Γ0[eq[:eq_g], endo[:g_t]] = 1
        Γ1[eq[:eq_g], endo[:g_t]] = m[:ρ_g]
        if standard_normal_exog_shocks
            Ψ[eq[:eq_g], exo[:g_sh]] = m[:σ_g]
        else
            Ψ[eq[:eq_g], exo[:g_sh]] = 1
        end

        ### 6. Technology

        Γ0[eq[:eq_z], endo[:z_t]] = 1
        Γ1[eq[:eq_z], endo[:z_t]] = m[:ρ_z]
        if standard_normal_exog_shocks
            Ψ[eq[:eq_z], exo[:z_sh]] = m[:σ_z]
        else
            Ψ[eq[:eq_z], exo[:z_sh]] = 1
        end


        ### 7. Expected output

        Γ0[eq[:eq_Ey], endo[:y_t]] = 1
        Γ1[eq[:eq_Ey], endo[:Ey_t1]] = 1
        Π[eq[:eq_Ey], ex[:Ey_sh]] = 1

        ### 8. Expected inflation

        Γ0[eq[:eq_Eπ], endo[:π_t]] = 1
        Γ1[eq[:eq_Eπ], endo[:Eπ_t1]] = 1
        Π[eq[:eq_Eπ], ex[:Eπ_sh]] = 1

        return Γ0, Γ1, C, Ψ, Π
    elseif method == :klein

        endo = m.endogenous_states
        exo  = m.exogenous_shocks
        eq   = m.equilibrium_conditions
        n_endo = get_setting(m, :n_endogenous_states_klein)
        n_exo  = n_shocks_exogenous(m)

        Γ0 = zeros(Real, n_endo, n_endo)
        Γ1 = zeros(Real, n_endo, n_endo)
        Γ2 = zeros(Real, n_endo, n_endo)
        Γ3 = zeros(Real, n_endo, n_exo)

        ### 1. Consumption Euler Equation

        Γ0[eq[:eq_euler], endo[:y_t]] = 1.
        Γ0[eq[:eq_euler], endo[:R_t]] = 1. / m[:τ]
        Γ0[eq[:eq_euler], endo[:g_t]] = -(1-m[:ρ_g])
        Γ0[eq[:eq_euler], endo[:z_t]] = -m[:ρ_z]/m[:τ]
        Γ1[eq[:eq_euler], endo[:y_t]] = 1.
        Γ1[eq[:eq_euler], endo[:π_t]] = 1. / m[:τ]

        ### 2. NK Phillips Curve

        Γ0[eq[:eq_phillips], endo[:y_t]] = -m[:κ]
        Γ0[eq[:eq_phillips], endo[:π_t]] = 1.
        Γ0[eq[:eq_phillips], endo[:g_t]] = m[:κ]
        Γ1[eq[:eq_phillips], endo[:π_t]] = 1. / (1+m[:rA]/400)

        ### 3. Monetary Policy Rule

        Γ0[eq[:eq_mp], endo[:y_t]] = -(1-m[:ρ_R])*m[:ψ_2]
        Γ0[eq[:eq_mp], endo[:π_t]] = -(1-m[:ρ_R])*m[:ψ_1]
        Γ0[eq[:eq_mp], endo[:R_t]] = 1.
        Γ0[eq[:eq_mp], endo[:g_t]] = (1-m[:ρ_R])*m[:ψ_2]
        Γ2[eq[:eq_mp], endo[:R_t]] = m[:ρ_R]
        Γ3[eq[:eq_mp], exo[:rm_sh]] = m[:σ_R]

        ### 4. Output lag

        Γ0[eq[:eq_y_t1], endo[:y_t1]] = 1.
        Γ2[eq[:eq_y_t1], endo[:y_t]] = 1.

        ### 5. Government spending

        Γ0[eq[:eq_g], endo[:g_t]] = 1.
        Γ2[eq[:eq_g], endo[:g_t]] = m[:ρ_g]
        Γ3[eq[:eq_g], exo[:g_sh]] = m[:σ_g]

        ### 6. Technology

        Γ0[eq[:eq_z], endo[:z_t]] = 1.
        Γ2[eq[:eq_z], endo[:z_t]] = m[:ρ_z]
        Γ3[eq[:eq_z], exo[:z_sh]] = m[:σ_z]

        return Γ0, Γ1, Γ2, Γ3
    end
end
