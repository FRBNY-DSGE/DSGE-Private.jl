"""
```
eqcond(m::SmallLinear)
```

Expresses the equilibrium conditions in canonical form using Γ0, Γ1, C, Ψ, and Π matrices.
Using the mappings of states/equations to integers defined in small_linear.jl, coefficients are
specified in their proper positions.

### Outputs

* `Γ0` (`n_states` x `n_states`) holds coefficients of current time states.
* `Γ1` (`n_states` x `n_states`) holds coefficients of lagged states.
* `C`  (`n_states` x `1`) is a vector of constants
* `Ψ`  (`n_states` x `n_shocks_exogenous`) holds coefficients of iid shocks.
* `Π`  (`n_states` x `n_states_expectational`) holds coefficients of expectational states.
"""
function eqcond(m::SmallLinear)
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

    ### 1. Home c_t (77)

    Γ0[eq[:eq_c_t], endo[:c_t]]  = 1
    Γ0[eq[:eq_c_t], endo[:r_n_t]]  = m[:σ]
    Γ0[eq[:eq_c_t], endo[:Eπ_ct1]]  = -m[:σ]
    Γ0[eq[:eq_c_t], endo[:Ec_t1]]  = -1

    ### 2. Home w_t (78)

    Γ0[eq[:eq_w_t], endo[:w_t]]  = 1
    Γ0[eq[:eq_w_t], endo[:l_t]]  = -m[:𝛘]
    Γ0[eq[:eq_w_t], endo[:c_t]]  = -1/m[:σ]

    ### 3. Home c_dt (79)

    Γ0[eq[:eq_c_dt], endo[:c_dt]]  = 1
    Γ0[eq[:eq_c_dt], endo[:𝜏_t]]  = -m[:η]*m[:ω]
    Γ0[eq[:eq_c_dt], endo[:c_t]]  = -1

    ### 4. Home m_ct (80)

    Γ0[eq[:eq_m_ct], endo[:m_ct]]  = 1
    Γ0[eq[:eq_m_ct], endo[:𝜏_t]]  = m[:η]*(1-m[:ω])
    Γ0[eq[:eq_m_ct], endo[:c_t]]  = -1

    ### 5. Home y_t_1 (81)

    Γ0[eq[:eq_y_t_1], endo[:y_t]]  = 1
    Γ0[eq[:eq_y_t_1], endo[:l_t]]  = -(1-m[:α])
    Γ0[eq[:eq_y_t_1], endo[:z_t]]  = -(1-m[:α])

    ### 6. Home mc_t (82)

    Γ0[eq[:eq_mc_t], endo[:mc_t]]  = 1
    Γ0[eq[:eq_mc_t], endo[:w_t]]  = -1
    Γ0[eq[:eq_mc_t], endo[:l_t]]  = -m[:α]

    ### 7. Home π_t (83)

    Γ0[eq[:eq_π_t], endo[:π_t]]  = 1
    Γ0[eq[:eq_π_t], endo[:mc_t]]  = -(1-m[:β]*m[:ξ_p])*(1-m[:ξ_p])/m[:ξ_p]
    Γ0[eq[:eq_π_t], endo[:𝜏_t]]  = -m[:ω]*(1-m[:β]*m[:ξ_p])*(1-m[:ξ_p])/m[:ξ_p]
    Γ0[eq[:eq_π_t], endo[:Eπ_t1]]  = -m[:β]

    ### 8. Home π_ct (84)

    Γ0[eq[:eq_π_ct], endo[:π_ct]]  = 1
    Γ0[eq[:eq_π_ct], endo[:π_t]]  = -1
    Γ0[eq[:eq_π_ct], endo[:𝜏_t]]  = -m[:ω]
    Γ1[eq[:eq_π_ct], endo[:𝜏_t]]  = -m[:ω]

    ### 9. Home y_t_2 (85)

    Γ0[eq[:eq_y_t_2], endo[:y_t]]  = 1
    Γ0[eq[:eq_y_t_2], endo[:c_dt]]  = -(1-m[:ω])
    Γ0[eq[:eq_y_t_2], endo[:m_ct_f]]  = -m[:ω]

    ### 10. Home r_n_t (86)

    Γ0[eq[:eq_r_n_t], endo[:r_n_t]]  = 1
    Γ0[eq[:eq_r_n_t], endo[:π_t]]  = -(1-m[:γ_r])*m[:γ_π]
    Γ0[eq[:eq_r_n_t], endo[:e_rt]] = -1
    Γ1[eq[:eq_r_n_t], endo[:r_n_t]]  = m[:γ_r]

    ### 11. Home Monetary Shock

    Γ0[eq[:eq_e_rt], endo[:e_rt]]  = 1
    Γ1[eq[:eq_e_rt], endo[:e_rt]]  = m[:ρ_m]
    Ψ[eq[:eq_e_rt], exo[:rm_sh]] = m[:σ_m]

    ### 12. Home TFP Shock
    Γ0[eq[:eq_z_t], endo[:z_t]]  = 1
    Γ1[eq[:eq_z_t], endo[:z_t]]  = m[:ρ_z]
    Ψ[eq[:eq_z_t], exo[:z_sh]] = m[:σ_z]

    ### 13. Expected Shock Eπ_ct
    Γ0[eq[:eq_Eπ_ct], endo[:π_ct]]  = 1
    Γ1[eq[:eq_Eπ_ct], endo[:Eπ_ct1]]  = 1
    Π[eq[:eq_Eπ_ct], ex[:Eπ_ct_sh]] = 1

    ### 14. Expected Shock Ec_t
    Γ0[eq[:eq_Ec], endo[:c_t]]  = 1
    Γ1[eq[:eq_Ec], endo[:Ec_t1]]  = 1
    Π[eq[:eq_Ec], ex[:Ec_sh]] = 1

    ### 15. Expected Shock Eπ_t
    Γ0[eq[:eq_Eπ_t], endo[:π_t]]  = 1
    Γ1[eq[:eq_Eπ_t], endo[:Eπ_t1]]  = 1
    Π[eq[:eq_Eπ_t], ex[:Eπ_t_sh]] = 1

    ### 16. Home Output Lag
    Γ0[eq[:eq_y_t1], endo[:y_t1]]  = 1
    Γ1[eq[:eq_y_t1], endo[:y_t]]  = 1

    #####

    ### 1. Common c_t (87)

    Γ0[eq[:eq_c_t_s], endo[:c_t]]  = 1
    Γ0[eq[:eq_c_t_s], endo[:c_t_f]]  = -1
    Γ0[eq[:eq_c_t_s], endo[:𝜏_t]]  = -m[:σ]*(1-2*m[:ω])
    Γ0[eq[:eq_c_t_s], endo[:uip_t]]  = 1

    ### 2. UIP Shock
    Γ0[eq[:eq_uip_t], endo[:uip_t]]  = 1
    Γ1[eq[:eq_uip_t], endo[:uip_t]]  = m[:ρ_uip]
    Ψ[eq[:eq_uip_t], exo[:uip_sh]] = m[:σ_uip]

    #####

    ### 1. Foreign c_t (88)

    Γ0[eq[:eq_c_t_f], endo[:c_t_f]]  = 1
    Γ0[eq[:eq_c_t_f], endo[:r_n_t_f]]  = m[:σ]
    Γ0[eq[:eq_c_t_f], endo[:Eπ_ct1_f]]  = -m[:σ]
    Γ0[eq[:eq_c_t_f], endo[:Ec_t1_f]]  = -1

    ### 2. Foreign w_t (89)

    Γ0[eq[:eq_w_t_f], endo[:w_t_f]]  = 1
    Γ0[eq[:eq_w_t_f], endo[:l_t_f]]  = -m[:𝛘]
    Γ0[eq[:eq_w_t_f], endo[:c_t_f]]  = -1/m[:σ]

    ### 3. Foreign c_dt (90)

    Γ0[eq[:eq_c_dt_f], endo[:c_dt_f]]  = 1
    Γ0[eq[:eq_c_dt_f], endo[:𝜏_t]]  = m[:η]*m[:ω]
    Γ0[eq[:eq_c_dt_f], endo[:c_t_f]]  = -1

    ### 4. Foreign m_ct (91)

    Γ0[eq[:eq_m_ct_f], endo[:m_ct_f]]  = 1
    Γ0[eq[:eq_m_ct_f], endo[:𝜏_t]]  = -m[:η]*(1-m[:ω])
    Γ0[eq[:eq_m_ct_f], endo[:c_t_f]]  = -1

    ### 5. Foreign y_t_1 (92)

    Γ0[eq[:eq_y_t_1_f], endo[:y_t_f]]  = 1
    Γ0[eq[:eq_y_t_1_f], endo[:l_t_f]]  = -(1-m[:α])
    Γ0[eq[:eq_y_t_1_f], endo[:z_t_f]]  = -(1-m[:α])

    ### 6. Foreign mc_t (93)

    Γ0[eq[:eq_mc_t_f], endo[:mc_t_f]]  = 1
    Γ0[eq[:eq_mc_t_f], endo[:w_t_f]]  = -1
    Γ0[eq[:eq_mc_t_f], endo[:l_t_f]]  = -m[:α]

    ### 7. Foreign π_t (94)

    Γ0[eq[:eq_π_t_f], endo[:π_t_f]]  = 1
    Γ0[eq[:eq_π_t_f], endo[:mc_t_f]]  = -(1-m[:β]*m[:ξ_p])*(1-m[:ξ_p])/m[:ξ_p]
    Γ0[eq[:eq_π_t_f], endo[:𝜏_t]]  = m[:ω]*(1-m[:β]*m[:ξ_p])*(1-m[:ξ_p])/m[:ξ_p]
    Γ0[eq[:eq_π_t_f], endo[:Eπ_t1_f]]  = -m[:β]

    ### 8. Foreign π_ct (95)

    Γ0[eq[:eq_π_ct_f], endo[:π_ct_f]]  = 1
    Γ0[eq[:eq_π_ct_f], endo[:π_t_f]]  = -1
    Γ0[eq[:eq_π_ct_f], endo[:𝜏_t]]  = m[:ω]
    Γ1[eq[:eq_π_ct_f], endo[:𝜏_t]]  = m[:ω]

    ### 9. Foreign y_t_2 (96)

    Γ0[eq[:eq_y_t_2_f], endo[:y_t_f]]  = 1
    Γ0[eq[:eq_y_t_2_f], endo[:c_dt_f]]  = -(1-m[:ω])
    Γ0[eq[:eq_y_t_2_f], endo[:m_ct]]  = -m[:ω]

    ### 10. Foreign r_n_t (97)

    Γ0[eq[:eq_r_n_t_f], endo[:r_n_t_f]]  = 1
    Γ0[eq[:eq_r_n_t_f], endo[:π_t_f]]  = -(1-m[:γ_r])*m[:γ_π]
    Γ0[eq[:eq_r_n_t_f], endo[:e_f_rt]] = -1
    Γ1[eq[:eq_r_n_t_f], endo[:r_n_t_f]]  = m[:γ_r]

    ### 11. Foreign Monetary Shock

    Γ0[eq[:eq_e_f_rt], endo[:e_f_rt]] = 1
    Γ1[eq[:eq_e_f_rt], endo[:e_f_rt]] = m[:ρ_m]
    Ψ[eq[:eq_e_f_rt], exo[:rm_f_sh]] = m[:σ_m]

    ### 12. Foreign TFP Shock
    Γ0[eq[:eq_z_t_f], endo[:z_t_f]]  = 1
    Γ1[eq[:eq_z_t_f], endo[:z_t_f]]  = m[:ρ_z]
    Ψ[eq[:eq_z_t_f], exo[:z_f_sh]] = m[:σ_z]

    ### 13. Foreign Expected Shock Eπ_ct
    Γ0[eq[:eq_Eπ_ct_f], endo[:π_ct_f]]  = 1
    Γ1[eq[:eq_Eπ_ct_f], endo[:Eπ_ct1_f]]  = 1
    Π[eq[:eq_Eπ_ct_f], ex[:Eπ_ct_f_sh]] = 1

    ### 14. Foreign Expected Shock Ec_t
    Γ0[eq[:eq_Ec_f], endo[:c_t_f]]  = 1
    Γ1[eq[:eq_Ec_f], endo[:Ec_t1_f]]  = 1
    Π[eq[:eq_Ec_f], ex[:Ec_f_sh]] = 1

    ### 15. Foreign Expected Shock Eπ_t
    Γ0[eq[:eq_Eπ_t_f], endo[:π_t_f]]  = 1
    Γ1[eq[:eq_Eπ_t_f], endo[:Eπ_t1_f]]  = 1
    Π[eq[:eq_Eπ_t_f], ex[:Eπ_t_f_sh]] = 1

    ### 16. Foreign Output Lag
    Γ0[eq[:eq_y_t1_f], endo[:y_t1_f]]  = 1
    Γ1[eq[:eq_y_t1_f], endo[:y_t_f]]  = 1

    return Γ0, Γ1, C, Ψ, Π
end
