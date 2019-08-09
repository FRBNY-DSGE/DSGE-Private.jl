# Expresses the equilibrium conditions in canonical form using Γ0, Γ1, C, Ψ, and Π matrices.
# Using the mappings of states/equations to integers defined in m990.jl, coefficients are
# specified in their proper positions.

# Γ0 (n_states x n_states) holds coefficients of current time states.
# Γ1 (n_states x n_states) holds coefficients of lagged states.
# C  (n_states x 1) is a vector of constants
# Ψ  (n_states x n_shocks_exogenous) holds coefficients of iid shocks.
# Π  (n_states x n_states_expectational) holds coefficients of expectational states.

function eqcond(m::GHLS)
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

    # Sticky prices and wages
    Γ0[eq[:eq_euler], endo[:c_t]]    = -m[:β] * m[:gamtil]^2 - 1.0
    Γ0[eq[:eq_euler], endo[:Ec_t]]   = m[:β] * m[:gamtil]
    Γ0[eq[:eq_euler], endo[:λ]] = -(1.0-m[:gamtil])*(1.0-m[:β]*m[:gamtil])
    Γ0[eq[:eq_euler], endo[:z_t]] = -1.0*m[:gamtil]
    Γ0[eq[:eq_euler], endo[:Ez_t]] = m[:β] * m[:gamtil]
    Γ1[eq[:eq_euler], endo[:c_t]]    = -1.0*m[:gamtil]

    ### 2. Investment Euler Equation

    # Sticky prices and wages
    Γ0[eq[:eq_inv], endo[:k_t]] = 1.0
    Γ0[eq[:eq_inv], endo[:i_t]] = (1.0-m[:δ])/m[:gz] - 1.0
    Γ0[eq[:eq_inv], endo[:μ_t]] = (1.0-m[:δ])/m[:gz] - 1.
    Γ0[eq[:eq_inv], endo[:z_t]] = (1.0-m[:δ])/m[:gz]
    Γ1[eq[:eq_inv], endo[:k_t]] = (1.0-m[:δ])/m[:gz]

    ### 3. Value of Capital

    # Sticky prices and wages
    Γ0[eq[:eq_capval], endo[:λ]] = -1.0
    Γ0[eq[:eq_capval], endo[:qk_t]] = -1.0
    Γ0[eq[:eq_capval], endo[:Erk_t]] = -1.0*m[:β]*1.0/m[:gz]*(-1.0*m[:δ] + 1.0) + 1.0
    Γ0[eq[:eq_capval], endo[:Ez_t]] = -1
    Γ0[eq[:eq_capval], endo[:Eqk_t]] = m[:β]*1.0/m[:gz]*(-1.0*m[:δ] + 1.0)
    Γ0[eq[:eq_capval], endo[:Eλ_t]] = 1.0

    ### 4. Aggregate Production Function

    # Sticky prices and wages
    Γ0[eq[:eq_output], endo[:c_t]] = m[:gg]*m[:shrcy]
    Γ0[eq[:eq_output], endo[:i_t]] = m[:gg]*m[:shriy]
    Γ0[eq[:eq_output], endo[:y_t]] = -1.0
    Γ0[eq[:eq_output], endo[:u_t]] = m[:α]*1.0/m[:ϵ_p]*m[:gg]*(m[:ϵ_p] - 1.0)
    Γ0[eq[:eq_output], endo[:g_t]] = 1.0

    ### 5. Capital Utilization

    # Sticky prices and wages
    Γ0[eq[:eq_caputil], endo[:u_t]] = -1.0
    Γ0[eq[:eq_caputil], endo[:rk_t]] = 1.0/m[:σ_a]

    ### 6. Rental Rate of Capital

    # Sticky prices and wages
    Γ0[eq[:eq_capsrv], endo[:w_t]] = 1.0 # This is real wage
    Γ0[eq[:eq_capsrv], endo[:N_t]] = 1.0
    Γ0[eq[:eq_capsrv], endo[:u_t]] = -1.0
    Γ0[eq[:eq_capsrv], endo[:rk_t]] = -1.0
    Γ0[eq[:eq_capsrv], endo[:z_t]] = 1.0 # Techshock

    Γ1[eq[:eq_capsrv], endo[:k_t]] = 1.0

    ### 7. Evolution of Capital

    ### 8. Price Markup

    ### 9. Phillips Curve

    # Sticky prices and wages
    Γ0[eq[:eq_phlps], endo[:π_t]] = -1.0
    Γ0[eq[:eq_phlps], endo[:mc_t]] = m[:κ_p]
    Γ0[eq[:eq_phlps], endo[:Eπ_t]] = m[:β]*1.0/(m[:β]*(-1.0*m[:ap] + 1.0) + 1.0)

    Γ1[eq[:eq_phlps], endo[:π_t]] = -1.0*(-1.0*m[:ap] + 1.0)*1.0/(m[:β]*(-1.0*m[:ap] + 1.0) + 1.0)

    ### 10. Rental Rate of Capital

    ### 11. Marginal Substitution

    ### 12. Evolution of Wages

    # Sticky prices and wages
    Γ0[eq[:eq_wage], endo[:w_t]] = -1.0
    Γ0[eq[:eq_wage], endo[:π_t]] = -1.0
    Γ0[eq[:eq_wage], endo[:π_w]] = 1.0
    Γ0[eq[:eq_wage], endo[:z_t]] = -1.0

    Γ1[eq[:eq_wage], endo[:w_t]] = -1.0

    ### 13. Monetary Policy Rule
    ### Notional Rate in Gust Et Al.

    # Sticky prices and wages
    Γ0[eq[:eq_mpnot], endo[:Rnotional_t]] = -1.0
    Γ0[eq[:eq_mpnot], endo[:π_t]] = m[:γ_π]*(-1.0*m[:ρ_R] + 1.0)
    Γ0[eq[:eq_mpnot], endo[:y_t]] = m[:γ_g]*(-1.0*m[:ρ_R] + 1.0)
    Γ0[eq[:eq_mpnot], endo[:x_t]] = m[:γ_x]*(-1.0*m[:ρ_R] + 1.0)
    Γ0[eq[:eq_mpnot], endo[:z_t]] = m[:γ_g]*(-1.0*m[:ρ_R] + 1.0)
    Γ0[eq[:eq_mpnot], endo[:Rnotionalshock_t]] = 1.0

    Γ1[eq[:eq_mpnot], endo[:Rnotional_t]] = -1.0*m[:ρ_R]
    Γ1[eq[:eq_mpnot], endo[:y_t]] = m[:γ_g]*(-1.0*m[:ρ_R] + 1.0)

    ### 13.5 Nominal Interest Rate
    Γ0[eq[:eq_mpnom], endo[:Rnotional_t]] = 1.0
    Γ0[eq[:eq_mpnom], endo[:Rnominal_t]] = -1.0

    ### 14. Resource Constraint

    ### 15. Output Gap
    Γ0[eq[:eq_outgap], endo[:x_t]] = -1.0
    Γ0[eq[:eq_outgap], endo[:N_t]] = -1.0*m[:α] + 1.0
    Γ0[eq[:eq_outgap], endo[:u_t]] = m[:α]

    ### 16. Tobin's Q
    Γ0[eq[:eq_tobq], endo[:i_t]] = m[:phii_jpt]*(m[:β]+1.0)
    Γ0[eq[:eq_tobq], endo[:qk_t]] = -1.0
    Γ0[eq[:eq_tobq], endo[:μ_t]] = -1.0
    Γ0[eq[:eq_tobq], endo[:z_t]] = m[:phii_jpt]
    Γ0[eq[:eq_tobq], endo[:Ei_t]] = -1.0*m[:β]*m[:phii_jpt]
    Γ0[eq[:eq_tobq], endo[:Ez_t]] = -1.0*m[:β]*m[:phii_jpt]

    Γ1[eq[:eq_tobq], endo[:i_t]] = m[:phii_jpt]


    ### 17. Hours Worked
    Γ0[eq[:eq_N], endo[:y_t]] = 1.0
    Γ0[eq[:eq_N], endo[:N_t]] = m[:α] - 1.0
    Γ0[eq[:eq_N], endo[:u_t]] = -1.0*m[:α]
    Γ0[eq[:eq_N], endo[:z_t]] = m[:α]
    Γ0[eq[:eq_N], endo[:a_t]] = m[:α]-1.0 # a shock

    Γ1[eq[:eq_N], endo[:k_t]] = m[:α]


    ### 18. Marginal Cost
    Γ0[eq[:eq_mcost], endo[:w_t]] = 1.0
    Γ0[eq[:eq_mcost], endo[:y_t]] = -1.0
    Γ0[eq[:eq_mcost], endo[:N_t]] = 1.0
    Γ0[eq[:eq_mcost], endo[:mc_t]] = -1.0


    ### 19. Marginal Utility of Consumption
    Γ0[eq[:eq_muc], endo[:c_t]] = -1.0
    Γ0[eq[:eq_muc], endo[:muc_t]] = m[:gamtil] -1.0
    Γ0[eq[:eq_muc], endo[:z_t]] = -1.0*m[:gamtil]

    Γ1[eq[:eq_muc], endo[:c_t]] = -1.0*m[:gamtil]


    ### 20. Lagged GDP
    Γ0[eq[:eq_laggdp], endo[:y_t1]] = -1.0 # y
    Γ1[eq[:eq_laggdp], endo[:y_t]] = -1.0


    ### 21. Lagged Consumption
    Γ0[eq[:eq_lagc], endo[:c_t1]] = -1.0
    Γ1[eq[:eq_lagc], endo[:c_t]] = -1.0


    ### 22. Lagged Investment
    Γ0[eq[:eq_lagi], endo[:i_t1]] = -1.0
    Γ1[eq[:eq_lagi], endo[:i_t]] = -1.0


    ### 23. Lagged Wage (Zeta)
    Γ0[eq[:eq_lagwage], endo[:w_t1]] = -1.0
    Γ1[eq[:eq_lagwage], endo[:w_t]] = -1.0


    ### 24. Liquidity Shock (η Shock?) - Unneeded, repeated later
    # Γ0[eq[:eq_liqshk], endo[:b_t]] = -1.0 # b_t is b_t shock, coming from etashock
    # Γ1[eq[:eq_liqshk], endo[:η_t]] = -0.85 # Same as above


    ### 25. Value of Consumption (λ)
    Γ0[eq[:eq_λ], endo[:Rnotional_t]] = 1.0
    Γ0[eq[:eq_λ], endo[:λ]] = -1.0
    Γ0[eq[:eq_λ], endo[:η_t]] = 1.0 # η_t is η shock here.
    Γ0[eq[:eq_λ], endo[:Ez_t]] = -1.0
    Γ0[eq[:eq_λ], endo[:Eπ_t]] = -1.0
    Γ0[eq[:eq_λ], endo[:Eλ_t]] = 1.0


    ### 26. Wage Inflation
    Γ0[eq[:eq_π_w], endo[:Vw_t]] = 1.0
    Γ0[eq[:eq_π_w], endo[:π_w]] = -1.0
    Γ0[eq[:eq_π_w], endo[:z_t]] = -1.0*m[:bw] + 1.0 # Techshock
    Γ1[eq[:eq_π_w], endo[:π_t]] = m[:aw] - 1.0


    ### 27. V_i
    Γ0[eq[:eq_vi], endo[:i_t]] = 1.0
    Γ0[eq[:eq_vi], endo[:Vi_t]] = -1.0
    Γ0[eq[:eq_vi], endo[:z_t]] = 1.0
    Γ1[eq[:eq_vi], endo[:i_t]] = 1.0


    ### 28. V_p
    Γ0[eq[:eq_vp], endo[:π_t]] = 1.0
    Γ0[eq[:eq_vp], endo[:Vp_t]] = -1.0
    Γ1[eq[:eq_vp], endo[:π_t]] = 1.0 - m[:ap]


    ### 29. V_w
    Γ0[eq[:eq_vw], endo[:w_t]] = -1.0 * m[:κ_w]
    Γ0[eq[:eq_vw], endo[:λ]] = -1.0 * m[:κ_w]
    Γ0[eq[:eq_vw], endo[:N_t]] = m[:κ_w] * m[:σ_L]
    Γ0[eq[:eq_vw], endo[:Vw_t]] = -1.0
    Γ0[eq[:eq_vw], endo[:EVw_t]] = m[:β]


    ### 30. bc
    Γ0[eq[:eq_bc], endo[:c_t]] = m[:gamtil] * 1.0/(1.0 - m[:gamtil])
    Γ0[eq[:eq_bc], endo[:bc_t]] = -1.0
    Γ0[eq[:eq_bc], endo[:Ec_t]] = -1.0*1.0/(-1.0*m[:gamtil] + 1.0)
    Γ0[eq[:eq_bc], endo[:Ez_t]] = -1.0*m[:gamtil]*1.0/(-1.0*m[:gamtil] + 1.0)


    ### 31. bi
    Γ0[eq[:eq_bi], endo[:i_t]] = -1.0
    Γ0[eq[:eq_bi], endo[:bi_t]] = -1.0
    Γ0[eq[:eq_bi], endo[:Ei_t]] = 1.0
    Γ0[eq[:eq_bi], endo[:Ez_t]] = 1.0


    ### EXOGENOUS SHOCKS ###

    # Technology
    Γ0[eq[:eq_z], endo[:z_t]] = -1.0
    Ψ[eq[:eq_z], exo[:z_sh]]     = -1.0

    # Government spending
    Γ0[eq[:eq_g], endo[:g_t]] = -1.0
    Γ1[eq[:eq_g], endo[:g_t]] = -1.0*m[:ρ_g]
    Ψ[eq[:eq_g], exo[:g_sh]]  = -1.0

    # Asset shock (treat as liquidity/η shock)
    Γ0[eq[:eq_η], endo[:η_t]] = -1.0
    Γ1[eq[:eq_η], endo[:η_t]] = -0.85
    Ψ[eq[:eq_η], exo[:η_sh]]  = -1.0

    # Investment-specific technology
    Γ0[eq[:eq_μ], endo[:μ_t]] = -1.0
    Γ1[eq[:eq_μ], endo[:μ_t]] = -1.0*m[:ρ_μ]
    Ψ[eq[:eq_μ], exo[:μ_sh]]  = -1.0

    # Monetary policy shock
    ### Putting interest rate shock in here:
    Γ0[eq[:eq_Rnotionalshock], endo[:Rnotionalshock_t]] = -1.0
    Γ1[eq[:eq_Rnotionalshock], endo[:Rnotionalshock_t]] = -1.0*m[:ρ_int]
    Ψ[eq[:eq_Rnotionalshock], exo[:Rnotional_sh]]  = -1.0

    # Elast Shock
    Γ0[eq[:eq_elast], endo[:elast_t]] = -1.0
    Γ1[eq[:eq_elast], endo[:elast_t]] = -1.0*m[:ρ_elast]
    Ψ[eq[:eq_elast], exo[:elast_sh]]  = -1.

    # Elastw Shock
    Γ0[eq[:eq_elastw], endo[:elastw_t]] = -1.0
    Γ1[eq[:eq_elastw], endo[:elastw_t]] = -1.0*m[:ρ_elastw]
    Ψ[eq[:eq_elastw], exo[:elastw_sh]]  = -1.0


    # A Shock
    Γ0[eq[:eq_a], endo[:a_t]] = -1.0
    Ψ[eq[:eq_a], exo[:a_sh]]  = -1.0

    ### EXPECTATION ERRORS ###

    ### E(c)

    # Sticky prices and wages
    Γ0[eq[:eq_Ec], endo[:c_t]]         = -1.0
    Γ1[eq[:eq_Ec], endo[:Ec_t]]        = -1.0
    Π[eq[:eq_Ec], ex[:Ec_sh]]          = -1.0

    ### E(q)

    # Sticky prices and wages
    Γ0[eq[:eq_Eqk], endo[:qk_t]]       = -1.0
    Γ1[eq[:eq_Eqk], endo[:Eqk_t]]      = -1.0
    Π[eq[:eq_Eqk], ex[:Eqk_sh]]        = -1.0

    ### E(i)

    # Sticky prices and wages
    Γ0[eq[:eq_Ei], endo[:i_t]]         = -1.0
    Γ1[eq[:eq_Ei], endo[:Ei_t]]         = -1.0
    Π[eq[:eq_Ei], ex[:Ei_sh]]          = -1.0

    ### E(π)

    # Sticky prices and wages
    Γ0[eq[:eq_Eπ], endo[:π_t]]       = -1.0
    Γ1[eq[:eq_Eπ], endo[:Eπ_t]]       = -1.0
    Π[eq[:eq_Eπ], ex[:Eπ_sh]]        = -1.0

    ### E(l)

    ### E(rk)

    # Sticky prices and wages
    Γ0[eq[:eq_Erk], endo[:rk_t]]       = -1.0
    Γ1[eq[:eq_Erk], endo[:Erk_t]]       = -1.0
    Π[eq[:eq_Erk], ex[:Erk_sh]]        = -1.0

    ### E(w)

    ### E(techshock) (Added)
    Γ0[eq[:eq_Ez], endo[:z_t]]         = -1.
    Γ1[eq[:eq_Ez], endo[:Ez_t]]         = -1.
    Π[eq[:eq_Ez], ex[:Ez_sh]]          = -1.


    ### E(λ_t) (Added)
    Γ0[eq[:eq_Eλ], endo[:λ]]         = -1.
    Γ1[eq[:eq_Eλ], endo[:Eλ_t]]         = -1.
    Π[eq[:eq_Eλ], ex[:Eλ_sh]]          = -1.


    ### E(VW) (Added)
    Γ0[eq[:eq_EVw], endo[:Vw_t]]         = -1.
    Γ1[eq[:eq_EVw], endo[:EVw_t]]         = -1.
    Π[eq[:eq_EVw], ex[:EVw_sh]]          = -1.

    return Γ0, Γ1, C, Ψ, Π
end
