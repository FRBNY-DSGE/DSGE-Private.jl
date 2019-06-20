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
    Γ0[eq[:eq_euler], endo[:lamc]] = -(1.0-m[:gamtil])*(1.0-m[:β]*m[:gamtil]) # Doesn't work: need to figure out what lamc is in DSGE
    Γ0[eq[:eq_euler], endo[:ztil_t]] = -1.0*m[:gamtil]
    Γ0[eq[:eq_euler], endo[:Etechshock_t]] = m[:β] * m[:gamtil] # Doesn't work: need to igure out what E[techshock] is in DSGE
    Γ1[eq[:eq_euler], endp[:c_t]]    = -1.0*m[:gamtil]

    ### 2. Investment Euler Equation

    # Sticky prices and wages
    Γ0[eq[:eq_inv], endo[:k_t]] = 1.0 # Not sure if this is right, using kp
    Γ0[eq[:eq_inv], endo[:i_t]] = (1.0-m[:δ])/m[:gz] - 1.0 # I want Gz, maybe z_star is right
    Γ0[eq[:eq_inv], endo[:μ_t]] = (1.0-m[:δ])/m[:gz] - 1.0 # I want invshock and Gz, not sure if μ_t and z_star is right
    Γ0[eq[:eq_inv], endo[:ztil_t]] = (1.0-m[:δ])/m[:gz] # I want Gz and techshock, not sure if z_star and ztil_t are right
    Γ1[eq[:eq_inv], endo[:k_t]] = (1.0-m[:δ])/m[:gz] # I want kp and Gz, not sure if k_t and z_star is right
eq_inv_f], endo[:z_t]]    = 1/(1 + m[:β]*exp((1 - m[:σ_c])*m[:zstar]

    ### 3. Value of Capital

    # Sticky prices and wages
    Γ0[eq[:eq_capval], endo[:lamc]] = -1.0 # What is lamc? Value placed on consumption?
    Γ0[eq[:eq_capval], endo[:qk_t]] = -1.0 # tobq = qk_t?
    Γ0[eq[:eq_capval], endo[:rk_t]] = -1.0*m[:β]*1.0/m[:gz]*(-1.0*m[:δ] + 1.0) + 1.0
    Γ0[eq[:eq_capval], endo[:Etechshock_t]] = -1 # Etechshock_t is not a thing
    Γ0[eq[:eq_capval], endo[:Eqk_t]] = m[:β]*1.0/m[:gz]*(-1.0*m[:δ] + 1.0) # Expeted qk_t
    Γ0[eq[:eq_capval], endo[:Elamc_t]] = 1.0 # What is Elamc_t

    # Flexible prices and wages
    Γ0[eq[:eq_capval_f], endo[:lamc_f]] = -1.0 # What is lamc? Value placed on consumption?
    Γ0[eq[:eq_capval_f], endo[:qk_f_t]] = -1.0 # tobq = qk_t?
    Γ0[eq[:eq_capval_f], endo[:rk_f_t]] = -1.0*m[:β]*1.0/m[:gz]*(-1.0*m[:δ] + 1.0) + 1.0
    Γ0[eq[:eq_capval_f], endo[:Etechshock_f_t]] = -1 # Etechshock_t is not a thing
    Γ0[eq[:eq_capval_f], endo[:Eqk_f_t]] = m[:β]*1.0/m[:gz]*(-1.0*m[:δ] + 1.0) # Expeted qk_t
    Γ0[eq[:eq_capval_f], endo[:Elamc_f_t]] = 1.0 # What is Elamc_t

    ### 4. Aggregate Production Function

    # Sticky prices and wages
    Γ0[eq[:eq_output], endo[:c_t]] = m[:gg]*m[:shrcy]
    Γ0[eq[:eq_output], endo[:i_t]] = m[:gg]*m[:shriy]
    Γ0[eq[:eq_output], endo[:y_t]] = -1.0
    Γ0[eq[:eq_output], endo[:u_t]] = m[:α]*1.0/m[:ϵ]*m[:gg]*(m[:ϵ] - 1.0) # util = u_t?
    Γ0[eq[:eq_output], endo[:g_t]] = 1.0

    # Flexible prices and wages
    Γ0[eq[:eq_output_f], endo[:c_f_t]] = m[:gg]*m[:shrcy]
    Γ0[eq[:eq_output_f], endo[:i_f_t]] = m[:gg]*m[:shriy]
    Γ0[eq[:eq_output_f], endo[:y_f_t]] = -1.0
    Γ0[eq[:eq_output_f], endo[:u_f_t]] = m[:α]*1.0/m[:ϵ]*m[:gg]*(m[:ϵ] - 1.0) # util = u_t?
    Γ0[eq[:eq_output_f], endo[:g_t]] = 1.0

    ### 5. Capital Utilization

    # Sticky prices and wages
    Γ0[eq[:eq_caputil], endo[u_t]] = -1.0
    Γ0[eq[:eq_caputil], endo[rk_t]] = 1.0/m[:σ_a] # Note: σ_a doesn't exist in DSGE. It's stdev of cost of utilizing physical capital into capital services

    ### 6. Rental Rate of Capital

    # Sticky prices and wages
    Γ0[eq[:eq_capsrv], endo[:w_t]] = 1.0 # This is real wage
    Γ0[eq[:eq_capsrv], endo[:L_t]] = 1.0 # Assuming lab or le equals total hours L
    Γ0[eq[:eq_capsrv], endo[:u_t]] = -1.0
    Γ0[eq[:eq_capsrv], endo[:rk_t]] = -1.0
    Γ0[eq[:eq_capsrv], endo[:ztil_t]] = 1.0 # Tech shock

    Γ1[eq[:eq_capsrv], endo[:k_t]] = 1.0 # Assuming cap or kp is k_t

    ### 7. Evolution of Capital

    ### 8. Price Markup

    ### 9. Phillips Curve

    # Sticky prices and wages
    Γ0[eq[:eq_phlps], endo[:π_t]] = -1.0
    Γ0[eq[:eq_phlps], endo[:mc_t]] = m[:κp] #Check if this is what kappap is called
    Γ0[eq[:eq_phlps], endo[:Eπ_t]] = m[:β]*1.0/(m[:β]*(-1.0*m[:ap] + 1.0) + 1.0)

    Γ1[eq[:eq_phlps], endo[:π_t]] = -1.0*(-1.0*m[:ap] + 1.0)*1.0/(m[:β]*(-1.0*m[:ap] + 1.0) + 1.0)

    ### 10. Rental Rate of Capital

    ### 11. Marginal Substitution

    ### 12. Evolution of Wages

    # Sticky prices and wages
    Γ0[eq[:eq_wage], endo[:w_t]] = -1.0
    Γ0[eq[:eq_wage], endo[:π_t]] = -1.0
    Γ0[eq[:eq_wage], endo[:dw_t]] = -1.0 # DW is wage inflation
    Γ0[eq[:eq_wage], endo[:ztil_t]] = -1.0 # Techshock

    Γ1[eq[:eq_wage], endo[:w_t]] = -1.0

    ### 13. Monetary Policy Rule
    ### Notional Rate in Gust Et Al.

    # Sticky prices and wages
    Γ0[eq[:eq_mp], endo[:rm_t]] = 1.0 # This rm_t should be notional R_t
    Γ0[eq[:eq_mp], endo[:π_t]] = m[:gam_dp]*(-1.0*m[:gam_rs] + 1.0)
    Γ0[eq[:eq_mp], endo[:y_t]] = m[:gamdy]*(-1.0*m[:gam_rs] + 1.0)
    Γ0[eq[:eq_mp], endo[:ztil_t]] = m[:gamdy]*(-1.0*m[:gam_rs] + 1.0)
    Γ0[eq[:eq_mp], endo[:R_t]] = -1.0 # Interest Rate Shock

    Γ1[eq[:eq_mp], endo[:rm_t]] = -1.0*m[:gam_rs]
    Γ1[eq[:eq_mp], endo[:y_t]] = m[:gamdy]*(-1.0*m[:gam_rs] + 1.0)

    ### 13.5 Nominal Interest Rate
    Γ0[eq[:eq_mp], endo[:rm_t]] = 1.0 # rm_t should be notional R_t
    Γ0[eq[:eq_mp], endo[:R_t]] = -1.0 # Nominal Interest Rate (not shock)

    ### 14. Resource Constraint

    ### 15. Output Gap
    Γ0[eq[:eq_outgap], endo[:x_t]] = -1.0 # Note x_t doesn't exist. Create output gap
    Γ0[eq[:eq_outgap], endo[:L_t]] = -1.0*m[:α] + 1.0
    Γ0[eq[:eq_outgap], endo[:u_t]] = m[:α]

    ### 16. Tobin's Q
    Γ0[eq[:eq_tobq], endo[:rm_t]] = m[:ϕi_jpt]*(m[:β]+1.0) # Treating rm_t as notional rate and phii_jpt as ϕi_jpt
    Γ0[eq[:eq_tobq], endo[:qk_t]] = -1.0
    Γ0[eq[:eq_tobq], endo[:μ_t]] = -1.0 # Investment Shock, not sure if μ_t is right
    Γ0[eq[:eq_tobq], endo[:z_t]] = m[:ϕi_jpt]
    Γ0[eq[:eq_tobq], endo[:Ei_t]] = -1.0*m[:β]*m[:ϕi_jpt]
    Γ0[eq[:eq_tobq], endo[:ztil_t]] = -1.0*m[:β]*m[:ϕi_jpt] # This should expected tech shock, not ztil_t

    Γ1[eq[:eq_tobq], endo[:i_t]] = m[:ϕi_jpt]


    ### 17. Hours Worked
    Γ0[eq[:eq_L], endo[:y_t]] = 1.0
    Γ0[eq[:eq_L], endo[:L_t]] = m[:α] - 1.0
    Γ0[eq[:eq_L], endo[:u_t]] = -1.0*m[:α]
    Γ0[eq[:eq_L], endo[:ztil_t]] = m[:α] # Techshock
    Γ0[eq[:eq_L], endo[:unkshk_t]] = m[:α]-1.0 # Unknown shock - this is column 30

    Γ1[eq[:eq_L], endo[:k_t]] = m[:α]


    ### 18. Marginal Cost
    Γ0[eq[:eq_mcost], endo[:w_t]] = 1.0
    Γ0[eq[:eq_mcost], endo[:y_t]] = -1.0
    Γ0[eq[:eq_mcost], endo[:L_t]] = 1.0
    Γ0[eq[:eq_mcost], endo[:mc_t]] = -1.0


    ### 19. Marginal Utility of Consumption
    Γ0[eq[:eq_muc], endo[:c_t]] = -1.0
    Γ0[eq[:eq_muc], endo[:muc_t]] = m[:gamtil] -1.0 # Need muc_t, marginal utility of consumption
    Γ0[eq[:eq_muc], endo[:ztil_t]] = -1.0*m[:gamtil] # Techshock

    Γ1[eq[:eq_muc], endo[:c_t]] = -1.0*m[:gamtil]


    ### 20. Lagged GDP
    Γ0[eq[:eq_laggdp], endo[:y_t1]] = -1.0 # y_t1 is in endogenous_states_augmented not endo, so change needed. Also assuming 23rd col and 31st row is lagged GDP
    Γ1[eq[:eq_laggdp], endo[:y_t]] = -1.0


    ### 21. Lagged Consumption
    Γ0[eq[:eq_lagcc], endo[:c_t1]] = -1.0 # Again, note this is endogenous_states_augmented, 24th and 32nd row is lagged consumption
    Γ1[eq[:eq_lagcc], endo[:c_t]] = -1.0


    ### 22. Lagged Investment
    Γ0[eq[:eq_lagit], endo[:i_t1]] = -1.0 # i_t1 is in endogenous_states_augmented not endo, so change needed. Also assuming 25th col and 33rd row is lagged Investment
    Γ1[eq[:eq_lagit], endo[:i_t]] = -1.0


    ### 23. Lagged Wage (Zeta)
    Γ0[eq[:eq_lagwage], endo[:w_t1]] = -1.0 # w_t1 is in endogenous_states_augmented not endo, so change needed. Also assuming 26rd col and 34th row is lagged wage
    Γ1[eq[:eq_lagwage], endo[:w_t]] = -1.0


    ### 24. Liquidity Shock (η Shock?) - Unneeded, repeated later
    # Γ0[eq[:eq_liqshk], endo[:b_t]] = -1.0 # b_t is b_t shock, coming from etashock
    # Γ1[eq[:eq_liqshk], endo[:b_t]] = -0.85 # Same as above


    ### 25. Value of Consumption (λ)
    Γ0[eq[:eq_λc], endo[:w_t]] = 1.0
    Γ0[eq[:eq_λc], endo[:λc]] = -1.0 # λc is not a state. Needs to be added.
    Γ0[eq[:eq_λc], endo[:b_t]] = 1.0 # b_t is η shock here.
    Γ0[eq[:eq_λc], endo[:Etechshock]] = -1.0 # Etechshock doesn't exist
    Γ0[eq[:eq_λc], endo[:Eπ_t]] = -1.0
    Γ0[eq[:eq_λc], endo[:Eλc]] = 1.0 #Eλc doesn't exist


    ### 26. Wage Inflation
    Γ0[eq[:eq_π_w], endo[:zzw_t]] = 1.0 # Don't know what zzw_t is
    Γ0[eq[:eq_π_w], endo[:π_w]] = -1.0 # Need to add π_w
    Γ0[eq[:eq_π_w], endo[:ztil_t]] = -1.0*m[:bw] + 1.0 # Techshock
    Γ1[eq[:eq_π_w], endo[:π_t]] = m[:aw] - 1.0


    ### 27. V_i
    Γ0[eq[:eq_vi], endo[:i_t]] = 1.0
    Γ0[eq[:eq_vi], endo[:Vi_t]] = -1.0 # Vi_t doesn't exist yet
    Γ0[eq[:eq_vi], endo[:ztil_t]] = 1.0
    Γ1[eq[:eq_vi], endo[:i_t]] = 1.0


    ### 28. V_p
    Γ0[eq[:eq_vp], endo[:π_t]] = 1.0
    Γ0[eq[:eq_vp], endo[:Vp_t]] = -1.0 # Vp_t doesn't exist yet
    Γ1[eq[:eq_vp], endo[:π_t]] = 1.0 - m[:ap]


    ### 29. V_w
    Γ0[eq[:eq_vw], endo[:w_t]] = -1.0 * m[:κw] # Check κw
    Γ0[eq[:eq_vw], endo[:λc]] = -1.0 * m[:κw] # Check κw and λc doesn't exist
    Γ0[eq[:eq_vw], endo[:L_t]] = m[:κw] * m[:σ_l] # Check σ_l and κw
    Γ0[eq[:eq_vw], endo[:Vw_t]] = -1.0 # Vw_t doesn't exist yet
    Γ0[eq[:eq_vw], endo[:EVw_t]] = m[:β] # EVw_t doesn't exist yet


    ### 30. bc
    Γ0[eq[:eq_bc], endo[:c_t]] = m[:gamtil] * 1.0/(1.0 - m[:gamtil])
    Γ0[eq[:eq_bc], endo[:bc_t]] = -1.0 # Need to create bc_t
    Γ0[eq[:eq_bc], endo[:Ec_t]] = -1.0*1.0/(-1.0*m[:gamtil] + 1.0)
    Γ0[eq[:eq_bc], endo[:Etechshock]] = -1.0*m[:gamtil]*1.0/(-1.0*m[:gamtil] + 1.0) # Need to create Etechshock


    ### 31. bi
    Γ0[eq[:eq_bi], endo[:i_t]] = -1.0
    Γ0[eq[:eq_bi], endo[:bi_t]] = -1.0 # Need to create bi_t
    Γ0[eq[:eq_bi], endo[:Ei_t]] = 1.0
    Γ0[eq[:eq_bi], endo[:Etechshock]] = 1.0 # Need to create Etechshock


    ### EXOGENOUS SHOCKS ###

    # Neutral technology CHECK THIS
    Γ0[eq[:eq_ztil], endo[:ztil_t]] = -1.0
    Ψ[eq[:eq_ztil], exo[:ztil_sh]]     = -1.0

    # Government spending
    Γ0[eq[:eq_g], endo[:g_t]] = -1.0
    Γ1[eq[:eq_g], endo[:g_t]] = -1.0*m[:ρ_g]
    Ψ[eq[:eq_g], exo[:g_sh]]  = -1.0

    # Asset shock (treat as liquidity/η shock)
    Γ0[eq[:eq_b], endo[:b_t]] = -1.0
    Γ1[eq[:eq_b], endo[:b_t]] = -0.85
    Ψ[eq[:eq_b], exo[:b_sh]]  = -1.0

    # Investment-specific technology
    Γ0[eq[:eq_μ], endo[:μ_t]] = -1.0
    Γ1[eq[:eq_μ], endo[:μ_t]] = -1.0*m[:ρ_inv] # Check how ρ_inv is done
    Ψ[eq[:eq_μ], exo[:μ_sh]]  = -1.0

    # Monetary policy shock
    ### Putting interest rate shock in here:
    Γ0[eq[:eq_mon], endo[:mon_t]] = -1.0
    Γ1[eq[:eq_mon], endo[:mon_t]] = -1.0*m[:ρ_rm]
    Ψ[eq[:eq_mon], exo[:rm_sh]]  = -1.0

    # ElastShock (Unknown shock 1) Maybe it's ashock?
    Γ0[eq[:eq_elast], endo[:elast_t]] = -1.0
    Γ1[eq[:eq_elast], endo[:elast_t]] = -1.0*m[:ρ_elast] # Check ρ_elast
    Ψ[eq[:eq_elast], exo[:elast_sh]]  = -1.

    # ElastwShock (Unknown shock 2)
    Γ0[eq[:eq_elastw], endo[:elastw_t]] = -1.0
    Γ1[eq[:eq_elastw], endo[:elastw_t]] = -1.0*m[:ρ_elastw] # Check ρ
    Ψ[eq[:eq_elastw], exo[:elastw_sh]]  = -1.0


    # UnkShock (Unknown shock 3)
    Γ0[eq[:eq_unk], endo[:unk_t]] = -1.0
    Ψ[eq[:eq_unk], exo[:unk_sh]]  = -1.0


    # Anticipated policy shocks
    if n_anticipated_shocks(m) > 0

        # This section adds the anticipated shocks. There is one state for all the
        # anticipated shocks that will hit in a given period (i.e. rm_tl2 holds those that
        # will hit in two periods), and the equations are set up so that rm_tl2 last period
        # will feed into rm_tl1 this period (and so on for other numbers), and last period's
        # rm_tl1 will feed into the rm_t process (and affect the Taylor Rule this period).

        Γ1[eq[:eq_rm], endo[:rm_tl1]]   = 1.
        Γ0[eq[:eq_rml1], endo[:rm_tl1]] = 1.
        Ψ[eq[:eq_rml1], exo[:rm_shl1]]  = 1.

        if n_anticipated_shocks(m) > 1
            for i = 2:n_anticipated_shocks(m)
                Γ1[eq[Symbol("eq_rml$(i-1)")], endo[Symbol("rm_tl$i")]] = 1.
                Γ0[eq[Symbol("eq_rml$i")], endo[Symbol("rm_tl$i")]] = 1.
                Ψ[eq[Symbol("eq_rml$i")], exo[Symbol("rm_shl$i")]] = 1.
            end
        end
    end



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

    # Sticky prices and wages
    Γ0[eq[:eq_EL], endo[:L_t]]         = 1.
    Γ1[eq[:eq_EL], endo[:EL_t]]         = 1.
    Π[eq[:eq_EL], ex[:EL_sh]]          = 1.

    ### E(rk)

    # Sticky prices and wages
    Γ0[eq[:eq_Erk], endo[:rk_t]]       = -1.0
    Γ1[eq[:eq_Erk], endo[:Erk_t]]       = -1.0
    Π[eq[:eq_Erk], ex[:Erk_sh]]        = -1.0

    ### E(w)

    ### E(techshock) (Added)
    Γ0[eq[:eq_Ez], endo[:ztil_t]]         = -1. # Techshock
    Γ1[eq[:eq_Ez], endo[:Etechshock]]         = -1. # Need Ez_t, expected Techshock
    Π[eq[:eq_Ez], ex[:Etechshock_sh]]          = -1. # Doesn't exist - change


    ### E(λ_c) (Added)
    Γ0[eq[:eq_Eλc], endo[:λc]]         = -1. # λc doesn't exist yet
    Γ1[eq[:eq_Eλc], endo[:Eλc]]         = -1. # Need Eλc, expected λ_c
    Π[eq[:eq_Eλc], ex[:Eλc_sh]]          = -1. # Doesn't exist - change


    ### E(VW) (Added)
    Γ0[eq[:eq_EVw], endo[:Vw_t]]         = -1. # Vw_t doesn't exist yet
    Γ1[eq[:eq_EVw], endo[:EVw_t]]         = -1. # Need EVw_t
    Π[eq[:eq_EVw], ex[:EVw_sh]]          = -1. # Doesn't exist - change


    return Γ0, Γ1, C, Ψ, Π
end
