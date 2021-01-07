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
function eqcond(m::AnSchorfheide)
    eqcond(m, 1)
end

function eqcond(m::AnSchorfheide, reg::Int)
    endo = m.endogenous_states
    exo  = m.exogenous_shocks
    ex   = m.expected_shocks
    eq   = m.equilibrium_conditions

    Γ0 = zeros(n_states(m), n_states(m))
    Γ1 = zeros(n_states(m), n_states(m))
    C  = zeros(n_states(m))
    Ψ  = zeros(n_states(m), n_shocks_exogenous(m))
    Π  = zeros(n_states(m), n_shocks_expectational(m))

    for para in m.parameters
        if !isempty(para.regimes)
            if (haskey(get_settings(m), :model2para_regime) ? haskey(get_setting(m, :model2para_regime), para.key) : false)
                ModelConstructors.toggle_regime!(para, reg, get_setting(m, :model2para_regime)[para.key])
            else
                ModelConstructors.toggle_regime!(para, reg)
            end
        end
    end

    ### ENDOGENOUS STATES ###

    ### 1. Consumption Euler Equation

    Γ0[eq[:eq_euler], endo[:y_t]]  = 1.
    Γ0[eq[:eq_euler], endo[:R_t]] = 1/m[:τ]
    Γ0[eq[:eq_euler], endo[:g_t]] = -(1-m[:ρ_g])
    Γ0[eq[:eq_euler], endo[:z_t]] = -m[:ρ_z]/m[:τ]
    Γ0[eq[:eq_euler], endo[:Ey_t]] = -1
    Γ0[eq[:eq_euler], endo[:Eπ_t]] = -1/m[:τ]

    ### 2. NK Phillips Curve

    Γ0[eq[:eq_phillips], endo[:y_t]] = -m[:κ]
    Γ0[eq[:eq_phillips], endo[:π_t]] = 1
    Γ0[eq[:eq_phillips], endo[:g_t]] = m[:κ]
    Γ0[eq[:eq_phillips], endo[:Eπ_t]] = -1/(1+m[:rA]/400)

    ### 3. Monetary Policy Rule

    Γ0[eq[:eq_mp], endo[:y_t]] = -(1-m[:ρ_R])*m[:ψ_2]
    Γ0[eq[:eq_mp], endo[:π_t]] = -(1-m[:ρ_R])*m[:ψ_1]
    Γ0[eq[:eq_mp], endo[:R_t]] = 1
    Γ0[eq[:eq_mp], endo[:g_t]] = (1-m[:ρ_R])*m[:ψ_2]
    Γ1[eq[:eq_mp], endo[:R_t]] = m[:ρ_R]
    Ψ[eq[:eq_mp], exo[:rm_sh]] = 1

    ### 4. Output lag

    Γ0[eq[:eq_y_t1], endo[:y_t1]] = 1
    Γ1[eq[:eq_y_t1], endo[:y_t]] = 1

    ### 5. Government spending

    Γ0[eq[:eq_g], endo[:g_t]] = 1
    Γ1[eq[:eq_g], endo[:g_t]] = m[:ρ_g]
    Ψ[eq[:eq_g], exo[:g_sh]] = 1

    ### 6. Technology

    Γ0[eq[:eq_z], endo[:z_t]] = 1
    Γ1[eq[:eq_z], endo[:z_t]] = m[:ρ_z]
    Ψ[eq[:eq_z], exo[:z_sh]] = 1

    ### 7. Expected output

    Γ0[eq[:eq_Ey], endo[:y_t]] = 1
    Γ1[eq[:eq_Ey], endo[:Ey_t]] = 1
    Π[eq[:eq_Ey], ex[:Ey_sh]] = 1

    ### 8. Expected inflation

    Γ0[eq[:eq_Eπ], endo[:π_t]] = 1
    Γ1[eq[:eq_Eπ], endo[:Eπ_t]] = 1
    Π[eq[:eq_Eπ], ex[:Eπ_sh]] = 1

#=
   # We additionally need to directly add the equation(s) for pgap, ygap, etc. here rather than just
   # in the altpolicy files b/c
   # (1) Regime-switching won't work otherwise
   # (2) we may want to define the their values at the beginning of the forecast period
   #     rather than just when we start using the alt rule.
   if haskey(m.settings, :add_altpolicy_pgap) ? get_setting(m, :add_altpolicy_pgap) : false
       Γ0[eq[:eq_pgap], endo[:pgap_t]]  =  1.
       if haskey(m.settings, :replace_eqcond_func_dict)
           if reg >= minimum(keys(get_setting(m, :replace_eqcond_func_dict))) &&
               reg <= maximum(keys(get_setting(m, :replace_eqcond_func_dict))) &&
               haskey(m.settings, :pgap_type)
               if get_setting(m, :pgap_type) == :ngdp
                   Γ0[eq[:eq_pgap], endo[:pgap_t]] =  1.
                   Γ0[eq[:eq_pgap], endo[:π_t]]    = -1.
                   Γ1[eq[:eq_pgap], endo[:pgap_t]] =  1.

                   Γ0[eq[:eq_pgap], endo[:y_t]]    = -1.
                   Γ0[eq[:eq_pgap], endo[:z_t]]    = -1.
                   Γ1[eq[:eq_pgap], endo[:y_t]]    = -1.
               elseif get_setting(m, :pgap_type) == :ait
                   Thalf = haskey(get_settings(m), :ait_Thalf) ? get_setting(m, :ait_Thalf) : 10.
                   ρ_ait = exp(log(0.5) / Thalf)
                   Γ0[eq[:eq_pgap], endo[:pgap_t]] = 1.
                   Γ0[eq[:eq_pgap], endo[:π_t]]    = -1.
                   Γ1[eq[:eq_pgap], endo[:pgap_t]] = ρ_ait
               elseif get_setting(m, :pgap_type) == :smooth_ait
                   Thalf = haskey(get_settings(m), :ait_Thalf) ? get_setting(m, :ait_Thalf) : 10.
                   ρ_pgap = exp(log(0.5) / Thalf)
                   Γ0[eq[:eq_pgap], endo[:pgap_t]] = 1.
                   Γ0[eq[:eq_pgap], endo[:π_t]]    = -1.
                   Γ1[eq[:eq_pgap], endo[:pgap_t]] = ρ_pgap
               elseif get_setting(m, :pgap_type) in [:smooth_ait_gdp, :smooth_ait_gdp_alt, :flexible_ait, :rw]
                   Thalf = haskey(get_settings(m), :ait_Thalf) ? get_setting(m, :ait_Thalf) : 10.
                   ρ_pgap = exp(log(0.5) / Thalf)
                   Γ0[eq[:eq_pgap], endo[:pgap_t]] = 1.
                   Γ0[eq[:eq_pgap], endo[:π_t]]    = -1.
                   Γ1[eq[:eq_pgap], endo[:pgap_t]] = ρ_pgap
               end
               if (haskey(get_settings(m), :add_initialize_pgap_ygap_pseudoobs) ?
                   get_setting(m, :add_initialize_pgap_ygap_pseudoobs) : false)
                   Ψ[eq[:eq_pgap], exo[:pgap_sh]] = 1.
               end
           end
       end
   end

   if haskey(m.settings, :add_altpolicy_ygap) ? get_setting(m, :add_altpolicy_ygap) : false
       Γ0[eq[:eq_ygap], endo[:ygap_t]]  =  1.
       if haskey(m.settings, :replace_eqcond_func_dict)
           if reg >= minimum(keys(get_setting(m, :replace_eqcond_func_dict))) &&
               reg <= maximum(keys(get_setting(m, :replace_eqcond_func_dict))) &&
               haskey(m.settings, :ygap_type)
               if get_setting(m, :ygap_type) in [:smooth_ait_gdp, :smooth_ait_gdp_alt, :flexible_ait, :rw]
                   Thalf  = haskey(get_settings(m), :gdp_Thalf) ? get_setting(m, :gdp_Thalf) : 10.
                   ρ_ygap = exp(log(0.5) / Thalf)

                   Γ0[eq[:eq_ygap], endo[:ygap_t]] = 1.
                   # Γ0[eq[:eq_ygap], endo[:π_t]]    = -1.
                   Γ1[eq[:eq_ygap], endo[:ygap_t]] = ρ_ygap

                   Γ0[eq[:eq_ygap], endo[:y_t]]    = -1.
                   Γ0[eq[:eq_ygap], endo[:z_t]]    = -1.
                   Γ1[eq[:eq_ygap], endo[:y_t]]    = -1.
               end
               if (haskey(get_settings(m), :add_initialize_pgap_ygap_pseudoobs) ?
                   get_setting(m, :add_initialize_pgap_ygap_pseudoobs) : false)
                   Ψ[eq[:eq_ygap], exo[:ygap_sh]] = 1.
               end
           end
       end
   end

   if haskey(m.settings, :add_rw) ? get_setting(m, :add_rw) : false
       Γ0[eq[:eq_rw], endo[:rw_t]]     =  1.
       Γ0[eq[:eq_Rref], endo[:Rref_t]] =  1.

       if haskey(m.settings, :replace_eqcond_func_dict)
           if reg >= minimum(keys(get_setting(m, :replace_eqcond_func_dict))) &&
               reg <= maximum(keys(get_setting(m, :replace_eqcond_func_dict))) &&
               haskey(m.settings, :Rref_type)

               ρ_rw = haskey(get_settings(m), :ρ_rw) ? get_setting(m, :ρ_rw) : 0.93
               Γ0[eq[:eq_rw], endo[:rw_t]]   = 1.
               Γ1[eq[:eq_rw], endo[:rw_t]]   = ρ_rw

               if get_setting(m, :Rref_type) in [:ait]
                   Thalf  = haskey(get_settings(m), :ait_Thalf) ? get_setting(m, :ait_Thalf) : 10.
                   ρ_pgap = exp(log(0.5) / Thalf)
                   φ      = haskey(get_settings(m), :ait_φ) ? get_setting(m, :ait_φ) : 0.25

                   Γ0[eq[:eq_Rref], endo[:Rref_t]] = 1.
                   Γ0[eq[:eq_Rref], endo[:pgap_t]] = -φ * (1/(1 - ρ_pgap))
                   Γ1[eq[:eq_Rref], endo[:Rref_t]] = 0.
                   C[eq[:eq_Rref]]                 = 0.
                   Γ0[eq[:eq_Rref], endo[:rw_t]]   = -1.

               elseif get_setting(m, :Rref_type) in [:smooth_ait_gdp, :smooth_ait_gdp_alt, :flexible_ait, :rw]

                   ait_Thalf = haskey(get_settings(m), :ait_Thalf) ? get_setting(m, :ait_Thalf) : 10.
                   gdp_Thalf = haskey(get_settings(m), :gdp_Thalf) ? get_setting(m, :gdp_Thalf) : 10.
                   ρ_pgap    = exp(log(0.5) / ait_Thalf)
                   ρ_ygap    = exp(log(0.5) / gdp_Thalf)
                   ρ_smooth  = haskey(get_settings(m), :rw_ρ_smooth) ? get_setting(m, :rw_ρ_smooth) : 0.656
                   φ_π       = haskey(get_settings(m), :rw_φ_π) ? get_setting(m, :rw_φ_π) : 11.13
                   φ_y       = haskey(get_settings(m), :rw_φ_y) ? get_setting(m, :rw_φ_y) : 11.13

                   Γ0[eq[:eq_Rref], endo[:Rref_t]] = 1.
                   Γ1[eq[:eq_Rref], endo[:Rref_t]] = ρ_smooth
                   C[eq[:eq_Rref]]                 = 0.

                   Γ0[eq[:eq_Rref], endo[:pgap_t]]   = -φ_π * (1. - ρ_pgap) * (1. - ρ_smooth) # This is the AIT part
                   Γ0[eq[:eq_Rref], endo[:ygap_t]]   = -φ_y * (1. - ρ_ygap) * (1. - ρ_smooth) # This is the GDP part
               end
           end
       end
   end

   if haskey(m.settings, :add_pgap)
       if get_setting(m, :add_pgap)
           if get_setting(m, :pgap_type) in [:smooth_ait_gdp, :smooth_ait, :ait, :smooth_ait_gdp_alt, :flexible_ait]
               Thalf = haskey(m.settings, :ait_Thalf) ? get_setting(m, :ait_Thalf) : 10
               ρ_pgap = exp(log(0.5) / Thalf)
               Γ0[eq[:eq_pgap], endo[:pgap_t]] = 1.
               Γ0[eq[:eq_pgap], endo[:π_t]]    = -1.
               Γ1[eq[:eq_pgap], endo[:pgap_t]] = ρ_pgap
               if (haskey(get_settings(m), :add_initialize_pgap_ygap_pseudoobs) ?
                   get_setting(m, :add_initialize_pgap_ygap_pseudoobs) : false)
                   Ψ[eq[:eq_pgap], exo[:pgap_sh]] = 1.
               end
           end
       end
   end

   if haskey(m.settings, :add_ygap)
       if get_setting(m, :add_ygap)
           if get_setting(m, :ygap_type) in [:smooth_ait_gdp, :smooth_ait_gdp_alt, :flexible_ait]
               Thalf  = haskey(get_settings(m), :gdp_Thalf) ? get_setting(m, :gdp_Thalf) : 10.
               ρ_ygap = exp(log(0.5) / Thalf)

               Γ0[eq[:eq_ygap], endo[:ygap_t]] = 1.
               # Γ0[eq[:eq_ygap], endo[:π_t]]    = -1.
               Γ1[eq[:eq_ygap], endo[:ygap_t]] = ρ_ygap

               Γ0[eq[:eq_ygap], endo[:y_t]]    = -1.
               Γ0[eq[:eq_ygap], endo[:z_t]]    = -1.
               Γ1[eq[:eq_ygap], endo[:y_t]]    = -1.
               if (haskey(get_settings(m), :add_initialize_pgap_ygap_pseudoobs) ?
                   get_setting(m, :add_initialize_pgap_ygap_pseudoobs) : false)
                   Ψ[eq[:eq_ygap], exo[:ygap_sh]] = 1.
               end
           end
       end
   end

   # Switch to Flexible AIT in 2020-Q3 and beyond
   if (haskey(m.settings, :flexible_ait_policy_change) ? get_setting(m, :flexible_ait_policy_change) : false)
       if get_setting(m, :regime_dates)[reg] >= get_setting(m, :flexible_ait_policy_change_date)
           Γ0, Γ1, C, Ψ, Π = flexible_ait_replace_eq_entries(m, Γ0, Γ1, C, Ψ, Π)
       end
   end

   # If running temporary alterantive policies, we update the gensys matrices here.
   # This step MUST be the last block of code prior to switching parameter values back to the first regime.
   if haskey(m.settings, :replace_eqcond) ? get_setting(m, :replace_eqcond) : false
       if haskey(m.settings, :replace_eqcond_func_dict)
           if haskey(get_setting(m, :replace_eqcond_func_dict), reg) && reg != get_setting(m, :n_regimes)
               Γ0, Γ1, C, Ψ, Π = get_setting(m, :replace_eqcond_func_dict)[reg](m, Γ0, Γ1, C, Ψ, Π)
           end
       end
   end
=#
   for para in m.parameters
        if !isempty(para.regimes)
            ModelConstructors.toggle_regime!(para, 1)
        end
    end

    return Γ0, Γ1, C, Ψ, Π
end
