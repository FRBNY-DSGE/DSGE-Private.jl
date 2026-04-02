using DSGE: @sslogdeviations2levels, @sslogdeviations2levels_unprimekeys, @unpack_and_first
#include("helpers/q_cons2.jl")
#include("mBBQ.jl")
#include("models/7_KShape_Share/index2.jl")    
#include("models/7_KShape_Share/macros2.jl")   

"""
```
Fsys_agg(F, X, XPrime, θ, grids, id, nt, eq)
```
Return deviations from aggregate equilibrium conditions for simplified 6-variable HANK model.

### Inputs
- `F::AbstractVector`: vector holding deviations for aggregate equilibrium conditions
- `X::AbstractArray`,`XPrime::AbstractArray`: deviations from steady state in periods t [`X`] and t+1 [`XPrime`]
- `θ::NamedTuple`: maps parameter name to value
- `grids::OrderedDict`: maps names of quantities related to the idiosyncratic state space to their values
- `id::OrderedDict{Symbol, Int or UnitRange{Int}}`: maps variable name (e.g. `MRS_t` and `MRS′_t`)
    to its index/indices in `X` and `XPrime`.
- `nt::NamedTuple`: maps variable name (e.g. `MRS_t` and `MRS′_t`) to steady-state log level
- `eq::OrderedDict{Symbol, Int or UnitRange{Int}}`: maps name of equilibrium conditions
    to its index/indices in `F`

"""#=
function Fsys_agg(F::AbstractVector, X::AbstractArray, XPrime::AbstractArray,
                  ss::NamedTuple, grids::OrderedDict, id::OrderedDict{Symbol, I1}, nt::NamedTuple,
                  eq::OrderedDict{Symbol, I2}) where {I1 <: Union{Int, UnitRange}, I2 <: Union{Int, UnitRange}, reg = 1}

 =#


 function Fsys_agg(F, m, grid, StateSS, ControlSS, Xt1, Yt1, Xt, Yt, ss, state_id, control_id, reg=1)
#=
                    #TODO: make all of these arguments into function
                  jld2file = "XssYss.jld2"
                  @load jld2file StateSS ControlSS SS_stats grid param
                  
                  m = mBBQ()
                  
                  #one time setup to move prev donggyu format to bbl format
                  grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "nCOP", "oc", "numstates"])
                  state_id, control_id = build_indices(grid, length(StateSS), length(ControlSS))
                  ss = build_ss(param, SS_stats)
                  
                  #test state and control controls of zero
                  Xt1 = zeros(length(StateSS))
                  Yt1 = zeros(length(ControlSS))
                  
  =#



                  #F = Dict{Symbol,Float64}() #TODO: need to change typing of this

     #Yt1 = Xprime #TODO: def change
     #Xt1 = X
    #===================================================================#
    #eq1: eq_rate_monetary_policy TODO: regime change
    @sslogdeviations2levels R_cb_t, eps_R_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys R_cb′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels pi_t, unemp_t = Yt, control_id, ControlSS

    monpol = get_setting(m, :regime_monpol)[reg]

    if monpol == :taylor
        F[:eq_rate_monetary_policy] = log(R_cb′_t) - ss[:R_cb_t] - 
        (1 - m[:ρ_R]) * (m[:ϕ_π] * log(pi_t/m[:π_cb]) -
        m[:ϕ_u] * (unemp_t - ss[:u_t])) -  
        m[:ρ_R]*(log(R_cb_t) - ss[:R_cb_t]) + log(eps_R_t)

        println("in taylor")
    elseif monpol == :qe
        F[:eq_rate_monetary_policy] = log(R_cb′_t) - ss[:R_cb_t] 

        println("in qe")

    end


    #===================================================================#
    #eq2: wage
    @sslogdeviations2levels w_t, eps_w_t, π_past_t,ψ_w_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys w′_t, ψ_w′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels h_t = Yt, control_id, ControlSS
    #used to be ψ_w_t unprimed 
    #delta might be new param d
    F[:eq_wage] = log(w′_t) - ss[:w_t] - 
    m[:ρ_w].value * (log(w_t) - ss[:w_t] + 
    m[:d].value * log(log(ss[:π_t])/log(pi_t)) + (1-m[:d].value)*log(log(π_past_t)/log(pi_t))) - 
    (1 - m[:ρ_w].value) * eps_w_t * (ψ_w_t + h_t - ss[:r_l_t])
    (1 - m[:ρ_w].value) * eps_w_t * log(ψ_w′_t + h_t - ss[:r_l_t])


    #===================================================================#
    #eq3: asset a (illiquid)
    @sslogdeviations2levels lev_t, NW_b_t, Q_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys A_b′_t, lev′_t, NW_b′_t, Q′_t = Xt1, state_id, StateSS
   
    F[:eq_illiquid_assets] = log(A_b′_t) - log( lev′_t*NW_b′_t/Q′_t ) 


    #===================================================================#
    #eq4: asset b (bonds)
    @sslogdeviations2levels A_gaux_t, B_b_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys B_b′_t, Q′_t, A_gaux′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels T_t, B_gov_ncp_t, r_a_t, UB_t, LT_t, C_b_t, G_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys Profit_t, Profit_FI_t, B_gov_ncp′_t = Yt1, control_id, ControlSS
    A_g′_t = A_gaux′_t - 1
    A_g_t = A_gaux_t - 1

    #eq4: asset b (bonds)
    F[:eq_liquid_assets] =  log(B_b′_t) -
    log(T_t + B_gov_ncp′_t - B_gov_ncp_t * R_cb_t/pi_t + 
    (Q′_t + r_a_t - R_cb_t/pi_t*Q_t) * A_g_t - UB_t - LT_t - 
    m[:τ_cp].value * Q′_t * A_g′_t - G_t + 
    m[:b_a_aux2].value * R_cb_t / pi_t * B_b_t - 
    C_b_t ) 


    #===================================================================#
    #eq5: central bank assets TODO: regime change
    @sslogdeviations2levels x_cb_t = Xt, state_id, StateSS

    if monpol == :qe
        println("hello monpol")
        F[:eq_central_bank_assets] = log(A_gaux′_t) - log( x_cb_t * A_g_t + 1 )
    else
        F[:eq_central_bank_assets] = log(A_gaux′_t) - 
        log( m[:ρ_passive_QE]* A_g_t + 
        (1 - m[:ρ_passive_QE])* ss[:A_g_t] + 1 )
    end

    #===================================================================#
    #eq6: tobin's q
    # @sslogdeviations2levels  = Xt1, state_id, StateSS
    # @sslogdeviations2levels_unprimekeys  = Xt1, state_id, StateSS
    @sslogdeviations2levels ι_2_t, x_k_t, λ_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys ι_2′_t, x_k′_t = Yt1, control_id, ControlSS
    F[:eq_tobins_q] = log(Q′_t) -
    log(ι_2_t * (1 + m[:ϕ].value * log(x_k′_t)+m[:ϕ].value/2*(log(x_k′_t))^2 ) + 
    ι_2′_t * λ_t / ss[:λ_t] * m[:ϕ].value * log(x_k′_t) * x_k′_t )

    #===================================================================#
    #eq7: bank leverage
    @sslogdeviations2levels_unprimekeys lev′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels ee_t, vv_t = Yt, control_id, ControlSS

    F[:eq_leverage] = log(lev′_t) - log( ee_t / (m[:δ].value - vv_t) )

    #===================================================================#
    #eq8: NW_b_ind
    @sslogdeviations2levels lev_t, NW_b_t, A_b_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys NW_b′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels RRa_t = Yt, control_id, ControlSS

    F[:eq_net_worth_bank] = log(NW_b′_t) - 
    log( (m[:θ_b].value *((RRa_t - m[:b_a_aux2].value * R_cb_t / pi_t)*
    lev_t + 
    m[:b_a_aux2].value* R_cb_t / pi_t) * NW_b_t + 
    m[:ω].value * Q_t * A_b_t) ) 

    #===================================================================#
    #eq9: r tilde
    @sslogdeviations2levels D_t, R_tilde_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys R_tilde′_t, D′_t = Xt1, state_id, StateSS

    F[:eq_houshold_bond_rate] = log(R_tilde′_t) - log( R_cb′_t * D_t ) 
    #===================================================================#
    #eq10: eq_quantitative_easing #TODO: add regimes #TODO ADD shocks
    @sslogdeviations2levels_unprimekeys x_cb′_t = Xt1, state_id, StateSS

    if monpol == :taylor
        F[:eq_quantitative_easing] = log(x_cb′_t)
    elseif monpol == :qe
            F[:eq_quantitative_easing] = log(x_cb′_t) - 
        log( 1 + m[:ρ_X_QE] * (x_cb_t-1) + 
        (1-m[:ρ_X_QE])*(-m[:ϕ_π_QE]*log(pi_t / m[:π_cb])+
        m[:ϕ_u_QE] * (unemp_t - ss[:u_t])) )
    end


    #===================================================================#
    #eq11: inflation lag
    @sslogdeviations2levels_unprimekeys π_past′_t = Xt1, state_id, StateSS

    F[:eq_inflation_lag] = log(π_past′_t) - log(pi_t)

    #===================================================================#
    #eq12: ouput lag
    @sslogdeviations2levels_unprimekeys Y_past′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels Y_t = Yt, control_id, ControlSS

    F[:eq_output_lag] = log(Y_past′_t) - log(Y_t)

    #===================================================================#
    #eq13: consumption lag
    @sslogdeviations2levels_unprimekeys C_past′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels C_t = Yt, control_id, ControlSS

    F[:eq_consumption_lag] = log(C_past′_t) - log(C_t)

    #===================================================================#
    #eq14: investment lag
    @sslogdeviations2levels_unprimekeys I_past′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels I_t = Yt, control_id, ControlSS

    F[:eq_investment_lag] = log(I_past′_t) - log(I_t)

    #===================================================================#
    #eq15: profit lag
    @sslogdeviations2levels_unprimekeys Profit_past′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels Profit_t = Yt, control_id, ControlSS

    F[:eq_profit_lag] = log(Profit_past′_t) - log(Profit_t + m[:fix2].value)

    #===================================================================#
    #eq16: unemployment lag
    @sslogdeviations2levels_unprimekeys unemp_past′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels unemp_t = Yt, control_id, ControlSS

    F[:eq_unemployment_lag] = log(unemp_past′_t) - log(unemp_t)

    #===================================================================#
    #eq17: government spending lag 
    @sslogdeviations2levels_unprimekeys G_past′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels G_t = Yt, control_id, ControlSS

    F[:eq_government_spending_lag] = log(G_past′_t) - log(G_t)

    #===================================================================#
    #eq18: lump sum transfers
    @sslogdeviations2levels_unprimekeys LT_past′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels LT_t = Yt, control_id, ControlSS

    F[:eq_lump_sum_transfers] = log(LT_past′_t) - log(LT_t)

    #===================================================================#
    #eq19: R_star
    @sslogdeviations2levels R_star_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys  R_star′_t = Xt1, state_id, StateSS
    F[:eq_R_star] = log(R_star′_t) - ss[:R_cb_t] -
                    (1 - m[:ρ_R]) * (m[:ϕ_π] * log(pi_t/m[:π_cb]) -
                    m[:ϕ_u] * (unemp_t - ss[:u_t])) -
                    m[:ρ_R]*(log(R_star_t) - ss[:R_cb_t]) - log(eps_R_t)
  
    #===================================================================#
    #eq19: eq_fiscal_liability

    @sslogdeviations2levels B_F_t, eps_B_F_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys B_F′_t = Xt1, state_id, StateSS
   
    F[:eq_fiscal_liability] = log(B_F′_t) - ( m[:ρ_B_F].value * log(B_F_t) +
    (1 - m[:ρ_B_F].value) * log(ss[:Y_t] / (ss[:Y_t] - ss[:B_F_t] )) + log(eps_B_F_t))

    #===================================================================#
    #eq20: z 

    @sslogdeviations2levels Z_t, eps_Z_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys Z′_t = Xt1, state_id, StateSS

    F[:eq_z] = log(Z′_t) - (m[:ρ_Z].value * log(Z_t) + log(eps_Z_t))

    #===================================================================#
    #eq21:  

    @sslogdeviations2levels ψ_rp_t, eps_RP_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys ψ_rp′_t = Xt1, state_id, StateSS

    F[:eq_ψ] = log(ψ_rp′_t) - (m[:ρ_ψ_rp].value * log(ψ_rp_t))

    #===================================================================#
    #eq22: η
    @sslogdeviations2levels η_t, eps_eta_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys η′_t, p_m′_t = Xt1, state_id, StateSS

    #RHS(eta_ind)         = log(p_mark/(p_mark-1));
    F[:eq_η] = log(η′_t) - ( log(p_m′_t) - log(p_m′_t - 1) ) + log(eps_eta_t)

    #===================================================================#
    #eq23: D

    @sslogdeviations2levels D_t, eps_D_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys D′_t = Xt1, state_id, StateSS


    F[:eq_D] = log(D′_t) - (m[:ρ_D].value * log(D_t) + log(eps_D_t))
    #===================================================================#
    #eq24: GG TODO: add regime
    #SHOCK: eps_G
    @sslogdeviations2levels GG_t, eps_G_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys GG′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels G_t, C_b_t = Yt, control_id, ControlSS

    regime = "G"
    if regime == "G"
        F[:eq_GG] = log(GG′_t) - 
        (m[:ρ_G].value * log(GG_t) + (1 - m[:ρ_G].value) * 
         log(ss[:Y_t] / (ss[:Y_t] - ss[:LT_t] - ss[:C_b_t]))) + log(eps_G_t)
    elseif regime == "LT" #doesn't work
        F[:eq_GG] = log(GG′_t)  - 
        (m[:ρ_G]  * log(GG_t) + (1-m[:ρ_G])  * 
         log(ss[:Y_t] / (ss[:Y_t] - ss[:G_t]))) + log(eps_G_t)
    end

    #===================================================================#
    #eq25: 
    #SHOCK: eps_iota
    @sslogdeviations2levels ι_t, eps_iota_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys ι′_t = Xt1, state_id, StateSS

    F[:eq_ι] = log(ι′_t) - (m[:ρ_ι].value * log(ι_t)) + log(eps_iota_t)

    #===================================================================#
    #eq27: 
    #SHOCK: eps_BB
    @sslogdeviations2levels BB_t, eps_BB_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys BB′_t = Xt1, state_id, StateSS


    F[:eq_BB] = log(BB′_t) - (m[:ρ_BB].value * log(BB_t)) + log(eps_BB_t)

    #===================================================================#
    #eq28: psi_w
    #SHOCK: eps_w
    @sslogdeviations2levels ψ_w_t, eps_w_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys ψ_w′_t = Xt1, state_id, StateSS

    F[:eq_ψ_w] = log(ψ_w′_t) - (m[:ρ_ψ_w].value * log(ψ_w_t)) + log(eps_w_t)

    #===================================================================#
    #eq29: mp
    #SHOCK: eps_R
    @sslogdeviations2levels MP_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys MP′_t = Xt1, state_id, StateSS

    F[:eq_MP] = log(MP′_t) - (m[:ρ_mp].value * log(MP_t)) + log(eps_R_t)


    #===================================================================#
    #eq30: p_m

    @sslogdeviations2levels p_m_t = Xt, state_id, StateSS

    # param.rho_eta*log(p_markminus) + 
    #(1-param.rho_eta)*log(param.eta/(param.eta-1)) + eps_eta;
    F[:eq_pm] = log(p_m′_t) - (m[:ρ_η].value * log(p_m_t) + 
    (1-m[:ρ_η].value) * log(m[:η].value / (m[:η].value - 1)))


    #===================================================================#
    #control eq1: capital
    @sslogdeviations2levels K_t, A_hh_t = Yt, control_id, ControlSS
    @sslogdeviations2levels A_b_t = Xt, state_id, StateSS

    F[:eq_control_capital] = log(K_t) - log(A_hh_t + A_g_t  + A_b_t + ss[:A_F_t])


    #===================================================================#
    #control eq2: bonds
    @sslogdeviations2levels B_b_t = Xt, state_id, StateSS
    @sslogdeviations2levels B_t, B_hh_t = Yt, control_id, ControlSS

    F[:eq_control_bond] = B_t - (B_hh_t + B_b_t  + (1-1/1)*ss[:Y_t])


    #===================================================================#
    #control eq3: government bonds
    @sslogdeviations2levels B_gov_ncp_t = Yt, control_id, ControlSS

    F[:eq_control_government_bond] = B_gov_ncp_t - 
    (B_b_t  + B_hh_t + (1-1/1)*ss[:Y_t] - 
    (Q_t * A_b_t- NW_b_t) - Q_t * (1 + m[:τ_cp].value) * A_g_t)


    #===================================================================#
    #control eq4: tax
    @sslogdeviations2levels T_t, L_t, UB_t = Yt, control_id, ControlSS
    @sslogdeviations2levels Profit_FI_t  = Yt, control_id, ControlSS
    #=
    F[:eq_control_tax] = T_t - (m[:τ_w].value *(w′_t * L_t + UB_t) + m[:τ_w].value*(Profit_t))
=#
 F[:eq_control_tax] = T_t - (m[:τ_w].value * (w′_t * L_t + UB_t) +
        m[:τ_a].value * Profit_t +
        m[:τ_a].value * (Profit_FI_t - m[:fix2].value))



    #===================================================================#
    #control eq5: Lump-sum transfer #TODO: regime
    @sslogdeviations2levels LT_t = Yt, control_id, ControlSS
    regime = "LT"
    if regime == "G"
        F[:eq_control_transfer] = LT_t - ((1 - 1/GG′_t) * ss[:Y_t] -  ss[:C_b_t])
    elseif regime == "LT"
        F[:eq_control_transfer] = LT_t - (ss[:LT_t])
    end



    #===================================================================#
    #control eq6: Government spending #TODO: regime
    @sslogdeviations2levels G_t = Yt, control_id, ControlSS

    regime = "G"
    if regime == "G"
        F[:eq_control_government_spending] = G_t - ss[:G_t]
    elseif regime == "LT" #Doesnt work
        F[:eq_control_government_spending] = G_t - ((1-1/GG′_t)*ss[:Y_t])
    end
    
    #===================================================================#
    #control eq6 Stochastics Discount Factor
    @sslogdeviations2levels MRS_t = Yt, control_id, ControlSS
    F[:eq_control_lambda] = λ_t - MRS_t * m[:λ_aux] / m[:λ_b_aux]

    #===================================================================#
    #control eq7: Inflation
    @sslogdeviations2levels pi_t = Yt, control_id, ControlSS
   
#=    F[:eq_control_inflation] = log(pi_t / m[:π_cb].value) - (m[:π_cb].value * exp(m[:ρ_B].value / (m[:ρ_B].value + m[:γ_π].value))
                                        * log(B_gov_ncp_t * R_cb_t / (ss[:B_gov_ncp_t] * R_cb_t))
                                        - m[:γ_T].value/(m[:ρ_B].value + m[:γ_π].value) * log(T_t / ss[:T_t]) -
                                        1/(m[:ρ_B].value + m[:γ_π].value)*log(B_gov_ncp′_t / ss[:B_gov_ncp_t]))
=#
F[:eq_control_inflation] = log(pi_t / m[:π_cb].value) -
        (m[:ρ_B].value / (m[:ρ_B].value + m[:γ_π].value) *
            log(B_gov_ncp_t * R_cb_t / (ss[:B_gov_ncp_t] * exp(ss[:R_cb_t])))
        - m[:γ_T].value / (m[:ρ_B].value + m[:γ_π].value) * log(T_t / ss[:T_t])
        - 1 / (m[:ρ_B].value + m[:γ_π].value) * log(B_gov_ncp′_t / ss[:B_gov_ncp_t]))

    
    #===================================================================#
    #control eq9: Vacancies
    @sslogdeviations2levels V_t, M_t, J_t = Yt, control_id, ControlSS
    F[:eq_control_vacancies] = V_t - (M_t / m[:ι].value * only(J_t' * grid[:s_dist]) )

    #===================================================================#
    #control eq10 to 14: J #TODO: standardize type of array in ss
    @sslogdeviations2levels nn_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys l_λ_′t, J′_t = Yt1, control_id, ControlSS

    F[:eq_control_j] = J_t - 
        ((h_t - m[:fix_L].value - w_t) .* (grid[:s]) .* nn_t + 
        (λ_t * (1 - m[:dr].value) * (1-l_λ_′t) *
        (1-m[:in].value)) * ss[:P_SS_t] * J′_t)

    #===================================================================#
    #control eq15: r^l
    @sslogdeviations2levels MC_t, K_t, L_t, v_t = Yt, control_id, ControlSS

    F[:eq_control_mpl] = h_t - 
    (Z′_t * MC_t *(v_t * K_t) .^ m[:θ].value .* (L_t).^(m[:θ_2].value - 1) * 
    (m[:θ_2].value + m[:α_lk].value * 
    log(v_t * K_t / ss[:v_t] / grid[:K] ) + 
    m[:α_ll]*log(L_t / ss[:L_t])))

    #===================================================================#
    #control eq16: v
    @sslogdeviations2levels r_k_t = Yt, control_id, ControlSS

    F[:eq_control_elasticity_labor] = v_t - 
    min( (r_k_t/(m[:δ_0].value * m[:δ_1].value ))^(1/(m[:δ_1].value-1)),1)

    #===================================================================#
    #control eq17: Y

    F[:eq_control_output] = Y_t - 
    Z′_t * (v_t * K_t).^(m[:θ].value).*(L_t).^(1-m[:θ].value)

    #===================================================================#
    #control eq18: Profit
    @sslogdeviations2levels_unprimekeys η′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels Profit_FI_t, V_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys K′_t = Yt1, control_id, ControlSS

    F[:eq_control_profit] = Profit_t - 
    ((Y_t * (1- η′_t / (2 * m[:κ]) * (log(pi_t) - (1 - m[:γ]) * log(m[:π_cb]) - 
    m[:γ]*log(π_past_t)) .^ 2) + Y_t.*(-MC_t) - m[:fix] + (h_t  - m[:fix_L] - w_t) .* L_t - 
    m[:ι].*V_t) + (r_k_t * v_t - m[:δ_0].* v_t .^ m[:δ_1]) .* K_t +  
    Q_t *(K′_t- K_t)-(K′_t-K_t) - m[:ϕ]  /2 * (K′_t/K_t-1)^2*K_t  + Profit_FI_t - m[:fix2] )

    #===================================================================#
    #control eq19: r^k
    @sslogdeviations2levels_unprimekeys η′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels Profit_FI_t, V_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys K′_t = Yt1, control_id, ControlSS


    F[:eq_control_mpk] = r_k_t - 
    (Z′_t * MC_t * (v_t * K_t) .^ ( m[:θ] - 1) .* (L_t ) .^ m[:θ_2] * 
    (m[:θ] + m[:α_lk] *log(L_t / ss[:L_t] ) + m[:α_kk] * log(v_t * K_t / ss[:v_t] / grid[:K])))

    #===================================================================#
    #control eq20: r^a
    @sslogdeviations2levels r_a_t = Yt, control_id, ControlSS

    F[:eq_control_mpa] = r_a_t - 
    ((1 - ss[:tau_a_t])*(1 - m[:Eratio] - m[:b_share])* Profit_t / K_t)


    #===================================================================#
    #control eq21: marginal cost
    @sslogdeviations2levels_unprimekeys η2_t = Yt1, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys Y′_t, pi′_t, η2′_t = Yt1, control_id, ControlSS

    F[:eq_control_marginal_cost] = MC_t - 
    (1 - 1/η′_t + (log(pi_t/(π_past_t ^ m[:γ] * ss[:π_t] ^(1 - m[:γ]))) - 
    λ_t * η2′_t/η2_t * Y′_t / Y_t * log(pi′_t / (pi_t^m[:γ] * ss[:π_t] ^ (1-m[:γ])) )) / m[:κ])

    #===================================================================#
    #22-25 unemployment rate

    #control eq25: unemployment rate
    @sslogdeviations2levels_unprimekeys N_t, M_t = Yt1, control_id, ControlSS

    F[:eq_control_unemployment_rate] = unemp_t - 
    ( (1-((1-ss[:lambda_t])*N_t + M_t)- m[:Eshare] ) / (1 - m[:Eshare]) )


    #===================================================================#
    #control eq26: nn
    @sslogdeviations2levels_unprimekeys N_t, M_t, l_λ_t = Yt1, control_id, ControlSS


    F[:eq_control_nn] = nn_t - 
    ( ((1- ss[:tau_w_t] )* w_t / (m[:ψ]))^( 1 / m[:ξ]) )


    #===================================================================#

    #control eq27: M

    F[:eq_control_M] = M_t - 
    ((1 - N_t - m[:Eshare]) + l_λ_t .* (N_t)) .* V_t/((((1 - N_t - m[:Eshare])+
    l_λ_t.*(N_t)) ^ m[:α] + V_t ^ m[:α]) ^ (1 / m[:α]))


    #===================================================================#


    #control eq24: f
    @sslogdeviations2levels f_t = Yt1, control_id, ControlSS


    F[:eq_control_f] = f_t - 
    (M_t / ((1 - N_t - m[:Eshare]) + l_λ_t *(N_t)))


    #===================================================================#

    #control eq25: zz
    @sslogdeviations2levels zz_t = Yt1, control_id, ControlSS

    F[:eq_control_zz] = zz_t - 
    ((RRa_t - m[:b_a_aux2] * R_cb_t / pi_t) * lev_t +  
    m[:b_a_aux2] * R_cb_t / pi_t)


    #===================================================================#
    #control eq26: x
    @sslogdeviations2levels xx_t = Yt1, control_id, ControlSS


    F[:eq_control_xx] = zz_t - 
    ( (lev′_t / lev_t ) * zz_t)


    #===================================================================#
    #control eq27: vv
    @sslogdeviations2levels_unprimekeys BB′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels vv_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys RRa′_t, xx′_t, vv′_t = Yt1, control_id, ControlSS

    F[:eq_control_vv] = vv_t - 
    (((1- m[:θ_b])* BB′_t * λ_t * (RRa′_t - R_cb_t / pi′_t ) +
    m[:θ_b]* BB′_t * λ_t * xx′_t * vv′_t))

    #===================================================================#
    #control eq28: ee
    @sslogdeviations2levels ee_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys zz′_t, ee′_t = Yt1, control_id, ControlSS

    F[:eq_control_ee] = ee_t - 
    ((1-m[:θ_b])* BB′_t * λ_t * R_cb_t / pi′_t + 
    m[:θ_b] * BB′_t * λ_t *zz′_t*ee′_t)

    #===================================================================#
    #control eq29: cb
    @sslogdeviations2levels B_b_t = Xt, state_id, StateSS
    @sslogdeviations2levels_unprimekeys B_b′_t = Xt1, state_id, StateSS
    @sslogdeviations2levels_unprimekeys C_b′_t = Yt1, control_id, ControlSS

    F[:eq_control_cb] = C_b_t - 
    ((1 * m[:β_b] * R_cb_t / pi′_t)/(1 + m[:ϕ_b]*(B_b′_t / B_b_t - 1))*
    C_b′_t ^(-m[:σ_2]))^(-1 / m[:σ_2])

    #===================================================================#
    #control eq30: Profit_FI
   
    F[:eq_control_profit_fi] = Profit_FI_t - 
    ((1- m[:θ_b])*((RRa_t - R_cb_t / pi_t)* lev_t + R_cb_t/pi_t) *
    NW_b_t - m[:ω]    * Q_t * A_b_t)

    #===================================================================#
    #control eq31: rra

    F[:eq_control_rra] = RRa_t - (Q′_t + r_a_t) / Q_t

    #===================================================================#

    #control eq32: rr
    @sslogdeviations2levels RR_t = Yt, control_id, ControlSS

    F[:eq_control_rr] = RR_t - R_cb_t / pi_t


    #===================================================================#
    #control eq33: I
    @sslogdeviations2levels I_t = Yt, control_id, ControlSS
    @sslogdeviations2levels_unprimekeys A_hh′_t = Yt1, control_id, ControlSS

    F[:eq_control_investment] = I_t - 
    (ι′_t * (1+ m[:ϕ] / 2 * (log(x_k_t))^2) * 
    (A_hh′_t + A_g′_t + A_b′_t + ss[:A_F_t]) - 
    (1-m[:δ_0]*v_t^ m[:δ_1]) * K_t;)

    #===================================================================#
    #control eq34: x_k
    @sslogdeviations2levels x_k_t = Yt, control_id, ControlSS

    F[:eq_control_x_k] = x_k_t - K′_t / K_t

    #===================================================================#
    #control eq39: a_g_obs
    @sslogdeviations2levels A_g_obs_t = Yt, control_id, ControlSS

    F[:eq_control_a_g_obs] = A_g_obs_t - (A_gaux′_t - 1)

    #===================================================================#
    #control eq41: y_obs
    @sslogdeviations2levels Y_obs_t = Yt, control_id, ControlSS

    F[:eq_control_y_obs] = Y_obs_t - Y_t


    #===================================================================#
    #control eq42: c_obs
    @sslogdeviations2levels C_obs_t = Yt, control_id, ControlSS

    F[:eq_control_c_obs] = C_obs_t - C_t


    #===================================================================#
    #control eq43: i_obs
    @sslogdeviations2levels I_obs_t = Yt, control_id, ControlSS

    F[:eq_control_i_obs] = I_obs_t - I_t

    #===================================================================#
    #control eq44: i_obs
    @sslogdeviations2levels w_obs_t = Yt, control_id, ControlSS

    F[:eq_control_w_obs] = w_obs_t - w_t

    #===================================================================#
    #control eq45: profit_obs
    @sslogdeviations2levels Profit_obs_t = Yt, control_id, ControlSS

    F[:eq_control_profit_obs] = Profit_obs_t - Profit_t - m[:fix2] - log(B_F′_t)
    #===================================================================#
    # Shocks
    @sslogdeviations2levels_unprimekeys eps_QE′_t, eps_B_F′_t, eps_RP′_t, eps_BB′_t,
      eps_Z′_t, eps_G′_t, eps_D′_t, eps_R′_t, eps_iota′_t, eps_eta′_t, eps_w′_t = Xt1, state_id, StateSS
    F[:eps_QE_ind]   = log(eps_QE′_t)
    F[:eps_B_F_ind]  = log(eps_B_F′_t)
    F[:eps_RP_ind]   = log(eps_RP′_t)
    F[:eps_BB_ind]   = log(eps_BB′_t)
    F[:eps_Z_ind]    = log(eps_Z′_t)
    F[:eps_G_ind]    = log(eps_G′_t)
    F[:eps_D_ind]    = log(eps_D′_t)
    F[:eps_R_ind]    = log(eps_R′_t)
    F[:eps_iota_ind] = log(eps_iota′_t)
    F[:eps_eta_ind]  = log(eps_eta′_t)
    F[:eps_w_ind]    = log(eps_w′_t)
    #===================================================================#


    return F
end

