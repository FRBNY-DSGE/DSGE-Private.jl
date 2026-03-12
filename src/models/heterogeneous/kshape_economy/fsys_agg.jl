using DSGE: @sslogdeviations2levels, @sslogdeviations2levels_unprimekeys, @unpack_and_first
include("helpers/q_cons2.jl")
include("mBBQ.jl")
include("models/7_KShape_Share/index2.jl")    
include("models/7_KShape_Share/macros2.jl")   

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

"""
function Fsys_agg(F::AbstractVector, X::AbstractArray, XPrime::AbstractArray,
                  θ::NamedTuple, grids::OrderedDict, id::OrderedDict{Symbol, I1}, nt::NamedTuple,
                  eq::OrderedDict{Symbol, I2}) where {I1 <: Union{Int, UnitRange}, I2 <: Union{Int, UnitRange}}


                    #TODO: make all of these arguments into function
                  jld2file = "XssYss.jld2"
                  @load jld2file StateSS ControlSS SS_stats grid param
                  
                  m = mBBQ()
                  
                  #one time setup to move prev donggyu format to bbl format
                  grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "nCOP", "oc", "numstates"])
                  state_id, control_id = build_indices(grid, length(StateSS), length(ControlSS))
                  ss = build_ss(param, SS_stats)
                  
                  #test state and control controls of zero
                  State_zero = zeros(length(StateSS))
                  Control_zero = zeros(length(ControlSS))
                  
                  
                  F = Dict{Symbol,Float64}() #TODO: need to change typing of this
                  modelgroupings = Dict()
                  paramgroupings = Dict()
                  
                  
                  #===================================================================#
                  #eq1: eq_rate_monetary_policy TODO: regime change
                  @sslogdeviations2levels R_cb_t, eps_R_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys R_cb′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels pi_t, unemp_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_rate_monetary_policy] = log(R_cb′_t) - ss[:R_cb_t] - 
                  (1 - m[:ρ_R].value) * (m[:φ_π].value * log(pi_t/m[:π_cb].value) -
                  m[:φ_u].value * (unemp_t - ss[:u_t])) -  
                  m[:ρ_R].value*(log(R_cb_t) - ss[:R_cb_t]) - log(eps_R_t)
                  
                  #===================================================================#
                  #eq2: wage
                  @sslogdeviations2levels w_t, eps_w_t, π_past_t, ψ_w_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys w′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels h_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_wage] = log(w′_t) - ss[:w_t] - 
                  m[:ρ_w].value * (log(w_t) - ss[:w_t] + 
                  m[:d].value * log(ss[:π_t]/log(pi_t)) + (1-m[:d].value)*log(log(π_past_t)/log(pi_t))) - 
                  (1 - m[:ρ_w].value) * eps_w_t * log(ψ_w_t + h_t - ss[:r_l_t])
                  
                  #TODO check to see if pi_t is logged before or after add control macro
                  
                  #===================================================================#
                  #eq3: asset a (illiquid)
                  @sslogdeviations2levels lev_t, NW_b_t, Q_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys A_b′_t = State_zero, state_id, StateSS
                  
                  F[:eq_illiquid_assets] = log(A_b′_t) - log( lev_t*NW_b_t/Q_t ) 
                  
                  #===================================================================#
                  #eq4: asset b (bonds)
                  @sslogdeviations2levels A_gaux_t, B_b_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys B_b′_t, Q′_t, A_gaux′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels T_t, B_gov_ncp_t, r_a_t, UB_t, LT_t, C_b_t, G_t = Control_zero, control_id, ControlSS
                  @sslogdeviations2levels_unprimekeys B_gov_ncp′_t = Control_zero, control_id, ControlSS
                  A_g′_t = A_gaux′_t - 1
                  A_g_t = A_gaux_t - 1
                  
                  F[:eq_liquid_assets] =  log(B_b′_t) -
                  log(T_t + B_gov_ncp′_t - B_gov_ncp_t * R_cb_t/pi_t + 
                  (Q′_t + r_a_t - R_cb_t/pi_t*Q_t) * A_g_t - UB_t - LT_t - 
                  m[:τ_cp].value * Q′_t * A_g′_t - G_t + 
                  m[:b_a_aux2].value * R_cb_t / pi_t * B_b_t - 
                  C_b_t ) 
                  
                  #===================================================================#
                  #eq5: central bank assets TODO: regime change
                  @sslogdeviations2levels x_cb_t = State_zero, state_id, StateSS
                  
                  F[:eq_central_bank_assets] = log(A_gaux′_t) - log( x_cb_t * A_g_t + 1 )
                  
                  #===================================================================#
                  #eq6: tobin's q
                  @sslogdeviations2levels ι_2_t, x_k_t, λ_t = Control_zero, control_id, ControlSS
                  @sslogdeviations2levels_unprimekeys ι_2′_t, x_k′_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_tobins_q] = log(Q′_t) - 
                  log( ι_2_t * (1 + m[:ϕ].value * log(x_k_t)+m[:ϕ].value/2*(log(x_k_t))^2 ) + 
                  ι_2′_t * λ_t / ss[:λ_t] * m[:ϕ].value * log(x_k′_t) * x_k′_t )
                  
                  #===================================================================#
                  #eq7: bank leverage
                  @sslogdeviations2levels_unprimekeys lev′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels ee_t, vv_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_leverage] = log(lev′_t) - log( ee_t / (m[:δ].value - vv_t) )
                  
                  #===================================================================#
                  #eq7: bank leverage
                  @sslogdeviations2levels_unprimekeys lev′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels ee_t, vv_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_leverage] = log(lev′_t) - log( ee_t / (m[:δ].value - vv_t) )
                  
                  #===================================================================#
                  #eq8: NW_b_ind
                  @sslogdeviations2levels lev_t, NW_b_t, A_b_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys NW_b′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels RRa_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_net_worth_bank] = log(NW_b′_t) - 
                  log( (m[:θ_b].value *((RRa_t - m[:b_a_aux2].value * R_cb_t / pi_t)*
                  lev_t + 
                  m[:b_a_aux2].value* R_cb_t / pi_t) * NW_b_t + 
                  m[:ω].value * Q_t * A_b_t) ) 
                  
                  #===================================================================#
                  #eq9: NW_b_ind 
                  @sslogdeviations2levels D_t, R_tilde_t = State_zero, state_id, StateSS
                  
                  F[:eq_houshold_bond_rate] = log(R_tilde_t) - log( R_cb_t * D_t ) 
                  
                  #===================================================================#
                  #eq10: eq_quantitative_easing #TODO: add regimes
                  
                  F[:eq_quantitative_easing] = 0
                 
                  #===================================================================#
                  #eq11: inflation lag
                  @sslogdeviations2levels_unprimekeys π_past′_t = State_zero, state_id, StateSS
                  
                  F[:eq_inflation_lag] = log(π_past′_t) - log(pi_t)
                  
                  #===================================================================#
                  #eq12: ouput lag
                  @sslogdeviations2levels_unprimekeys Y_past′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels Y_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_output_lag] = log(Y_past′_t) - log(Y_t)
                  
                  #===================================================================#
                  #eq13: consumption lag
                  @sslogdeviations2levels_unprimekeys C_past′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels C_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_consumption_lag] = log(C_past′_t) - log(C_t)
                  
                  #===================================================================#
                  #eq14: investment lag
                  @sslogdeviations2levels_unprimekeys I_past′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels I_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_investment_lag] = log(I_past′_t) - log(I_t)
                  
                  #===================================================================#
                  #eq15: profit lag
                  @sslogdeviations2levels_unprimekeys Profit_past′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels Profit_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_profit_lag] = log(Profit_past′_t) - log(Profit_t + m[:fix2].value)
                  
                  #===================================================================#
                  #eq16: unemployment lag
                  @sslogdeviations2levels_unprimekeys unemp_past′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels unemp_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_unemployment_lag] = log(unemp_past′_t) - log(unemp_t)
                  
                  #===================================================================#
                  #eq17: government spending lag #TODO: regime
                  @sslogdeviations2levels_unprimekeys G_past′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels G_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_government_spending_lag] = log(G_past′_t) - log(G_t)
                  
                  #===================================================================#
                  #eq18: lump sum transfers
                  @sslogdeviations2levels_unprimekeys LT_past′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels LT_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_lump_sum_transfers] = log(LT_past′_t) - log(LT_t)
                  
                  #===================================================================#
                  #eq19: eq_fiscal_liability
                  @sslogdeviations2levels B_F_t, eps_B_F_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys B_F′_t = State_zero, state_id, StateSS
                  

                  F[:eq_fiscal_liability] = log(B_F′_t) - ( m[:ρ_B_F].value * log(B_F_t) +
                  (1 - m[:ρ_B_F].value) * log(ss[:Y_t] / (ss[:Y_t] - ss[:B_F_t] )) + log(eps_B_F_t))
                  
                  #===================================================================#
                  #eq20: z 
                  @sslogdeviations2levels Z_t, eps_Z_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys Z′_t = State_zero, state_id, StateSS
                  
                  
                  F[:eq_z] = log(Z′_t) - (m[:ρ_Z].value * log(Z_t) + log(eps_Z_t))
                  
                  #===================================================================#
                  #eq21: Ψ 
                  @sslogdeviations2levels ψ_rp_t, eps_RP_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys ψ_rp′_t = State_zero, state_id, StateSS
                  
                  F[:eq_ψ] = log(ψ_rp′_t) - (m[:ρ_ψ_rp].value * log(ψ_rp_t) + log(eps_RP_t))
                  
                  #===================================================================#
                  #eq22: η
                  @sslogdeviations2levels η_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys p_m′_t = State_zero, state_id, StateSS
                  
                  F[:eq_η] = log(η_t) - ( log(p_m′_t) - log(p_m′_t - 1) )
                  
                  #===================================================================#
                  #eq23: z 
                  @sslogdeviations2levels D_t, eps_D_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys D′_t = State_zero, state_id, StateSS
                                    
                  F[:eq_D] = log(D′_t) - (m[:ρ_D].value * log(D_t) + log(eps_D_t))
                  
                  #===================================================================#
                  #eq24: z 
                  @sslogdeviations2levels D_t, eps_D_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys D′_t = State_zero, state_id, StateSS
                                    
                  F[:eq_D] = log(D′_t) - (m[:ρ_D].value * log(D_t) + log(eps_D_t))
                  
                  #===================================================================#
                  #eq24: GG TODO: add regime
                  #SHOCK: eps_G
                  @sslogdeviations2levels GG_t, eps_D_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys GG′_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels G_t, C_b_t = Control_zero, control_id, ControlSS
                  
                  F[:eq_GG] = log(GG′_t) - (m[:ρ_G].value * log(GG_t) + (1 - m[:ρ_G].value) * 
                  log(ss[:Y_t] / (ss[:Y_t] - ss[:LT_t] - ss[:C_b_t])))
                  
                  #===================================================================#
                  #eq25: 
                  #SHOCK: eps_iota
                  @sslogdeviations2levels ι_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys ι′_t = State_zero, state_id, StateSS                  
                  
                  F[:eq_ι] = log(ι′_t) - (m[:ρ_ι].value * log(ι_t) )
                  
                  #===================================================================#
                  #eq27: 
                  @sslogdeviations2levels BB_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys BB′_t = State_zero, state_id, StateSS
                                    
                  F[:eq_BB] = log(BB′_t) - (m[:ρ_BB].value * log(BB_t))
                  
                  #===================================================================#
                  #eq28: psi_w
                  #SHOCK: eps_w
                  @sslogdeviations2levels ψ_w_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys ψ_w′_t = State_zero, state_id, StateSS
                  
                  F[:eq_ψ_w] = log(ψ_w′_t) - (m[:ρ_ψ_w].value * log(ψ_w_t))
                  
                  #===================================================================#
                  #eq29: mp
                  #SHOCK: eps_R
                  @sslogdeviations2levels MP_t = State_zero, state_id, StateSS
                  @sslogdeviations2levels_unprimekeys MP′_t = State_zero, state_id, StateSS
                  
                  F[:eq_MP] = log(MP′_t) - (m[:ρ_mp].value * log(MP_t) )
                  
                  #===================================================================#
                  #eq30: p_m
                  @sslogdeviations2levels p_m_t = State_zero, state_id, StateSS
                  
                  F[:eq_pm] = log(p_m′_t) - (m[:ρ_η].value * log(p_m_t) + 
                  (1-m[:ρ_η].value) * log(m[:η].value / (m[:η].value - 1)))
                 

                  #===================================================================#
    
    return F
end

