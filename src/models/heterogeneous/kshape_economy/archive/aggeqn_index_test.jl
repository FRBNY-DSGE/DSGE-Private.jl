using MAT
using Test
using LinearAlgebra
using OrderedCollections: OrderedDict
using DSGE
using JLD2


include("../../../src/mBBQ.jl")
include("../../../src/models/7_KShape_Share/index2.jl")    
include("../../../src/models/7_KShape_Share/macros2.jl")   


jld2file = "XssYss.jld2"
@load jld2file StateSS ControlSS SS_stats grid param

m = mBBQ()

#one time setup to move prev donggyu format to bbl format
grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "nCOP", "oc", "numstates", "s_dist", "s", "K"])
state_id, control_id = build_indices(grid, length(StateSS), length(ControlSS))
ss = build_ss(param, SS_stats)

#test state and control controls of zero
State_zero = zeros(length(StateSS))
Control_zero = zeros(length(ControlSS))


F = Dict{Symbol,Any}() #TODO: need to change typing of this
modelgroupings = Dict()
paramgroupings = Dict()



function test(eqn_symbol, F, modelgroupings, paramgroupings)
    @testset "$(eqn_symbol)" begin

    atol = 1e-8

    #f is 0
    @testset "F[:] = 0" begin
        if isa(F[eqn_symbol], AbstractArray)
            @test isapprox(F[eqn_symbol], zeros(size(F[eqn_symbol])); atol=atol)
        else
            @test isapprox(F[eqn_symbol], 0.0; atol=atol)
        end
    end

    #compare dsge model's param to matlab param
    @testset "compare model params" begin
        for param_tuple in paramgroupings[eqn_symbol]
            sym = param_tuple[:sym]
            name = param_tuple[:name]
            @testset "m[:$sym]" begin
                @test isapprox(m[sym].value, param[name])
            end
        end
    end

    println("Contents of modelgroupings[$(eqn_symbol)]:")
    for (k, v) in modelgroupings[eqn_symbol]
        println("grouping: $k, Values: ", v)
    end

    #compare all values in the modelgroupings
    @testset "compare values" begin
        for (k, v) in modelgroupings[eqn_symbol]

            base_val = v[1]
            @testset "$k grouping" begin
                for value in v
                    @test isapprox(value, base_val; atol=atol)
                end
            end

        end
    end





    end
end


#===================================================================#
#eq1: eq_rate_monetary_policy TODO: regime change
@sslogdeviations2levels R_cb_t, eps_R_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys R_cb′_t = State_zero, state_id, StateSS
@sslogdeviations2levels pi_t, unemp_t = Control_zero, control_id, ControlSS


F[:eq_rate_monetary_policy] = log(R_cb′_t) - ss[:R_cb_t] - 
(1 - m[:ρ_R].value) * (m[:φ_π].value * log(pi_t/m[:π_cb].value) -
m[:φ_u].value * (unemp_t - ss[:u_t])) -  
m[:ρ_R].value*(log(R_cb_t) - ss[:R_cb_t]) - log(eps_R_t)


#eq1: eq_rate_monetary_policy TEST
modelgroupings[:eq_rate_monetary_policy] = [
    :R_cb => [log(R_cb′_t), ss[:R_cb_t], log(R_cb_t)],
    :pi => [log(pi_t), log(m[:π_cb].value)],
    :unemployment => [unemp_t, ss[:u_t]],
    :shock => [log(eps_R_t)],
]

paramgroupings[:eq_rate_monetary_policy] = [
    (sym = :ρ_R, name = "rho_R"),
    (sym = :φ_π, name = "phi_pi"),
    (sym = :φ_u, name = "phi_u"),
    (sym = :π_cb, name = "pi_cb")
]

test(:eq_rate_monetary_policy, F, modelgroupings, paramgroupings)

#===================================================================#
#eq2: wage
@sslogdeviations2levels w_t, eps_w_t, π_past_t, ψ_w_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys w′_t = State_zero, state_id, StateSS
@sslogdeviations2levels h_t = Control_zero, control_id, ControlSS

# RHS = log(param.w_bar) + 
# param.rho_w * log(Wminus/param.w_bar)
#  + param.rho_w*(param.d*log(param.pi_bar/PI)+(1-param.d)*log(pastpiminus/PI)) + 
#  (1-param.rho_w)*param.eps_w*log(PSI_W*H/SS_stats.r_l);


#delta might be new param d
F[:eq_wage] = log(w′_t) - ss[:w_t] - 
m[:ρ_w].value * (log(w_t) - ss[:w_t] + 
m[:d].value * log(ss[:π_t]/log(pi_t)) + (1-m[:d].value)*log(log(π_past_t)/log(pi_t))) - 
(1 - m[:ρ_w].value) * eps_w_t * log(ψ_w_t + h_t - ss[:r_l_t])

#TODO check to see if pi_t is logged before or after add control macro

#eq2: wage TEST
modelgroupings[:eq_wage] = [
    :w => [log(w′_t), ss[:w_t], log(w_t)],
    :pi => [ss[:π_t], log(pi_t), log(π_past_t)],
    :shock_wage_markup => [ψ_w_t],
    :labor_tightness => [h_t],
    :labor_market_return => [ss[:r_l_t]]
]

paramgroupings[:eq_wage] = [
    (sym = :ρ_w, name = "rho_w"),
    (sym = :d, name = "d")
    # (sym = :π_cb, name = "pi_cb")
]

test(:eq_wage, F, modelgroupings, paramgroupings)


#===================================================================#
#eq3: asset a (illiquid)
@sslogdeviations2levels lev_t, NW_b_t, Q_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys A_b′_t = State_zero, state_id, StateSS
# @sslogdeviations2levels h_t = Control_zero, control_id, ControlSS


#delta might be new param d
F[:eq_illiquid_assets] = log(A_b′_t) - log( lev_t*NW_b_t/Q_t ) 


#eq3: asset a (illiquid) wage TEST
modelgroupings[:eq_illiquid_assets] = [
    :A_b′_t => [log(A_b′_t)],
    :lev_t => [log(lev_t)],
    :NW_b_t => [log(NW_b_t)],
    :Q_t => [log(Q_t)]
]

paramgroupings[:eq_illiquid_assets] = [
]

test(:eq_illiquid_assets, F, modelgroupings, paramgroupings)




#===================================================================#
#eq4: asset b (bonds)
@sslogdeviations2levels A_gaux_t, B_b_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys B_b′_t, Q′_t, A_gaux′_t = State_zero, state_id, StateSS
@sslogdeviations2levels T_t, B_gov_ncp_t, r_a_t, UB_t, LT_t, C_b_t, G_t = Control_zero, control_id, ControlSS
@sslogdeviations2levels_unprimekeys B_gov_ncp′_t = Control_zero, control_id, ControlSS
A_g′_t = A_gaux′_t - 1
A_g_t = A_gaux_t - 1

#RHS(B_b_ind)         = 
#log( T + B_gov_ncpnext - B_gov_ncp*R_cbminus/PI + 
#(Q+R_A-R_cbminus/PI*Qminus)*A_g - UB - LT - 
#param.tau_cp*Q*A_gnext - G + 
#param.b_a_aux2*R_cbminus/PI*B_b - 
#C_b );


#eq4: asset b (bonds)
F[:eq_liquid_assets] =  log(B_b′_t) -
log(T_t + B_gov_ncp′_t - B_gov_ncp_t * R_cb_t/pi_t + 
(Q′_t + r_a_t - R_cb_t/pi_t*Q_t) * A_g_t - UB_t - LT_t - 
m[:τ_cp].value * Q′_t * A_g′_t - G_t + 
m[:b_a_aux2].value * R_cb_t / pi_t * B_b_t - 
C_b_t ) 


# eq4: asset b (bonds) wage TEST
modelgroupings[:eq_liquid_assets] = [
    :B_b′_t => [log(B_b′_t)],
    :T_t => [log(T_t)],
    :B_gov_ncp => [log(B_gov_ncp′_t), log(B_gov_ncp_t)],
    :R_cb_t => [log(R_cb_t)],
    :pi_t => [log(pi_t)],
    :Q_t => [log(Q_t)],
    :r_a_t => [log(r_a_t)],
    :Q′_t => [log(Q′_t)],
    :A_g_t => [log(A_g_t)],
    :UB_t => [log(UB_t)],
    :LT_t => [log(LT_t)],
    :A_g′_t => [log(A_g′_t)],
    :G_t => [log(G_t)],
    :B_b_t => [log(B_b_t)],
    :C_b_t => [log(C_b_t)]
]

paramgroupings[:eq_liquid_assets] = [
    (sym = :τ_cp, name = "tau_cp"),
    (sym = :b_a_aux2, name = "b_a_aux2")
]


test(:eq_liquid_assets, F, modelgroupings, paramgroupings)




#===================================================================#
#eq5: central bank assets TODO: regime change
@sslogdeviations2levels x_cb_t = State_zero, state_id, StateSS


F[:eq_central_bank_assets] = log(A_gaux′_t) - log( x_cb_t * A_g_t + 1 )


# eq5 TEST
modelgroupings[:eq_central_bank_assets] = [
    :A_g => [log(A_gaux′_t), log(A_g_t + 1)],
    :x_cb_t => [log(x_cb_t)],
]

paramgroupings[:eq_central_bank_assets] = [

]


test(:eq_central_bank_assets, F, modelgroupings, paramgroupings)




#===================================================================#
#eq6: tobin's q
# @sslogdeviations2levels  = State_zero, state_id, StateSS
# @sslogdeviations2levels_unprimekeys  = State_zero, state_id, StateSS
@sslogdeviations2levels ι_2_t, x_k_t, λ_t = Control_zero, control_id, ControlSS
@sslogdeviations2levels_unprimekeys ι_2′_t, x_k′_t = Control_zero, control_id, ControlSS

F[:eq_tobins_q] = log(Q′_t) - 
log( ι_2_t * (1 + m[:ϕ].value * log(x_k_t)+m[:ϕ].value/2*(log(x_k_t))^2 ) + 
ι_2′_t * λ_t / ss[:λ_t] * m[:ϕ].value * log(x_k′_t) * x_k′_t )


# eq6 TEST
modelgroupings[:eq_tobins_q] = [
    :Q => [log(Q′_t)],
    :ι_2_t => [log(ι_2_t), log(ι_2′_t)],
    :x_k_t => [log(x_k_t), log(x_k′_t)],
    :λ_t => [λ_t, ss[:λ_t]],
]

paramgroupings[:eq_tobins_q] = [
    (sym = :ϕ, name = "phi")
]


test(:eq_tobins_q, F, modelgroupings, paramgroupings)



#===================================================================#
#eq7: bank leverage
@sslogdeviations2levels_unprimekeys lev′_t = State_zero, state_id, StateSS
@sslogdeviations2levels ee_t, vv_t = Control_zero, control_id, ControlSS


F[:eq_leverage] = log(lev′_t) - log( ee_t / (m[:δ].value - vv_t) )


# eq7 TEST
modelgroupings[:eq_leverage] = [
    :lev′_t => [log(lev′_t)],
    :ee_t => [ee_t],
    :vv_t => [vv_t],
]

paramgroupings[:eq_leverage] = [
    (sym = :δ, name = "DELTA")
]


test(:eq_leverage, F, modelgroupings, paramgroupings)



#===================================================================#
#eq7: bank leverage
@sslogdeviations2levels_unprimekeys lev′_t = State_zero, state_id, StateSS
@sslogdeviations2levels ee_t, vv_t = Control_zero, control_id, ControlSS


F[:eq_leverage] = log(lev′_t) - log( ee_t / (m[:δ].value - vv_t) )


# eq7 TEST
modelgroupings[:eq_leverage] = [
    :lev′_t => [log(lev′_t)],
    :ee_t => [ee_t],
    :vv_t => [vv_t],
]

paramgroupings[:eq_leverage] = [
    (sym = :δ, name = "DELTA")
]


test(:eq_leverage, F, modelgroupings, paramgroupings)


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


# eq8 TEST
modelgroupings[:eq_net_worth_bank] = [
    :NW_b=> [log(NW_b′_t), log(NW_b_t)],
    :lev_t => [lev_t],
    :A_b_t => [A_b_t],
    :RRa_t => [RRa_t],
    :R_cb_t => [R_cb_t],
    :pi_t => [pi_t],
    :Q_t => [Q_t],
]

paramgroupings[:eq_net_worth_bank] = [
    (sym = :θ_b, name = "theta_b"),
    (sym = :b_a_aux2, name = "b_a_aux2"),
    (sym = :ω, name = "omega")
]


test(:eq_net_worth_bank, F, modelgroupings, paramgroupings)



#===================================================================#
#eq9: NW_b_ind
@sslogdeviations2levels D_t, R_tilde_t = State_zero, state_id, StateSS

F[:eq_houshold_bond_rate] = log(R_tilde_t) - log( R_cb_t * D_t ) 


# eq9 TEST
modelgroupings[:eq_houshold_bond_rate] = [
    :D_t => [D_t],
    :R_tilde_t => [R_tilde_t],
    :R_cb_t => [R_cb_t],
]

paramgroupings[:eq_houshold_bond_rate] = [
]


test(:eq_houshold_bond_rate, F, modelgroupings, paramgroupings)


#===================================================================#
#eq10: eq_quantitative_easing #TODO: add regimes

F[:eq_quantitative_easing] = 0

# eq10 TEST
modelgroupings[:eq_quantitative_easing] = [
]

paramgroupings[:eq_quantitative_easing] = [
]


test(:eq_quantitative_easing, F, modelgroupings, paramgroupings)



#===================================================================#
#eq11: inflation lag
@sslogdeviations2levels_unprimekeys π_past′_t = State_zero, state_id, StateSS

F[:eq_inflation_lag] = log(π_past′_t) - log(pi_t)

# eq11 TEST
modelgroupings[:eq_inflation_lag] = [
    :pi=> [log(π_past′_t), log(pi_t)],
]

paramgroupings[:eq_inflation_lag] = []

test(:eq_inflation_lag, F, modelgroupings, paramgroupings)


#===================================================================#
#eq12: ouput lag
@sslogdeviations2levels_unprimekeys Y_past′_t = State_zero, state_id, StateSS
@sslogdeviations2levels Y_t = Control_zero, control_id, ControlSS

F[:eq_output_lag] = log(Y_past′_t) - log(Y_t)

# eq12 TEST
modelgroupings[:eq_output_lag] = [:Y=> [log(Y_past′_t), log(Y_t)]]
paramgroupings[:eq_output_lag] = []

test(:eq_output_lag, F, modelgroupings, paramgroupings)


#===================================================================#
#eq13: consumption lag
@sslogdeviations2levels_unprimekeys C_past′_t = State_zero, state_id, StateSS
@sslogdeviations2levels C_t = Control_zero, control_id, ControlSS

F[:eq_consumption_lag] = log(C_past′_t) - log(C_t)

# eq13 TEST
modelgroupings[:eq_consumption_lag] = [:C=> [log(C_past′_t), log(C_t)]]
paramgroupings[:eq_consumption_lag] = []

test(:eq_consumption_lag, F, modelgroupings, paramgroupings)


#===================================================================#
#eq14: investment lag
@sslogdeviations2levels_unprimekeys I_past′_t = State_zero, state_id, StateSS
@sslogdeviations2levels I_t = Control_zero, control_id, ControlSS

F[:eq_investment_lag] = log(I_past′_t) - log(I_t)

# eq14 TEST
modelgroupings[:eq_investment_lag] = [:I=> [log(I_past′_t), log(I_t)]]
paramgroupings[:eq_investment_lag] = []

test(:eq_investment_lag, F, modelgroupings, paramgroupings)


#===================================================================#
#eq15: profit lag
@sslogdeviations2levels_unprimekeys Profit_past′_t = State_zero, state_id, StateSS
@sslogdeviations2levels Profit_t = Control_zero, control_id, ControlSS

F[:eq_profit_lag] = log(Profit_past′_t) - log(Profit_t + m[:fix2].value)

# eq15 TEST
modelgroupings[:eq_profit_lag] = [
    :PROFIT=> [log(Profit_past′_t)],
    :PROFIT2=> [log(Profit_t)]

]
paramgroupings[:eq_profit_lag] = [
    (sym = :fix2, name = "fix2")
]

test(:eq_profit_lag, F, modelgroupings, paramgroupings)


#===================================================================#
#eq16: unemployment lag
@sslogdeviations2levels_unprimekeys unemp_past′_t = State_zero, state_id, StateSS
@sslogdeviations2levels unemp_t = Control_zero, control_id, ControlSS

F[:eq_unemployment_lag] = log(unemp_past′_t) - log(unemp_t)

# eq16 TEST
modelgroupings[:eq_unemployment_lag] = [:unemp=> [log(unemp_past′_t), log(unemp_t)]]
paramgroupings[:eq_unemployment_lag] = []

test(:eq_unemployment_lag, F, modelgroupings, paramgroupings)


#===================================================================#
#eq17: government spending lag #TODO: regime
@sslogdeviations2levels_unprimekeys G_past′_t = State_zero, state_id, StateSS
@sslogdeviations2levels G_t = Control_zero, control_id, ControlSS

F[:eq_government_spending_lag] = log(G_past′_t) - log(G_t)

# eq17 TEST
modelgroupings[:eq_government_spending_lag] = [:G=> [log(G_past′_t), log(G_t)]]
paramgroupings[:eq_government_spending_lag] = []

test(:eq_government_spending_lag, F, modelgroupings, paramgroupings)


#===================================================================#
#eq18: lump sum transfers
@sslogdeviations2levels_unprimekeys LT_past′_t = State_zero, state_id, StateSS
@sslogdeviations2levels LT_t = Control_zero, control_id, ControlSS

F[:eq_lump_sum_transfers] = log(LT_past′_t) - log(LT_t)

# eq18 TEST
modelgroupings[:eq_lump_sum_transfers] = [:LT=> [log(LT_past′_t), log(LT_t)]]
paramgroupings[:eq_lump_sum_transfers] = []

test(:eq_lump_sum_transfers, F, modelgroupings, paramgroupings)


#===================================================================#
#eq19: eq_fiscal_liability

@sslogdeviations2levels B_F_t, eps_B_F_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys B_F′_t = State_zero, state_id, StateSS
#RHS(B_F_ind)         = 
#param.rho_B_F*log(B_F) + 
#(1-param.rho_B_F)*log(SS_stats.Y/(SS_stats.Y-SS_stats.B_F)) + 
#eps_B_F;
F[:eq_fiscal_liability] = log(B_F′_t) - ( m[:ρ_B_F].value * log(B_F_t) +
(1 - m[:ρ_B_F].value) * log(ss[:Y_t] / (ss[:Y_t] - ss[:B_F_t] )) + log(eps_B_F_t))

# eq19 TEST
modelgroupings[:eq_fiscal_liability] = [
    :B_F => [log(B_F′_t), log(B_F_t), ss[:B_F_t]],
    :ssY => [ss[:Y_t]],
    :eps_B_F_t => [log(eps_B_F_t)]
]


paramgroupings[:eq_fiscal_liability] = [
    (sym = :ρ_B_F, name = "rho_B_F")
]

test(:eq_fiscal_liability, F, modelgroupings, paramgroupings)


#===================================================================#
#eq20: z 

@sslogdeviations2levels Z_t, eps_Z_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys Z′_t = State_zero, state_id, StateSS


F[:eq_z] = log(Z′_t) - (m[:ρ_Z].value * log(Z_t) + log(eps_Z_t))

# eq20 TEST
modelgroupings[:eq_z] = [
    :Z => [log(Z′_t), log(Z_t)],
    :epsZ => [eps_Z_t]
]

paramgroupings[:eq_z] = [
    (sym = :ρ_Z, name = "rho_Z")

]

test(:eq_z, F, modelgroupings, paramgroupings)


#===================================================================#
#eq21: Ψ 

@sslogdeviations2levels ψ_rp_t, eps_RP_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys ψ_rp′_t = State_zero, state_id, StateSS

F[:eq_ψ] = log(ψ_rp′_t) - (m[:ρ_ψ_rp].value * log(ψ_rp_t) + log(eps_RP_t))

# eq21 TEST
modelgroupings[:eq_ψ] = [
    :value => [log(ψ_rp′_t), log(ψ_rp_t)],
    :shock => [eps_RP_t]
]

paramgroupings[:eq_ψ] = [
    (sym = :ρ_ψ_rp, name = "rho_PSI_RP")
]

test(:eq_ψ, F, modelgroupings, paramgroupings)



#===================================================================#
#eq22: η

@sslogdeviations2levels η_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys p_m′_t = State_zero, state_id, StateSS

#RHS(eta_ind)         = log(p_mark/(p_mark-1));
F[:eq_η] = log(η_t) - ( log(p_m′_t) - log(p_m′_t - 1) )

# eq22 TEST
modelgroupings[:eq_η] = [
    :p_m′_t => [log(p_m′_t)],
    :η_t => [log(η_t)]
]

paramgroupings[:eq_η] = [
]

test(:eq_η, F, modelgroupings, paramgroupings)



#===================================================================#
#eq23: z 

@sslogdeviations2levels D_t, eps_D_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys D′_t = State_zero, state_id, StateSS

# param.rho_D        * log(Dminus)        + eps_D;

F[:eq_D] = log(D′_t) - (m[:ρ_D].value * log(D_t) + log(eps_D_t))

# eq23 TEST
modelgroupings[:eq_D] = [
    :value => [log(D′_t), log(D_t)],
    :shock => [eps_D_t]
]

paramgroupings[:eq_D] = [
    (sym = :ρ_D, name = "rho_D")

]

test(:eq_D, F, modelgroupings, paramgroupings)



#===================================================================#
#eq24: z 

@sslogdeviations2levels D_t, eps_D_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys D′_t = State_zero, state_id, StateSS

# param.rho_D        * log(Dminus)        + eps_D;

F[:eq_D] = log(D′_t) - (m[:ρ_D].value * log(D_t) + log(eps_D_t))

# eq23 TEST
modelgroupings[:eq_D] = [
    :value => [log(D′_t), log(D_t)],
    :shock => [eps_D_t]
]

paramgroupings[:eq_D] = [
    (sym = :ρ_D, name = "rho_D")

]

test(:eq_D, F, modelgroupings, paramgroupings)





#===================================================================#
#eq24: GG TODO: add regime
#SHOCK: eps_G
@sslogdeviations2levels GG_t, eps_D_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys GG′_t = State_zero, state_id, StateSS
@sslogdeviations2levels G_t, C_b_t = Control_zero, control_id, ControlSS

#param.rho_G  * log(GGminus) + 
#(1-param.rho_G) * log(SS_stats.Y/(SS_stats.Y-SS_stats.LT-SS_stats.C_b)) + eps_G;

F[:eq_GG] = log(GG′_t) - (m[:ρ_G].value * log(GG_t) + (1 - m[:ρ_G].value) * 
log(ss[:Y_t] / (ss[:Y_t] - ss[:LT_t] - ss[:C_b_t])))

# eq24 TEST
modelgroupings[:eq_GG] = [
    :g => [log(GG′_t), log(GG_t)],
    :ssy => [log(ss[:Y_t])],
    :sslt => [log(ss[:LT_t])],
    :sscb => [log(ss[:C_b_t])]
]

paramgroupings[:eq_GG] = [
    (sym = :ρ_G, name = "rho_G")

]

test(:eq_GG, F, modelgroupings, paramgroupings)





#===================================================================#
#eq25: 
#SHOCK: eps_iota
@sslogdeviations2levels ι_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys ι′_t = State_zero, state_id, StateSS

#param.rho_iota  * log(iotaminus)  + eps_iota;


F[:eq_ι] = log(ι′_t) - (m[:ρ_ι].value * log(ι_t) )

# eq24 TEST
modelgroupings[:eq_ι] = [
    :iota => [log(ι′_t), log(ι_t)],
]

paramgroupings[:eq_ι] = [
    (sym = :ρ_ι, name = "rho_iota")

]

test(:eq_ι, F, modelgroupings, paramgroupings)



#===================================================================#
#eq27: 
#SHOCK: eps_BB
@sslogdeviations2levels BB_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys BB′_t = State_zero, state_id, StateSS

#param.rho_iota  * log(iotaminus)  + eps_iota;


F[:eq_BB] = log(BB′_t) - (m[:ρ_BB].value * log(BB_t))

# eq26 TEST
modelgroupings[:eq_BB] = [
    :bb => [log(BB′_t), log(BB_t)],
]

paramgroupings[:eq_BB] = [
    (sym = :ρ_BB, name = "rho_BB")

]

test(:eq_BB, F, modelgroupings, paramgroupings)


#===================================================================#
#eq28: psi_w
#SHOCK: eps_w
@sslogdeviations2levels ψ_w_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys ψ_w′_t = State_zero, state_id, StateSS

F[:eq_ψ_w] = log(ψ_w′_t) - (m[:ρ_ψ_w].value * log(ψ_w_t))

# eq26 TEST
modelgroupings[:eq_ψ_w] = [
    :psi => [log(ψ_w_t), log(ψ_w′_t)],
]

paramgroupings[:eq_ψ_w] = [
    (sym = :ρ_ψ_w, name = "rho_PSI_W")

]

test(:eq_ψ_w, F, modelgroupings, paramgroupings)


#===================================================================#
#eq29: mp
#SHOCK: eps_R
@sslogdeviations2levels MP_t = State_zero, state_id, StateSS
@sslogdeviations2levels_unprimekeys MP′_t = State_zero, state_id, StateSS

F[:eq_MP] = log(MP′_t) - (m[:ρ_mp].value * log(MP_t) )

# eq29 TEST
modelgroupings[:eq_MP] = [
    :psi => [log(MP_t), log(MP′_t)],
]

paramgroupings[:eq_MP] = [
    (sym = :ρ_mp, name = "rho_MP")

]

test(:eq_MP, F, modelgroupings, paramgroupings)




#===================================================================#
#eq30: p_m

@sslogdeviations2levels p_m_t = State_zero, state_id, StateSS

# param.rho_eta*log(p_markminus) + 
#(1-param.rho_eta)*log(param.eta/(param.eta-1)) + eps_eta;
F[:eq_pm] = log(p_m′_t) - (m[:ρ_η].value * log(p_m_t) + 
(1-m[:ρ_η].value) * log(m[:η].value / (m[:η].value - 1)))

# eq22 TEST
modelgroupings[:eq_pm] = [
    :p_m′_t => [log(p_m′_t), log(p_m_t)],
]

paramgroupings[:eq_pm] = [
    (sym = :ρ_η, name = "rho_eta"),
    (sym = :η, name = "eta")


]

test(:eq_pm, F, modelgroupings, paramgroupings)



#===================================================================#
#control eq1: capital
@sslogdeviations2levels K_t, A_hh_t = Control_zero, control_id, ControlSS
@sslogdeviations2levels A_b_t = State_zero, state_id, StateSS

F[:eq_control_capital] = log(K_t) - log(A_hh_t + A_g_t  + A_b_t + ss[:A_F_t])

# eq1 TEST
modelgroupings[:eq_control_capital] = [
    :K_t => [K_t],
    :A_hh_t => [A_hh_t],
    :A_g_t => [A_g_t],
    :A_b_t => [A_b_t],
]

paramgroupings[:eq_control_capital] = [

]

test(:eq_control_capital, F, modelgroupings, paramgroupings)

#===================================================================#
#control eq2: bonds
@sslogdeviations2levels B_b_t = State_zero, state_id, StateSS
@sslogdeviations2levels B_t, B_hh_t = Control_zero, control_id, ControlSS

F[:eq_control_bond] = B_t - (B_hh_t + B_b_t  + (1-1/1)*ss[:Y_t])

# eq2 TEST
modelgroupings[:eq_control_bond] = [
    :B_t => [B_t],
    :B_hh_t => [B_hh_t],
    :B_b_t => [B_b_t],
]

paramgroupings[:eq_control_bond] = [

]

test(:eq_control_bond, F, modelgroupings, paramgroupings)



#===================================================================#
#control eq3: government bonds
@sslogdeviations2levels B_gov_ncp_t = Control_zero, control_id, ControlSS

F[:eq_control_government_bond] = B_gov_ncp_t - 
(B_b_t  + B_hh_t + (1-1/1)*ss[:Y_t] - 
(Q_t * A_b_t- NW_b_t) - Q_t * (1 + m[:τ_cp].value) * A_g_t)


# eq3 TEST
modelgroupings[:eq_control_government_bond] = [
    :B_gov_ncp_t => [B_gov_ncp_t],
    :B_hh_t => [B_hh_t],
    :B_b_t => [B_b_t],
    :A_b_t => [A_b_t],
    :A_g_t => [A_g_t],
    :Q_t => [Q_t],
    :NW_b_t => [NW_b_t],
]

paramgroupings[:eq_control_government_bond] = [
    (sym = :τ_cp, name = "tau_cp")

]

test(:eq_control_government_bond, F, modelgroupings, paramgroupings)




#===================================================================#
#control eq4: tax
@sslogdeviations2levels T_t, L_t, UB_t = Control_zero, control_id, ControlSS

F[:eq_control_tax] = T_t - (m[:τ_w].value *(w′_t * L_t + UB_t) + m[:τ_w].value*(Profit_t))

# eq4 TEST
modelgroupings[:eq_control_tax] = [
    :T_t => [T_t],
    :L_t => [L_t],
    :UB_t => [UB_t],
    :w′_t => [w′_t],
    :Profit_t => [Profit_t],
]

paramgroupings[:eq_control_tax] = [
    (sym = :τ_w, name = "tau_w"),
    (sym = :τ_a, name = "tau_a")

]

test(:eq_control_tax, F, modelgroupings, paramgroupings)

#===================================================================#
#control eq5: Lump-sum transfer #TODO: regime
@sslogdeviations2levels LT_t = Control_zero, control_id, ControlSS
F[:eq_control_transfer] = LT_t - ((1 - 1/GG_t) * ss[:Y_t] -  ss[:C_b_t])

#eq5 TEST
modelgroupings[:eq_control_transfer] = [:g => [log(GG_t)], 
                                     :ssy => [log(ss[:Y_t])],
                                     :LT_t => [LT_t],
                                     :sscb => [log(ss[:C_b_t])]
                                    ]
paramgroupings[:eq_control_transfer] = []
test(:eq_control_transfer, F, modelgroupings, paramgroupings)

#===================================================================#
#control eq6: Government spending #TODO: regime
@sslogdeviations2levels G_t = Control_zero, control_id, ControlSS
F[:eq_control_government_spending] = G_t - ss[:G_t]

modelgroupings[:eq_control_government_spending] = [:g => [log(G_t)],
                                                   :ssg => [log(ss[:G_t])]]

paramgroupings[:eq_control_government_spending] = []
test(:eq_control_government_spending, F, modelgroupings, paramgroupings)
#===================================================================#
#control eq6 Stochastics Discount Factor
@sslogdeviations2levels MRS_t = Control_zero, control_id, ControlSS
F[:eq_control_discount] = log(MRS_t) * m[:λ_aux].value / m[:λ_b_aux].value


# model

#===================================================================#
#control eq7: Inflation
@sslogdeviations2levels pi_t = Control_zero, control_id, ControlSS
# ρ_B = m[:ρ_B].value
# γ_π = m[:γ_π].value
#=
F[:eq_control_inflation] = pi_t - (m[:π_cb].value * exp(ρ_B / (ρ_B + γ_π)) 
                                   * log(B_gov_ncp_t * R_cb_t / (ss[:B_gov_ncp_t] * R_cb_t))
                                   - m[:γ_T].value/(ρ_B + γ_π) * log(T_t / ss[:T_t]) - 
                                   1/(ρ_B + γ_π)*log(B_gov_ncp′_t / ss[:B_gov_ncp_t])) =#
F[:eq_control_inflation] = log(pi_t / m[:π_cb].value) - (m[:π_cb].value * exp(m[:ρ_B].value / (m[:ρ_B].value + m[:γ_π].value))
                                    * log(B_gov_ncp_t * R_cb_t / (ss[:B_gov_ncp_t] * R_cb_t))
                                    - m[:γ_T].value/(m[:ρ_B].value + m[:γ_π].value) * log(T_t / ss[:T_t]) -
                                    1/(m[:ρ_B].value + m[:γ_π].value)*log(B_gov_ncp′_t / ss[:B_gov_ncp_t]))

modelgroupings[:eq_control_inflation] = [
                                         :pi_t => [log(pi_t)],
                                         :T_t => [T_t, ss[:T_t]],
                                         :B_gov_ncp_t => [B_gov_ncp_t, ],
                                         :RHS => [(m[:π_cb].value * exp(ρ_B / (ρ_B + γ_π)) 
                                   * log(B_gov_ncp_t * R_cb_t / (ss[:B_gov_ncp_t] * R_cb_t))
                                   - m[:γ_T].value/(ρ_B + γ_π) * log(T_t / ss[:T_t]) -
                                   1/(ρ_B + γ_π)*log(B_gov_ncp′_t / ss[:B_gov_ncp_t]))]
                                        ]

paramgroupings[:eq_control_inflation] = [
                                         (sym = :ρ_B, name = "rho_B"),
                                         (sym = :γ_π, name = "gamma_pi"),
                                         (sym = :γ_T, name = "gamma_T"),
                                         (sym = :π_cb, name = "pi_cb")
                                        ]

test(:eq_control_inflation, F, modelgroupings, paramgroupings)





#===================================================================#
#control eq9: Vacancies
@sslogdeviations2levels V_t, M_t, J_t = Control_zero, control_id, ControlSS
F[:eq_control_vacancies] = V_t - (M_t / m[:ι].value * only(J_t' * grid[:s_dist]) )

modelgroupings[:eq_control_vacancies] = [
    :V_t => [V_t], 
    :M_t => [M_t], 
    :ι => [m[:ι].value], 
    :J_t => [J_t], 
    :s_dist => [grid[:s_dist]]]

paramgroupings[:eq_control_vacancies] = [
    (sym = :ι, name = "iota"),
]


test(:eq_control_vacancies, F, modelgroupings, paramgroupings)



#===================================================================#
#control eq10 to 14: J #TODO: standardize type of array in ss
@sslogdeviations2levels nn_t = Control_zero, control_id, ControlSS
@sslogdeviations2levels_unprimekeys l_λ_′t, J′_t = Control_zero, control_id, ControlSS

# RHS(nx+J_ind)      = 
#(H-param.fix_L-W).*(grid.s')*nn + 
#LAMBDA*(1-param.death_rate)*(1-l_lambdanext)*(1-param.in)*param.P_SS*Jnext;



F[:eq_control_j] = J_t - 
    ((h_t - m[:fix_L].value - w_t) .* (grid[:s]) .* nn_t + 
    (λ_t * (1 - m[:dr].value) * (1-l_λ_′t) *
    (1-m[:in].value)) * ss[:P_SS_t] * J′_t)

modelgroupings[:eq_control_j] = [
    :J_t => [J_t, J′_t],
    :h_t => [h_t],
    :w_t => [w_t],
    :s => [grid[:s]],
    :nn_t => [nn_t],
    :λ_t => [λ_t],
    :P_SS_t => [ss[:P_SS_t], param["P_SS"]],
]

paramgroupings[:eq_control_j] = [
    (sym = :fix_L, name = "fix_L"),
    (sym = :dr, name = "death_rate"),
    (sym = :in, name = "in"),

]


test(:eq_control_j, F, modelgroupings, paramgroupings)


#===================================================================#



#===================================================================#
#control eq15: r^l
@sslogdeviations2levels MC_t, K_t, L_t, v_t = Control_zero, control_id, ControlSS

#RHS(nx+h_ind) = Z*MC*(v*K).^param.theta.*(L).^(param.theta_2-1)*(param.theta_2+param.alpha_lk*log(v*K/SS_stats.v/grid.K)+param.alpha_ll*log(L/SS_stats.L)); 


F[:eq_control_mpl] = h_t - 
(Z′_t * MC_t *(v_t * K_t) .^ m[:θ].value .* (L_t).^(m[:θ_2].value - 1) * 
(m[:θ_2].value + m[:α_lk].value * 
log(v_t * K_t / ss[:v_t] / grid[:K] ) + 
m[:α_ll]*log(L_t / ss[:L_t])))

modelgroupings[:eq_control_mpl] = OrderedDict(
    :MC_t => [MC_t],
    :K_t => [K_t],
    :L_t => [L_t],
    :v_t => [v_t],
    :Z′_t => [Z′_t],
    :h_t => [h_t],
)

paramgroupings[:eq_control_mpl] = [
    (sym = :θ, name = "theta"),
    (sym = :θ_2, name = "theta_2"),
    (sym = :α_lk, name = "alpha_lk"),
    (sym = :α_ll, name = "alpha_ll"),
]


test(:eq_control_mpl, F, modelgroupings, paramgroupings)


#===================================================================#
#control eq16: v
@sslogdeviations2levels r_k_t = Control_zero, control_id, ControlSS


F[:eq_control_elasticity_labor] = v_t - 
min( (r_k_t/(m[:δ_0].value * m[:δ_1].value ))^(1/(m[:δ_1].value-1)),1)


modelgroupings[:eq_control_elasticity_labor] = OrderedDict(
    :r_k_t => [r_k_t],
    :v_t => [v_t],
)

paramgroupings[:eq_control_elasticity_labor] = [
    (sym = :δ_0, name = "delta_0"),
    (sym = :δ_1, name = "delta_1"),
]


test(:eq_control_elasticity_labor, F, modelgroupings, paramgroupings)



#===================================================================#
#control eq17: Y


F[:eq_control_output] = Y_t - 
Z′_t * (v_t * K_t).^(m[:θ].value).*(L_t).^(1-m[:θ].value)


modelgroupings[:eq_control_output] = OrderedDict(
   
)

paramgroupings[:eq_control_output] = [
    (sym = :θ, name = "theta"),
]


test(:eq_control_output, F, modelgroupings, paramgroupings)



#===================================================================#
#control eq18: Profit
@sslogdeviations2levels_unprimekeys η′_t = State_zero, state_id, StateSS
@sslogdeviations2levels Profit_FI_t, V_t = Control_zero, control_id, ControlSS
@sslogdeviations2levels_unprimekeys K′_t = Control_zero, control_id, ControlSS

#(Y *(1-eta/(2*param.kappa)*(log(PI)-(1-param.GAMMA)*log(param.pi_cb)-
#param.GAMMA*log(pastpiminus)).^2)+ Y.*(-MC) - param.fix + (H-param.fix_L-W).*L - 
#param.iota.*V) + (R_K*v - param.delta_0.*v.^param.delta_1).*K +  
#Q*(Knext-K)-(Knext-K) - param.phi/2*(Knext/K-1)^2*K  + Profit_FI - param.fix2 


F[:eq_control_profit] = Profit_t - 
((Y_t * (1- η′_t / (2 * m[:κ]) * (log(pi_t) - (1 - m[:γ]) * log(m[:π_cb]) - 
m[:γ]*log(π_past_t)) .^ 2) + Y_t.*(-MC_t) - m[:fix] + (h_t  - m[:fix_L] - w_t) .* L_t - 
m[:ι].*V_t) + (r_k_t * v_t - m[:δ_0].* v_t .^ m[:δ_1]) .* K_t +  
Q_t *(K′_t- K_t)-(K′_t-K_t) - m[:ϕ]  /2 * (K′_t/K_t-1)^2*K_t  + Profit_FI_t - m[:fix2] )


modelgroupings[:eq_control_profit] = OrderedDict(
    :Profit_FI_t => [Profit_FI_t],
    :K′_t => [K′_t],
    :K_t => [K_t],
    :Q_t => [Q_t],
    :V_t => [V_t],
    :r_k_t => [r_k_t],
    :π_t => [pi_t],
    :π_past_t => [π_past_t],
    :L_t => [L_t],
    :MC_t => [MC_t],
    :h_t => [h_t],
    :w_t => [w_t],
    :Y_t => [Y_t],
)

paramgroupings[:eq_control_profit] = [
    (sym = :θ, name = "theta"),
    (sym = :η, name = "eta"),
    (sym = :κ, name = "kappa"),
    (sym = :γ, name = "GAMMA"),
    (sym = :π_cb, name = "pi_cb"),
    (sym = :fix, name = "fix"),
    (sym = :fix_L, name = "fix_L"),
    (sym = :ι, name = "iota"),
    (sym = :δ_0, name = "delta_0"),
    (sym = :δ_1, name = "delta_1"),
    (sym = :ϕ, name = "phi"),
    (sym = :fix2, name = "fix2"),
]


test(:eq_control_profit, F, modelgroupings, paramgroupings)


#===================================================================#

#control eq19: r^k
@sslogdeviations2levels_unprimekeys η′_t = State_zero, state_id, StateSS
@sslogdeviations2levels Profit_FI_t, V_t = Control_zero, control_id, ControlSS
@sslogdeviations2levels_unprimekeys K′_t = Control_zero, control_id, ControlSS


F[:eq_control_mpk] = r_k_t - 
(Z′_t * MC_t * (v_t * K_t) .^ ( m[:θ] - 1) .* (L_t ) .^ m[:θ_2] * 
(m[:θ] + m[:α_lk] *log(L_t / ss[:L_t] ) + m[:α_kk] * log(v_t * K_t / ss[:v_t] / grid[:K])))


modelgroupings[:eq_control_mpk] = OrderedDict(
    :r_k_t => [r_k_t],
    :Z′_t => [Z′_t],
    :MC_t => [MC_t],
    :v_t => [v_t],
    :K_t => [K_t],
    :L_t => [L_t],
    :ssL_t => [ss[:L_t]],
    :ssv_t => [ss[:v_t]]
)

paramgroupings[:eq_control_mpk] = [
    (sym = :θ, name = "theta"),
    (sym = :θ_2, name = "theta_2"),
    (sym = :α_lk, name = "alpha_lk"),
    (sym = :α_kk, name = "alpha_kk"),
]


test(:eq_control_mpk, F, modelgroupings, paramgroupings)


#===================================================================#


#control eq20: r^a
@sslogdeviations2levels r_a_t = Control_zero, control_id, ControlSS


F[:eq_control_mpa] = r_a_t - 
((1 - exp(ss[:tau_a_t]))*(1 - m[:Eratio] - m[:b_share])* Profit_t / K_t)


modelgroupings[:eq_control_mpa] = OrderedDict(
    :r_a_t => [r_a_t],
    :Profit_t => [Profit_t],
    :K_t => [K_t],
)

paramgroupings[:eq_control_mpa] = [
    (sym = :Eratio, name = "Eratio"),
    (sym = :b_share, name = "b_share"),
]


test(:eq_control_mpa, F, modelgroupings, paramgroupings)


#===================================================================#


#control eq21: marginal cost
@sslogdeviations2levels_unprimekeys η′_t = State_zero, state_id, StateSS
@sslogdeviations2levels Profit_FI_t, V_t = Control_zero, control_id, ControlSS
@sslogdeviations2levels_unprimekeys K′_t = Control_zero, control_id, ControlSS


F[:eq_control_marginal_cost] = r_k_t - 
(Z′_t * MC_t * (v_t * K_t) .^ ( m[:θ] - 1) .* (L_t ) .^ m[:θ_2] * 
(m[:θ] + m[:α_lk] *log(L_t / ss[:L_t] ) + m[:α_kk] * log(v_t * K_t / ss[:v_t] / grid[:K])))


modelgroupings[:eq_control_marginal_cost] = OrderedDict(
    :r_k_t => [r_k_t],
    :Z′_t => [Z′_t],
    :MC_t => [MC_t],
    :v_t => [v_t],
    :K_t => [K_t],
    :L_t => [L_t],
    :ssL_t => [ss[:L_t]],
    :ssv_t => [ss[:v_t]]
)

paramgroupings[:eq_control_marginal_cost] = [
    (sym = :θ, name = "theta"),
    (sym = :θ_2, name = "theta_2"),
    (sym = :α_lk, name = "alpha_lk"),
    (sym = :α_kk, name = "alpha_kk"),
]


test(:eq_control_marginal_cost, F, modelgroupings, paramgroupings)


#===================================================================#


