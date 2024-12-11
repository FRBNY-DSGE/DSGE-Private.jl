#Define eye function for MATLAB compatibility
@inline eye(n::Integer) = Matrix{Float64}(I,n,n)

function eqcond(m::OnionModel) #m::OnionModel

    n = get_setting(m, :n_sectors)
    # A x_{t+1} = B x_t
    if :eq_srec_1 ∉ collect(keys(m.equilibrium_conditions))
        println("Entering if statement in eqcond")
        #This means we have done a normalization. Need to reset.

        endogenous_states = [[Symbol("s_$(i)") for i in 1:n]; #(log deviation of) real sectoral prices
                             [Symbol("π_$i") for i in 1:n]; #sectoral inflation
                             [:r_t, :c_t, :πc_t, :πw_t, :w_t, :πKc_t] ; #interest rate, cons, CPI, wage Infl, wages
                             [:a_t, :b_t, :μw, :lτ, :τ, :πstar, :mp_t];
                             [Symbol("Eπ_$i") for i in 1:n];
                             [:Ec_t, :Eπc_t, :Eπw_t];
                             [Symbol("μ_$(i)") for i in collect(keys(get_setting(m, :subgroup_names)))]]

        endogenous_states_augmented = [:w_t1, :c_t1, :r_t1, :πc_t1] #, :e_meas_πc_t

        equilibrium_conditions = [[Symbol("eq_pc_$i") for i in 1:n];
                                  [Symbol("eq_srec_$i") for i in 1:n];
                                  [:eq_cpi, :eq_wpc, :eq_wrec, :eq_monpol, :eq_euler];
                                  [:eq_a_t,:eq_b_t,:eq_μw, :eq_τ, :eq_lτdef, :eq_πstar, :eq_mp_t];
                                  [:eq_Ect, :eq_Eπct, :eq_Eπwt, :eq_Kcpi];
                                  [Symbol("eq_Eπ_$i") for i in 1:n];
                                  [Symbol("eq_μ_$(i)") for i in collect(keys(get_setting(m, :subgroup_names)))]]

        for (i,k) in enumerate(equilibrium_conditions); m.equilibrium_conditions[k] = i end
        for (i,k) in enumerate(endogenous_states); m.endogenous_states[k] = i end
        for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i + length(endogenous_states) end


    end




    eq     = m.equilibrium_conditions
    endo   = m.endogenous_states
    exo    = m.exogenous_shocks
    exp_sh = m.expected_shocks

    N = length(m.endogenous_states)


    #@show length(m.endogenous_states), length(m.equilibrium_conditions)
    Γ0 = zeros(N,N)
    Γ1 = zeros(N,N)
    Ψ = zeros(N, length(exo))
    Π = zeros(N, length(exp_sh))
    Const1 = zeros(N-1) #define over N-1 to correct for state reduction in Norm step

    #@show size(Γ0), size(Γ1)

    inpshare = get_setting(m, :inpshare)
#= OLD MATLAB PARAMETERS
    # variable indices (order states first)
    ls  = 1:n      # lagged values of s - endogenous state
    lw  = n+1       # lagged real wages
    li  = n+2       # lagged nominal interest rate
    ltau = n+3      # lagged carbon tax
    tau = n+4       # carbon tax - exogenous state
    pii = n+5:2*n+4 # sectoral inflation - jump
    pic = 2*n+5     # cpi inflation - jump
    c   = 2*n+6     # consumption - jump
    piw = 2*n+7     # nominal wage inflation - jump

    # equation block indices
    pc     = 1:n     # phillips curve
    srec   = n+1:2*n # s recursion
    cpi    = 2*n+1   # cpi definition
    wpc    = 2*n+2   # wage phillips curve
    wrec   = 2*n+3   # real wage recursion
    monpol = 2*n+4   # monetary policy rule
    taurec = 2*n+5   # law of motion for tau
    euler  = 2*n+6   # euler eqn
    ltaudef = 2*n+7  # defn of lagged tau
=#

    #-m[:taxshare].value For oil kanzig IRFs, vector of 0 with 1 at oil index.
    if get_setting(m, :irf_type) == "oil"
        sec_shock = zeros(size(get_setting(m, :taxshare)))   #zeros(size(m[:taxshare].value))
        sec_shock[Int(get_setting(m, :oil))] = 1.
        #sec_shock[Int(m[:oil].value)] = 1.
    else
        sec_shock = get_setting(m, :taxshare)   #m[:taxshare].value
    end



# Phillips curve
Γ0[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[Symbol("π_1")]:endo[Symbol("π_$n")]]    = diagm(get_setting(m, :invkap))    #diagm(m[:invkap].value)
    Γ0[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[:τ]]  = - sec_shock'
    Γ0[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[:w_t]] = -get_setting(m, :labshare)    #- m[:labshare].value
    Γ0[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[:a_t]] = get_setting(m, :labshare)  #m[:labshare].value
    Γ0[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[Symbol("s_1")]:endo[Symbol("s_$n")]]  = - (inpshare - eye(n))
    Γ0[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[Symbol("Eπ_1")]:endo[Symbol("Eπ_$n")]]    = m[:bet] * diagm(get_setting(m, :invkap))      #- m[:bet]*diagm(m[:invkap].value)
    #Addl term for markup stochastic trend
    #Γ0[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[Symbol("mkup_trend_1")]:endo[Symbol("mkup_trend_$n")]] = - diagm(m[:invkap].value) # eye(n) #Addl term for markup shocks
    #IID markup shock shock
    #Γ0[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[Symbol("mkup_iid_1")]:endo[Symbol("mkup_iid_$n")]] = - diagm(m[:invkap].value)






    #Need to be careful here. I want to define this such that the invkap is 0 for sectors not in the relevant category.
    #e.g. if sector 50 is a service, the value of Γ0[eq[:eq_pc_50], endo[:μ_com_cgood]] =  Γ0[eq[:eq_pc_50], endo[:μ_com_energy]] = 0.
    #However, Γ0[eq[:eq_pc_50], endo[:μ_com_cservice]] = -m[:invkap].value[50]

    #If the test number is 69, we want to use a vector of all 200s for inv slopes of sectoral phillips curves. Otherwise, use the model values.
    invkap_value = get_setting(m, :marco_test_num) == 69 ? 200.0 * ones(length(get_setting(m,:invkap))) : get_setting(m,:invkap)       #ones(length(m[:invkap].value)) : m[:invkap].value
    #@show invkap_value[1]
#=
    #Leaving common shock for now
    Γ0[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[:μ_com]] = - invkap_value     # - m[:invkap].value


    for i in get_setting(m, :core_goods_sectors)
        Γ0[eq[Symbol("eq_pc_$(i)")], endo[:μ_com_goods]] = - invkap_value[i]    # -m[:invkap].value[i]
    end

    for i in get_setting(m, :core_service_sectors)
        Γ0[eq[Symbol("eq_pc_$(i)")], endo[:μ_com_services]] = - invkap_value[i]    #-m[:invkap].value[i]
    end

    for i in get_setting(m, :energy_sectors)
        Γ0[eq[Symbol("eq_pc_$(i)")], endo[:μ_com_energy]] = -invkap_value[i]  #-m[:invkap].value[i]
    end
    =#
    #Add sectoral markups to phillips curve
    for sect in collect(keys(get_setting(m, :subgroup_to_sector)))
        #Set up AR(1) process for this sector
        Γ0[eq[Symbol("eq_μ_$(sect)")], endo[Symbol("μ_$(sect)")]] = 1.
        Γ1[eq[Symbol("eq_μ_$(sect)")], endo[Symbol("μ_$(sect)")]] = m[Symbol("ρ_μ_$(sect)")]
        Ψ[eq[Symbol("eq_μ_$(sect)")], exo[Symbol("μ_$(sect)_sh")]] = 1.

        #Include in Phillips Curve
        for i in get_setting(m, :subgroup_to_sector)[sect]
            Γ0[eq[Symbol("eq_pc_$(i)")], endo[Symbol("μ_$(sect)")]] = -invkap_value[i]


        end
    end


    #=
    Γ0[eq[:eq_μ_com_goods], endo[:μ_com_goods]] = 1.
    Γ1[eq[:eq_μ_com_goods], endo[:μ_com_goods]] = m[:ρ_μ_goods].value[1]
    Ψ[eq[:eq_μ_com_goods], exo[:μ_com_goods_sh]] = 1.

    Γ0[eq[:eq_μ_com_services], endo[:μ_com_services]] = 1.
    Γ1[eq[:eq_μ_com_services], endo[:μ_com_services]] = m[:ρ_μ_services].value[1]
    Ψ[eq[:eq_μ_com_services], exo[:μ_com_services_sh]] = 1.

    Γ0[eq[:eq_μ_com_energy], endo[:μ_com_energy]] = 1.
    Γ1[eq[:eq_μ_com_energy], endo[:μ_com_energy]] = m[:ρ_μ_energy].value[1]
    Ψ[eq[:eq_μ_com_energy], exo[:μ_com_energy_sh]] = 1.

    #Common markup shock process
    Γ0[eq[:eq_μ_com], endo[:μ_com]] = 1.
    Γ1[eq[:eq_μ_com], endo[:μ_com]] = m[:ρ_μ_trend].value[1]
    Ψ[eq[:eq_μ_com], exo[:μ_com_sh]] = 1.
=#








    # IID MKUP PROCESS (for clarity)
    #Γ0[eq[Symbol("eq_mkup_iid_1")]:eq[Symbol("eq_mkup_iid_$n")], endo[Symbol("mkup_iid_1")]:endo[Symbol("mkup_iid_$n")]] = eye(n)
    #Ψ[eq[Symbol("eq_mkup_iid_1")]:eq[Symbol("eq_mkup_iid_$(n)")], exo[:μ_iid_1_sh]: exo[Symbol("μ_iid_$(n)_sh")]] =  eye(n) #Was negative BP


    # s recursion #Dyanmics of relative prices EQ
    Γ0[eq[Symbol("eq_srec_1")]:eq[Symbol("eq_srec_$n")], endo[:s_1]:endo[Symbol("s_$n")]] = eye(n)
    Γ1[eq[Symbol("eq_srec_1")]:eq[Symbol("eq_srec_$n")], endo[:s_1]:endo[Symbol("s_$n")]] = eye(n)
    Γ0[eq[Symbol("eq_srec_1")]:eq[Symbol("eq_srec_$n")], endo[Symbol("π_1")]:endo[Symbol("π_$n")]] = - eye(n)
    Γ0[eq[Symbol("eq_srec_1")]:eq[Symbol("eq_srec_$n")], endo[:πc_t]] = ones(n,1)


    # cpi definition #Definition of CPI
    Γ0[eq[:eq_cpi],endo[Symbol("π_1")]:endo[Symbol("π_$n")]] = get_setting(m, :gam)'    #m[:gam].value'
    Γ0[eq[:eq_cpi],endo[:πc_t]] = -1.


    #Keshav's CPI
    Γ0[eq[:eq_Kcpi],endo[Symbol("π_1")]:endo[Symbol("π_$n")]] = get_setting(m, :Kgam)'   #m[:Kgam].value'  #Replace with Keshav's gamma.
    Γ0[eq[:eq_Kcpi],endo[:πKc_t]] = -1.



    # Old implementation of N-WPC
#=
    # Nominal wage phillips curve: -lw_t+1 + bet*(1/kap)*piw_t+1 = (1/kap)*piw_t - c_t
    Γ0[eq[:eq_wpc],endo[:πw_t]] = m[:invkapw]
    Γ0[eq[:eq_wpc],endo[:c_t]]   = -1.
    Γ0[eq[:eq_wpc],endo[:w_t]]  = 1.
    Γ0[eq[:eq_wpc],endo[:Eπw_t]] = - m[:bet]*m[:invkapw]
=#

    #NOTE: To replicate kanzig IRFs, set habit formation to 0 (m[:h])
    #New implementation of Wage phillips curve
    Γ0[eq[:eq_wpc],endo[:Eπw_t]] = - m[:bet]*m[:invkapw]
    Γ0[eq[:eq_wpc],endo[:πw_t]] = m[:invkapw]
    Γ0[eq[:eq_wpc],endo[:c_t]]  = - 1.
    Γ0[eq[:eq_wpc],endo[:w_t]]  = 1.
    Γ1[eq[:eq_wpc], endo[:c_t]] = - m[:h] * exp(-m[:γ])
    Γ0[eq[:eq_wpc], endo[:μw]] = - m[:invkapw] #1.

    #AR(1) wage mark up shock
    Γ0[eq[:eq_μw], endo[:μw]] = 1.
    Γ1[eq[:eq_μw], endo[:μw]] = m[:ρ_μw]
    Ψ[eq[:eq_μw], exo[:μw_sh]] = 1.


    # real wage recursion lw_t+1 = lw_t + piw_t - pic_t
    Γ0[eq[:eq_wrec],endo[:w_t]] = 1.
    Γ0[eq[:eq_wrec],endo[:πc_t]] = 1.
    Γ0[eq[:eq_wrec],endo[:πw_t]] = -1.
    Γ1[eq[:eq_wrec],endo[:w_t]] = 1.


    # euler eq: c_t = c_{t+1} - (li_{t+1} - pic_{t+1})
#=
    Γ0[eq[:eq_euler],endo[:c_t]]   = 1.
    Γ0[eq[:eq_euler],endo[:r_t]]  = 1.
    Γ0[eq[:eq_euler],endo[:Eπc_t]]  = - 1.
    Γ0[eq[:eq_euler],endo[:Ec_t]]  = - 1.
=#

    #New Euler equation
    Γ0[eq[:eq_euler], endo[:c_t]] = 1. #c_t
    Γ0[eq[:eq_euler], endo[:r_t]] =  (1. - m[:h] * exp(-m[:γ]))/(m[:σ_c]* (1 +  m[:h] * exp(-m[:γ]))) #R_t
Γ0[eq[:eq_euler], endo[:Eπc_t]] = - (1. - m[:h] * exp(-m[:γ]))/(m[:σ_c]* (1 +  m[:h] * exp(-m[:γ]))) #Et[π_{t+1}]
Γ1[eq[:eq_euler], endo[:c_t]] = (m[:h] * exp(-m[:γ]))/(1 +  m[:h] * exp(-m[:γ])) #c_{t-1}
Γ0[eq[:eq_euler], endo[:Ec_t]] = -1. / (1. +  m[:h] * exp(-m[:γ])) #Et[c_{t+1}]
Γ0[eq[:eq_euler], endo[:b_t]] = -1. #Might need for this to be +1


    # monetary policy li_{t+1} = rho_i*li_t + (1-rho_i)*(mp_cpi_infl*pic_t +
    #                                                   mp_cons*c_t + mp_cstar*cstar_t)
# NOTE: careful about annualized vs monthly values when calibrating!

#Not convinced we need the following anymore, but leaving for now:
#numer = -m[:gam].value' * ((eye(n)-inpshare)\m[:taxshare].value)
#denom = m[:gam].value' * ((eye(n)-inpshare)\m[:taxshare].value)

numer = -get_setting(m,:gam)' * ((eye(n)-inpshare)\get_setting(m,:taxshare))
denom = get_setting(m, :gam)' * ((eye(n)-inpshare)\get_setting(m,:taxshare))
dcstar_dτ = numer/denom

#=
# Old policy rule:
Γ0[eq[:eq_monpol],endo[:r_t]] = 1. #R_t
Γ1[eq[:eq_monpol],endo[:r_t]] = m[:ρ_i] #ρ_R R_{t-1}
Γ0[eq[:eq_monpol],endo[:πc_t]] = -(1. -m[:ρ_i])*m[:mp_cpi_infl]
Γ0[eq[:eq_monpol],endo[:c_t]]  = - (1. -m[:ρ_i])*m[:mp_cons]
Γ0[eq[:eq_monpol],endo[:τ]]  = - (1. -m[:ρ_i])*m[:mp_cstar] * dcstar_dτ
=#



#New monetary policy rule:
Γ0[eq[:eq_monpol],endo[:r_t]] = 1.
Γ1[eq[:eq_monpol],endo[:r_t]] = m[:ρ_i]
Γ0[eq[:eq_monpol],endo[:πc_t]] = - (1. -m[:ρ_i])*m[:mp_cpi_infl]
Γ0[eq[:eq_monpol],endo[:πstar]] = (1. -m[:ρ_i])*m[:mp_cpi_infl]
Γ0[eq[:eq_monpol],endo[:c_t]]  = - (1. -m[:ρ_i])*m[:mp_cons] + m[:mp_habit]
Γ1[eq[:eq_monpol],endo[:c_t]]  = -m[:mp_habit]
Γ0[eq[:eq_monpol], endo[:mp_t]] = -1.


#IID M.P. shock
Γ0[eq[:eq_mp_t], endo[:mp_t]] = 1.
Ψ[eq[:eq_mp_t], exo[:mp_sh]] = 1. #Was +1 BP

#Time varying inflation target with iid shock
Γ0[eq[:eq_πstar], endo[:πstar]] = 1.
Γ1[eq[:eq_πstar], endo[:πstar]] = m[:ρ_πstar]
Ψ[eq[:eq_πstar], exo[:πstar_sh]] = 1.


## Lag Definitions, Expectational Errors ##


##Expectation Errors ##
#Consumption
Γ0[eq[:eq_Ect], endo[:c_t]] = 1.
Γ1[eq[:eq_Ect], endo[:Ec_t]] = 1.
Π[eq[:eq_Ect], exp_sh[:Ec_sh]] = 1.

#CPI Inflation
Γ0[eq[:eq_Eπct], endo[:πc_t]] = 1.
Γ1[eq[:eq_Eπct], endo[:Eπc_t]] = 1.
Π[eq[:eq_Eπct], exp_sh[:Eπc_sh]] = 1.

#Wage inflation
Γ0[eq[:eq_Eπwt], endo[:πw_t]] = 1.
Γ1[eq[:eq_Eπwt], endo[:Eπw_t]] = 1.
Π[eq[:eq_Eπwt], exp_sh[:Eπw_sh]] = 1.

#Sectoral Inflation
Γ0[eq[Symbol("eq_Eπ_1")]:eq[Symbol("eq_Eπ_$n")], endo[Symbol("π_1")]:endo[Symbol("π_$n")]]  = eye(n)
Γ1[eq[Symbol("eq_Eπ_1")]:eq[Symbol("eq_Eπ_$n")], endo[Symbol("Eπ_1")]:endo[Symbol("Eπ_$n")]]  = eye(n)
Π[eq[Symbol("eq_Eπ_1")]:eq[Symbol("eq_Eπ_$n")], exp_sh[Symbol("Eπ_1_sh")]: exp_sh[Symbol("Eπ_$(n)_sh")]] = eye(n)

## Exogenous Shocks ##

# tau recursion -- AR(2) process for τ shock
Γ0[eq[:eq_τ],endo[:τ]]  = 1.
Γ1[eq[:eq_τ],endo[:τ]]  = m[:ρ_τ]
Γ1[eq[:eq_τ],endo[:lτ]] = m[:ρ_τ2]
Ψ[eq[:eq_τ],exo[:τ_sh]] = 1.

#Define lag τ for AR(2)

Γ0[eq[:eq_lτdef],endo[:lτ]] = 1.
Γ1[eq[:eq_lτdef],endo[:τ]]  = 1.


#Discount rate shock
Γ0[eq[:eq_b_t], endo[:b_t]] = 1.
Γ1[eq[:eq_b_t], endo[:b_t]] = m[:ρ_b_t]
Ψ[eq[:eq_b_t], exo[:b_sh]] = 1.


#Common TFP shock -- within PC
Γ0[eq[:eq_a_t], endo[:a_t]] = 1.

#Print new TFP Persistence
#@show m[:ρ_a_t].value

Γ1[eq[:eq_a_t], endo[:a_t]] = m[:ρ_a_t]
Ψ[eq[:eq_a_t], exo[:a_sh]] = 1.


#Markup stochastic trend:
#μ^i_t = ρ_̅μ^i ̅μ^i_{t-1} + σ_̅μ^i ε_t^{̅μ^i}

#=
Γ0[eq[:eq_mkup_trend_1]:eq[Symbol("eq_mkup_trend_$(n)")], endo[:mkup_trend_1]: endo[Symbol("mkup_trend_$(n)")]] = eye(n)
Γ1[eq[:eq_mkup_trend_1]:eq[Symbol("eq_mkup_trend_$(n)")], endo[:mkup_trend_1]: endo[Symbol("mkup_trend_$(n)")]] = diagm(m[:ρ_μ_trend].value)

#Print to make sure we are using the correct markup persistence
@show m[:ρ_μ_trend].value

Ψ[eq[:eq_mkup_trend_1]:eq[Symbol("eq_mkup_trend_$(n)")], exo[:μ_trend_1_sh]:exo[Symbol("μ_trend_$(n)_sh")]] = eye(n)
=#











    #= enforce normalization that gam'*s = 0 to remove unit eigenvalue
    % s_t     = [s^1_t;s^2_t;...;s^n_t] is a nx1 vector
    % snorm_t = [-(gam^2/gam^1)*s^2_t - ... -(gam^n/gam^1)*s^n_t;
    %            s^2_t;
    %            ...;
    %            s^n_t] is a (n-1)x1 vector
    % let's make a n x (n-1) matrix C such that s_t = C*snorm_t
    =#


    C = eye(n)
C = C[:,2:n]
C[1,:] = -get_setting(m,:gam)[2:end]/get_setting(m,:gam)[1]
    #C[1,:] = -m[:gam].value[2:end]/m[:gam].value[1]


    #norm_mat = blkdiag(C,eye(N-n)) #blkdiag doesn't exist in Julia
    #Rough Julia translation. Create square 0s matrix,
    norm_mat = zeros(size(C,1) + (N-n), size(C,2) + (N-n))

    norm_mat[1:size(C,1), 1:size(C,2)] = C
norm_mat[size(C,1)+1:end, size(C,2)+1:end] = eye(N-n)
#@show size(Γ0), size(Γ1)
    Γ0_norm = Γ0[vcat(1:n,n+2:N),:]*norm_mat # remove redundant equation
    Γ1_norm = Γ1[vcat(1:n,n+2:N),:]*norm_mat

    Ψ_norm = Ψ[vcat(1:n, n+2:N), :]
    Π_norm = Π[vcat(1:n, n+2:N), :]


m <= Setting(:n_back_states, get_setting(m, :n_back_states) - 1)
    m <= Setting(:n_model_states,
                 get_setting(m, :n_back_states) + get_setting(m, :n_jump_states) + get_setting(m, :n_exo_states))

      #=
      ls_norm  = 1:n-1; % indices AFTER NORMALIZATION
      lw_norm  = lw-1;
      tau_norm = tau-1;
      li_norm = li-1;
      ltau_norm = ltau-1;
      =#

#Reindex after normalization:
#del_ind_eq = m.equilibrium_conditions[:eq_srec_1]
del_ind_st = m.endogenous_states[:s_1]
    delete!(m.endogenous_states, :s_1) #I want to delete the first lagged state

    for (k,v) in m.endogenous_states
        if v > del_ind_st
            m.endogenous_states[k] = v-1
        end
    end


    for (k,v) in m.endogenous_states_augmented
        if v > del_ind_st
            m.endogenous_states_augmented[k] = v-1
        end
    end

    del_ind = m.equilibrium_conditions[:eq_srec_1]
    delete!(m.equilibrium_conditions, :eq_srec_1) #Remove the redundant equation, now ls is only to n-1

    for (k,v) in m.equilibrium_conditions
        if v > del_ind
            m.equilibrium_conditions[k] = v-1
        end
    end


    return Γ0_norm, Γ1_norm, Const1, Ψ_norm, Π_norm, C, norm_mat

end
