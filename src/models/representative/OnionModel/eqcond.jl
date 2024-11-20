#Define eye function for MATLAB compatibility
@inline eye(n::Integer) = Matrix{Float64}(I,n,n)



#=endogenous_states = [[Symbol("ls_$i") for i in 1:n];
                     [:ls, :lw, :li, :lτ, :τ];
                     [Symbol("π_$i") for i in 1:n];
                      [:πc, :c, :πw, :pc]]
equilibrium_conditions = [[Symbol("eq_pc_$i") for i in 1:n];
                          [Symbol("eq_srec_$i"] for i in 1:n];
                          [:eq_cpi, :eq_wpc, :eq_wrec, :eq_monpol, :eq_τrec, :eq_euler, :eq_lτdef]]

for (i,k) in enumerate(endogenous_states)
    m.endogenous_states[k] = i
end
for (i,k) in enumerate(equilibrium_conditions)
    m.equilibrium_conditions[k] = i
end=#


function eqcond(m::OnionModel) #m::OnionModel
    ## linearized equations
    # 2*n+7 equations, 2*n+7 variables
    # A x_{t+1} = B x_t

    #Now adding back shocks. Just 1 for now
    eq     = m.equilibrium_conditions
    endo   = m.endogenous_states
    exo    = m.exogenous_shocks
    exp_sh = m.expected_shocks

    N = get_setting(m, :n_model_states) #+ get_setting(m, :n_exo_states)
    n = get_setting(m, :n_sectors)

    @show N, n

    A = zeros(N,N)
    B = zeros(N,N)
    Ψ = zeros(N, length(exo))
    Π = zeros(N, length(exp_sh)) #Will be m.states_expectational
    Const1 = zeros(N-1)

    inpshare = get_setting(m, :inpshare)
#=
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

    if get_setting(m, :irf_type) == "oil"
        sec_shock = zeros(size(m[:taxshare].value))
        sec_shock[Int(m[:oil].value)] = 1.
    else
        sec_shock = -m[:taxshare].value
    end

    # Phillips curve
    B[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[Symbol("π_1")]:endo[Symbol("π_$n")]]    = diagm(m[:invkap].value)
    B[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[:τ]]  = - sec_shock #-m[:taxshare].value For oil kanzig IRFs, vector of 0 with 1 at oil index.
    A[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[:lw]] =  m[:labshare].value
    A[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[Symbol("ls_1")]:endo[Symbol("ls_$n")]]  = inpshare - eye(n)
    A[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[Symbol("π_1")]:endo[Symbol("π_$n")]]    = m[:bet]*diagm(m[:invkap].value)

     B[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], endo[Symbol("mkup_1")]:endo[Symbol("mkup_$(n)")]] = eye(n) #Add mkup process
    Π[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], exp_sh[Symbol("Eπ_1_sh")]:exp_sh[Symbol("Eπ_$(n)_sh")]] =  eye(n)
    #Ψ[eq[Symbol("eq_pc_1")]:eq[Symbol("eq_pc_$n")], exo[Symbol("μ_1_sh")]:exo[Symbol("μ_$(n)_sh")]] = eye(n)



    # s recursion
    A[eq[Symbol("eq_srec_1")]:eq[Symbol("eq_srec_$n")], endo[:ls_1]:endo[Symbol("ls_$n")]] = eye(n)
    B[eq[Symbol("eq_srec_1")]:eq[Symbol("eq_srec_$n")], endo[:ls_1]:endo[Symbol("ls_$n")]] = eye(n)

    B[eq[Symbol("eq_srec_1")]:eq[Symbol("eq_srec_$n")], endo[Symbol("π_1")]:endo[Symbol("π_$n")]] = eye(n)
    B[eq[Symbol("eq_srec_1")]:eq[Symbol("eq_srec_$n")], endo[:πc]] = -ones(n,1)


    # cpi definition
    B[eq[:eq_cpi],endo[Symbol("π_1")]:endo[Symbol("π_$n")]] = m[:gam].value'
    B[eq[:eq_cpi],endo[:πc]] = -1.
    Π[eq[:eq_cpi],exp_sh[:Eπc_sh]] = 1.


    # wage phillips curve: -lw_t+1 + bet*(1/kap)*piw_t+1 = (1/kap)*piw_t - c_t
    B[eq[:eq_wpc],endo[:πw]] = m[:invkapw]
    B[eq[:eq_wpc],endo[:c]]   = -1.
    A[eq[:eq_wpc],endo[:lw]]  = -1.
    A[eq[:eq_wpc],endo[:πw]] = m[:bet]*m[:invkapw]
    Π[eq[:eq_wpc],exp_sh[:Eπw_sh]] = 1.

    # real wage recursion lw_t+1 = lw_t + piw_t - pic_t
    B[eq[:eq_wrec],endo[:lw]] =  1.
    B[eq[:eq_wrec],endo[:πc]] = -1.
    B[eq[:eq_wrec],endo[:πw]] =  1.
    A[eq[:eq_wrec],endo[:lw]] =  1.


    # monetary policy li_{t+1} = rho_i*li_t + (1-rho_i)*(mp_cpi_infl*pic_t +
    #                                                   mp_cons*c_t + mp_cstar*cstar_t)
    # NOTE: careful about annualized vs monthly values when calibrating!

    numer = -m[:gam].value' * ((eye(n)-inpshare)\m[:taxshare].value)
    denom = m[:gam].value' * ((eye(n)-inpshare)\m[:taxshare].value)
    dcstar_dτ = numer/denom

    A[eq[:eq_monpol],endo[:li]] = 1.
    B[eq[:eq_monpol],endo[:li]] = m[:ρ_i]
    B[eq[:eq_monpol],endo[:πc]] = (1. -m[:ρ_i])*m[:mp_cpi_infl]
    B[eq[:eq_monpol],endo[:c]]  = (1. -m[:ρ_i])*m[:mp_cons]
    B[eq[:eq_monpol],endo[:τ]]  = (1. -m[:ρ_i])*m[:mp_cstar] * dcstar_dτ

    ## Exogenous Shocks ##

    # tau recursion -- AR(2) process for τ shock
    A[eq[:eq_τrec],endo[:τ]]  = 1.
    B[eq[:eq_τrec],endo[:τ]]  = m[:ρ_τ]
    B[eq[:eq_τrec],endo[:lτ]] = m[:ρ_τ2]
    Ψ[eq[:eq_τrec],exo[:τ_sh]] = 1.


    #Markup shock
    #Markup is the sum of the stochastic trends ̅μ^i_t and iid markup shocks σ_μ^i ε_t^{μ^i}
    A[eq[:eq_mkup_1]:eq[Symbol("eq_mkup_$(n)")], endo[:mkup_1]: endo[Symbol("mkup_$(n)")]] = eye(n)
    B[eq[:eq_mkup_1]:eq[Symbol("eq_mkup_$(n)")], endo[:mkup_trend_1]: endo[Symbol("mkup_trend_$(n)")]] = eye(n)
    #IID: σ_μ^i ε_t^{μ^i}
    Ψ[eq[:eq_mkup_1]:eq[Symbol("eq_mkup_$(n)")], exo[:μ_iid_1_sh]: exo[Symbol("μ_iid_$(n)_sh")]] = eye(n)

    #Stochastic trend:
#μ^i_t = ρ_̅μ^i ̅μ^i_{t-1} + σ_̅μ^i ε_t^{̅μ^i}
A[eq[:eq_mkup_trend_1]:eq[Symbol("eq_mkup_trend_$(n)")], endo[:mkup_trend_1]: endo[Symbol("mkup_trend_$(n)")]] = eye(n)
B[eq[:eq_mkup_trend_1]:eq[Symbol("eq_mkup_trend_$(n)")], endo[:mkup_trend_1]: endo[Symbol("mkup_trend_$(n)")]] = diagm(m[:ρ_μ_trend].value)
Ψ[eq[:eq_mkup_trend_1]:eq[Symbol("eq_mkup_trend_$(n)")], exo[:μ_trend_1_sh]:exo[Symbol("μ_trend_$(n)_sh")]] = eye(n)




    # euler eq: c_t = c_{t+1} - (li_{t+1} - pic_{t+1})
    A[eq[:eq_euler],endo[:c]]   = 1.
    A[eq[:eq_euler],endo[:li]]  = -1.
    A[eq[:eq_euler],endo[:πc]]  = 1.
    B[eq[:eq_euler],endo[:c]]   = 1.
    Π[eq[:eq_euler],exp_sh[:Ec_sh]] = 1.

    # lag tau definition
    A[eq[:eq_lτdef],endo[:lτ]] = 1.
    B[eq[:eq_lτdef],endo[:τ]]  = 1.


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
    C[1,:] = -m[:gam].value[2:end]/m[:gam].value[1]


    #norm_mat = blkdiag(C,eye(N-n)) #blkdiag doesn't exist in Julia
    #Rough Julia translation. Create square 0s matrix,
    norm_mat = zeros(size(C,1) + (N-n), size(C,2) + (N-n))

    norm_mat[1:size(C,1), 1:size(C,2)] = C
    norm_mat[size(C,1)+1:end, size(C,2)+1:end] = eye(N-n)
    Anorm = A[vcat(1:n,n+2:N),:]*norm_mat # remove redundant equation
    Bnorm = B[vcat(1:n,n+2:N),:]*norm_mat

    Ψ = Ψ[vcat(1:n, n+2:N), :]
    Π = Π[vcat(1:n, n+2:N), :]

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

    del_ind = m.equilibrium_conditions[:eq_srec_1]
    delete!(m.endogenous_states, :ls_1) #I want to delete the first lagged state

    for (k,v) in m.endogenous_states
        if v > del_ind
            m.endogenous_states[k] = v-1
        end
    end


    for (k,v) in m.endogenous_states_augmented
        if v > del_ind
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

    return Anorm, Bnorm, Const1, Ψ, Π, C, norm_mat

end
