function jacobian(m::HetDSGEGovDebt, x::Vector{Float64})

    # Load in parameters, steady-state parameters, and grids
    α     = m[:α].value
    δ     = m[:δ].value
    γ     = m[:γ].scaledvalue
    η     = m[:η].value
    ρ_B    = m[:ρ_b].value
    ρ_G    = m[:ρ_g].value
    ρ_z    = m[:ρ_z].value
    ρ_μ    = m[:ρ_μ].value
    ρ_lamw = m[:ρ_λ_w].value
    ρ_lamf = m[:ρ_λ_f].value
    ρ_mon  = m[:ρ_rm].value
    spp   = m[:spp].value
    ϕh    = m[:ϕh].value
    ρ_R    = m[:ρR].value
    ψπ    = m[:ψπ].value
    ψy    = m[:ψy].value
    κ_p  = m[:κ_p].value
    κ_w  = m[:κ_w].value

    Tg = m[:Tg].value
    δb = m[:δb].value
    bg = m[:bg].value

    π_star = m[:π_star].value
    β = m[:βstar].value

    # Temp
    Φ_w = 1.
    ϕ = 1.
    τw = 1.
    π_star_w = 1.
    Φ_p = 1.
    τF = 1.
    iii = 1.

    xgrid = m.grids[:xgrid].points
    xwts  = m.grids[:xgrid].weights
    sgrid = m.grids[:sgrid].points
    swts  = m.grids[:sgrid].weights
    fgrid = m.grids[:fgrid]
    xswts = kron(swts,xwts)

    zlo = m[:zlo].value
    zhi = m[:zhi].value

    nx = DSGE.get_setting(m, :nx)
    ns = DSGE.get_setting(m, :ns)
    nxns = nx*ns

    @inline qfunction(x) = DSGE.mollifier_hetdsgegovdebt(x, zhi, zlo)

    @inline function sfunc(x::S) where S<:Real
        return (spp/2)*x^2
    end
    @inline function sfunc_prime(x::S) where S<:Real
        return spp*x
    end

    @inline function euler_equation_hetdsgegovdebt_eq(#nx::Int, ns::Int,
                                                      # qfunction::Function,
                                                      # xgrid::Vector{Float64}, sgrid::Vector{Float64},
                                                      # fgrid::Matrix{Float64},
                                                      # xswts::Vector{Float64},
                                                      R_t::S, z′_t::S, β::Float64,
                                                      η::Float64, l_t::Vector{S}, l′_t::Vector{S},
                                                      t′_t::S, ω′_t::S, L′_t::S, cfunc::Vector{S}, cfunc′::Vector{S}) where S<:Real
        nxns = nx*ns

        ret_ell = Vector{Number}(undef, nxns)

        # Loop over s
        for iss=1:ns
            # Loop over i
            for ia=1:nx
                # Index for outcome
                i  = nx*(iss-1)+ia
                ell_euler = 0.
                # Sum over s'
                for isp=1:ns
                    # Sum over j
                    for iap=1:nx
                        # Index for input
                        ip = nx*(isp-1)+iap
                        # ee = (aⱼ - (1+Rₜ)*e⁻ᶻ'(aⱼ - cₜ(aⱼ, s)) - T')/(w's'H')
                        ee = (xgrid[iap] - (1+R_t)*exp(-z′_t)*(xgrid[iap] - cfunc[ip]) - t′_t)/(ω′_t*L′_t*sgrid[isp])
                        # ι*∑∑(e⁻ᶻ'/c'(aⱼ,s'))*g(ee)*p(s'|s)/(w'*s'*H')
                        ell_euler += xswts[ip]*(exp(-z′_t)/cfunc'[ip]) * qfunction(ee)*(fgrid[iss,isp])/(ω′_t*L′_t*sgrid[isp])
                    end
                end
                ret_ell[i] = ell_euler
            end
        end
        return ret_ell
    end

    @inline function kolmogorov_fwd_hetdsgegovdebt_eq(#nx::Int, ns::Int,
                                                      #  qfunction::Function,
                                                      #  xgrid::Vector{Float64}, sgrid::Vector{Float64},
                                                      #  fgrid::Matrix{Float64},
                                                      #  xswts::Vector{Float64},
                                                      R_t::S, z′_t::S,
                                                      η::Float64, kf_t::Vector{S}, l_t::Vector{S}, t′_t::S, w′_t::S, H′_t::Real, cfunc::Vector{S}) where S<:Real
        nxns = nx*ns
        ret_kol = Vector{Real}(undef, nxns)

        # Loop over j
        for isp=1:ns
            # Loop over s'
            for iap=1:nx
                m_kolmogorov = 0.
                # Index for outcome
                ip  = nx*(isp-1)+iap
                # Sum over i
                for ia=1:nx
                    # Sum over s
                    for iss=1:ns
                        # Index for input
                        i = nx*(iss-1)+ia
                        # (aⱼ - (1+rₜ)e⁻ᶻ'(aᵢ - cₜ(aᵢ, s) - T')/ w's'H'
                        ee = (xgrid[iap] - (1+R_t)*(exp(-z′_t))*(xgrid[ia] - cfunc[ip]) - t′_t)/(w′_t*H′_t*sgrid[isp])
                        # mₜ(aᵢ, s) * g(ee) * p(s'|s)/w's'H' # f[iss, isp] = p(isp|iss)
                        m_kolmogorov += xswts[i]*kf_t[i]* qfunction(ee)*(fgrid[iss,isp])/(w′_t*H′_t*sgrid[isp])
                    end
                end
                # Set index j (outcome)
                ret_kol[ip] = m_kolmogorov
            end
        end
        return ret_kol
    end

    function F(x::Vector{S}) where {S<:Real}
        endo = DSGE.augment_model_states(m.endogenous_states_original, DSGE.n_model_states_original(m))
        for (key,value) in endo
            @eval if length($(x[value])) == 1
                @eval (($(key)) = ($(x[value][1])))
            else
                @eval (($(key)) = ($(x[value])))
            end
        end

        cfunc′ = min.(1 ./ l′_t, repeat(xgrid, ns).+η)
        cfunc = min.(1 ./ l_t, repeat(xgrid, ns).+η)

        ell_euler_eq = euler_equation_hetdsgegovdebt_eq(#nx, ns, #qfunction_hetdsgegovdebt,
                                                        #     xgrid, sgrid, fgrid, xswts,
                                                             R′_t, z′_t, β, η, l_t, l′_t,
                                                             t′_t, w′_t, L′_t, cfunc, cfunc′)
        # KF Equation
        kf_eq = kolmogorov_fwd_hetdsgegovdebt_eq(#nx, ns, #qfunction_hetdsgegovdebt,
                                                  #    xgrid, sgrid, fgrid, xswts,
                                                      R′_t, z′_t, η, kf_t, l_t,
                                                      t′_t, w′_t, L′_t, cfunc)

        eq_euler = l_t - β*b′_t*(1+R_t)*ell_euler_eq
        eq_kolmogorov_fwd = kf′_t - kf_eq
        eq_agg_consumption = C_t - (xswts .* kf_t)'cfunc
        eq_lambda = margutil_t - (xswts .* kf_t)'*(1 ./ cfunc)
        eq_transfers = capreturn_t*k_t*exp(-z_t) - I_t + (1-mc_t)*y_t - tg_t - t_t
        # note this beta is listed as βtil which differs from ̱β but i think it drops out when take derivative so doesnt matter
        eq_investment = margutil_t*Q_t*μ_t*(1-sfunc(I_t/I_t1*exp(-z_t)))+β*margutil′_t*exp(-z′_t)*Q′_t*μ′_t*sfunc_prime(I′_t/I_t*exp(z′_t))*(I′_t/I_t*exp(z′_t))^2 - margutil_t*(1+Q_t*μ_t*sfunc_prime(I_t/I_t1*exp(z_t))*I_t/I_t1*exp(z_t))
        eq_tobin_q = β*margutil′_t*exp(-z′_t)/margutil_t*(capreturn′_t + Q′_t*(1-δ))
        eq_capital_accumulation = (1-δ)*k_t*exp(-z_t) + μ_t*(1-sfunc(I_t/I_t1 * exp(z_t)))*I_t - k′_t
        eq_wage_phillips =  1/(λ_w_t*Φ_w) * ((1+λ_w_t)*ϕ*L_t^ϕh - (1-τw)*margutil_t*w_t) + β*b′_t*(π_w′_t/π_star_w - 1)*(π_w′_t/π_star_w)*(L′_t/L_t) - (π_w_t/π_star_w - 1)*(π_w_t/π_star_w)  #figure out how to d\eal with kappa_w and kappa_p coefficients
        eq_price_phillips = -1/(λ_w_t*Φ_p)*((1+τF) - (1+λ_f_t)*mc_t) + (1/(1 + R_t))*(π′_t/π_t -1)*(π′_t/π_star)*y′_t*exp(z′_t)*y_t - (π_t/π_star -1)*(π_t/π_star)# see above
        eq_marginal_cost = ((1-α)^(1-α)*α^α)^(-1)*w_t^(1-α)*capreturn_t^α - mc_t
        eq_gdp = exp(-α*z_t)*k_t^α*L_t^(1-α) - y_t
        eq_optimal_kl = α/(1-α)*L_t/k_t*exp(-z_t) - capreturn_t/w_t
        eq_taylor = ((1+i_t)/(1+iii))^ρ_R*((π/π_star)^ψπ * (y_t/y_t1*exp(-γ))^ψy)^(1-ρ_R)*exp(rm′_t) - (1+i_t)/(1+iii)
        eq_fisher = (1 + i_t)/π′_t - 1 - R_t
        eq_nominal_wage_inflation = π_t*exp(z_t)*w_t/w_t1
        eq_fiscal_rule = δb*(bg_t*exp(-z_t) + (1-1/g_t)*y_t - bg/(1+R_t)) + (1-δ)*Tg - tg_t
        eq_g_budget_constraint = bg_t*exp(-z_t) + g_t - tg_t - bg′_t/(1+R_t)
        eq_resource_constraint = y_t/g_t - C_t - I_t
        eq_LI = i′_t1 - i_t
        eq_LY = y′_t1 - y_t
        eq_LW = w′_t1 - w_t
        eq_LX = I′_t1 - I_t
        eq_b = b′_t - ρ_B*b_t #plus shock for this and all below but doesnt matter because taking derivative
        eq_g = g′_t - ρ_G*g_t
        eq_z = z′_t - ρ_z*z_t
        eq_μ = μ′_t - ρ_μ*μ_t
        eq_λ_w = λ_w′_t - ρ_lamw*λ_w_t
        eq_λ_f = λ_f′_t - ρ_lamf*λ_f_t
        eq_rm = rm′_t - ρ_mon*rm_t

        return [eq_euler; eq_kolmogorov_fwd; eq_agg_consumption; eq_lambda; eq_transfers;
                eq_investment; eq_tobin_q; eq_capital_accumulation; eq_wage_phillips;
                eq_price_phillips; eq_marginal_cost; eq_gdp; eq_optimal_kl; eq_taylor; eq_fisher;
                eq_nominal_wage_inflation; eq_fiscal_rule; eq_g_budget_constraint;
                eq_resource_constraint;
                eq_LI; eq_LY; eq_LW; eq_LX;
                eq_b; eq_g; eq_z; eq_μ; eq_λ_w; eq_λ_f; eq_rm]

    end

    JJ = ForwardDiff.jacobian(F, x)
    return JJ
end

#=function consumption_eq(ell, η, xswts, μ, ns, nx)
    con_eq = 0.
    for iss=1:ns
        for ia=1:nx
            i  = nx*(iss-1)+ia
            cfunc = min(1/ell[ia], xswts[i])
            con_eq += xswts[i]*cfunc*μ[ia]
        end
    end
    return con_eq
end

function lambda_eq(ell, η, xswts, μ, ns, nx)
    lam_eq = 0.
    for iss=1:ns
        for ia=1:nx
            i  = nx*(iss-1)+ia
            cfunc = min(1/ell[ia], xswts[i])
            lam_eq += xswts[i]*μ[ia]/cfunc
        end
    end
    return lam_eq
end=#


function normalize(m::HetDSGEGovDebt, JJ::Matrix{Float64})

    Qx, Qy, Qleft, Qright  = compose_normalization_matrices(m)

    m <= Setting(:n_predetermined_variables, size(Qx, 1))
    m <= Setting(:Qx, Qx)
    m <= Setting(:Qy, Qy)
    #Qleft = get_setting(m, :Qleft) #m[:Qleft].value
    #Qright = get_setting(m, :Qright) #m[:Qright].value

	Jac1 = Qleft*JJ*Qright

    return Jac1
end

# use proxy distribuitions method - now `averaging' so that Qx'*Qx should not `add mass'
function avg_prox(n::Int, binsize::Int = 3, minn_noag::Int = 0)
    minn_noag_use = min(minn_noag,n)
    n_gps = Int(floor((n-minn_noag_use)/binsize)) # number of groups
    n_noag = n - binsize*n_gps
    Prox = cat(eye(n_noag), kron(eye(n_gps),ones(1,binsize)./sqrt(binsize)), dims = [1 2])
    n_new = n_noag + n_gps
    return (Prox, n_new, n_noag, n_gps)
end
# note: setting minn_noag = 0 seems to work the best

function make_S(n::Int, n_noag::Int = 0, binsize::Int = 1)
    P1 = ones(n,1)*sqrt(binsize)
    P1[1:n_noag] .= 1.
    Ptemp = eye(n)
    Ptemp = Ptemp[:,2:end]
    P2 = Ptemp
    P = [P1 P2]
    (QQQ,Rjunk)=qr(P)
    S         = QQQ[:,2:end]'
    return S
end

function compose_normalization_matrices(m::HetDSGEGovDebt)
    if get_setting(m, :poor_man_reduc)
        nx = get_setting(m, :nx1_state) #:nx)
        ns = get_setting(m, :ns)
        n = nx*ns #get_setting(m, :nx1) + get_setting(m, :nx2) #
        nscalars = get_setting(m, :nscalars)
        nyscalars = get_setting(m, :nyscalars)
        nxscalars = get_setting(m, :nxscalars)
        mindens = get_setting(m, :mindens)
        μ = m[:μstar].value

        nx1 = maximum(findall(μ[1:nx].>mindens))
        m <= Setting(:nx1_state, nx1)
        m <= Setting(:nx1_jump, nx1)

        setup_indices!(m)
        init_states_and_jumps!(m, get_setting(m, :states), get_setting(m, :jumps))
        normalize_model_state_indices!(m)
        endogenous_states_augmented = [:C_t1]
        for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i + first(m.endogenous_states[get_setting(m, :jumps)[end]]) end #first(collect(values(m.endogenous_states))[end]) end

        m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states) +
                     length(m.endogenous_states_augmented))

        Reduc = eye(n)
        Reduc=Reduc[[1:nx1;nx+1:n],:]

        minn_noag = 0
        binsize = get_setting(m, :binsize) # this is the maximum binsize which seemed to leave the IRFs unchanged, you can experiment with this
        # setting binsize = 1 should return what we had before
        (ProxL, n_newL, n_noagL, n_gpsL) = avg_prox(nx1, binsize, minn_noag)
        (ProxH, n_newH, n_noagH, n_gpsH) = avg_prox(nx, binsize,minn_noag)
        m <= Setting(:nx1_state, n_newL)
        m <= Setting(:nx2_state, n_newH)
        if get_setting(m, :reduce_ell)
            m <= Setting(:nx1_jump, n_newL)
            m <= Setting(:nx2_jump, n_newH)
        end
        setup_indices!(m)
        init_states_and_jumps!(m, get_setting(m, :states), get_setting(m, :jumps))
        normalize_model_state_indices!(m)
        endogenous_states_augmented = [:C_t1]
        for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i + first(m.endogenous_states[get_setting(m, :jumps)[end]]) end #first(collect(values(m.endogenous_states))[end]) end

        m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states) +
                     length(m.endogenous_states_augmented))


        Prox = cat(ProxL, ProxH, dims = [1 2])
        SL = make_S(n_newL, n_noagL, binsize)
        SH = make_S(n_newH, n_noagH, binsize)
        S = cat(SL, SH, dims = [1 2])

        if get_setting(m, :reduce_ell)
            Qleft     = cat(Prox*Reduc,S*Prox*Reduc,eye(nscalars), dims = [1 2])
            Qx        = cat(S*Prox*Reduc,eye(nxscalars), dims = [1 2])
            Qy        = cat(Prox*Reduc,eye(nyscalars), dims = [1 2])
        else
            Qleft     = cat(Reduc,S*Prox*Reduc,eye(nscalars), dims = [1 2])
            Qx        = cat(S*Prox*Reduc,eye(nxscalars), dims = [1 2])
            Qy        = cat(Reduc,eye(nyscalars), dims = [1 2])
        end

        Qright    = cat(Qx',Qy',Qx',Qy', dims = [1,2])

        return Qx, Qy, Qleft, Qright

    else
        nx = get_setting(m, :nx)
        ns = get_setting(m, :ns)
        nscalars = get_setting(m, :nscalars)
        nyscalars = get_setting(m, :nyscalars)
        nxscalars = get_setting(m, :nxscalars)

        # Create PPP matrix
        P1 = kron(Matrix{Float64}(I, ns,ns),ones(nx,1))
        Ptemp = Matrix{Float64}(I, nx, nx)
        Ptemp = Ptemp[:, 2:end]
        P2 = kron(Matrix{Float64}(I, ns, ns), Ptemp)
        P  = hcat(P1, P2)

        Q,R = qr(P)
        Q = Array(Q)
        S         = Q[:, ns+1:end]'

        nxns = nx*ns

        Qleft     = cat(Matrix{Float64}(I, nxns, nxns),S,Matrix{Float64}(I, nscalars, nscalars), dims = [1 2])
        Qx        = cat(S,Matrix{Float64}(I, nxscalars, nxscalars), dims = [1 2])
        Qy        = cat(Matrix{Float64}(I, nxns, nxns),Matrix{Float64}(I, nyscalars, nyscalars), dims = [1 2])
        Qright    = cat(Qx',Qy',Qx',Qy', dims = [1,2])

        return Qx, Qy, Qleft, Qright

    end
end

function truncate_distribution!(m::HetDSGEGovDebt)
    mindens = get_setting(m, :mindens)
    trunc_distr = get_setting(m, :trunc_distr)
    rescale_weights = get_setting(m, :rescale_weights)

    nx = get_setting(m, :nx)
    μ = m[:μstar].value
    ell = m[:lstar].value
    c = m[:cstar].value
    xgrid = m.grids[:xgrid].points
    #xlo = get_setting(m, :xlo)
    swts::Vector{Float64}  = m.grids[:sgrid].weights

    if trunc_distr
        oldnx = nx
        nx = maximum(findall(μ[1:nx]+μ[nx+1:2*nx] .> mindens)) # used to be 1e-8
        m[:μstar] = μ[[1:nx;oldnx+1:oldnx+nx]]
        m[:lstar] = ell[[1:nx;oldnx+1:oldnx+nx]]
        m[:cstar] = c[[1:nx;oldnx+1:oldnx+nx]]
        if rescale_weights
            xhi = xgrid[nx]
            xlo = xgrid[1]
            xscale = xhi-xlo
        end
        m <= Setting(:nx, nx)
        m <= Setting(:xhi, xhi)
        m <= Setting(:xscale, xscale)
        m.grids[:xgrid] = Grid(uniform_quadrature(xscale), xlo, xhi, nx, scale = xscale)
        m.grids[:weights_total] = kron(swts, m.grids[:xgrid].weights)
        nxns = nx*get_setting(m, :ns)
        m <= Setting(:n, nxns)
        m <= Setting(:nx, nx)

        setup_indices!(m)

        init_states_and_jumps!(m, get_setting(m, :states), get_setting(m, :jumps))

        normalize_model_state_indices!(m)

        endogenous_states_augmented = [:C_t1]
        for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i + first(m.endogenous_states[get_setting(m, :jumps)[end]]) end #first(collect(values(m.endogenous_states))[end]) end

        m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states) +
                     length(m.endogenous_states_augmented))

    end
end
