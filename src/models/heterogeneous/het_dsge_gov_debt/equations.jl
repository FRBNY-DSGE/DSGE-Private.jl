function F(m::HetDSGEGovDebt)
    reset_grids!(m)
    #truncate_distribution!(m)

    # Load in endogenous state and eq cond indices
    endo = augment_model_states(m.endogenous_states_original,#m.endogenous_states_unnormalized,
                         n_model_states_original(m))
    eq   = m.equilibrium_conditions

    # Load in parameters, steady-state parameters, and grids
    r::Float64     = m[:r].scaledvalue
    α::Float64     = m[:α].value
    H::Float64     = m[:H].value
    δ::Float64     = m[:δ].value
    μ_sp::Float64  = m[:μ_sp].value
#    ρ_sp::Float64  = m[:ρ_sp].value
#    σ_sp::Float64  = m[:σ_sp].value
    γ::Float64     = m[:γ].scaledvalue
    g::Float64     = m[:g].value
    η::Float64     = m[:η].value
    ρ_B::Float64    = m[:ρ_b].value
    ρ_G::Float64    = m[:ρ_g].value
    ρ_z::Float64    = m[:ρ_z].value
    ρ_μ::Float64    = m[:ρ_μ].value
    ρ_lamw::Float64 = m[:ρ_λ_w].value
    ρ_lamf::Float64 = m[:ρ_λ_f].value
    ρ_mon::Float64  = m[:ρ_rm].value
    spp::Float64   = m[:spp].value
    ϕh::Float64    = m[:ϕh].value
    ρ_R::Float64    = m[:ρR].value
    ψπ::Float64    = m[:ψπ].value
    ψy::Float64    = m[:ψy].value
    κ_p::Float64  = m[:κ_p].value
    κ_w::Float64  = m[:κ_w].value

    Tg::Float64 = m[:Tg].value
    δb::Float64 = m[:δb].value
    bg::Float64 = m[:bg].value


    R = 1 + r

    ell::Vector{Float64}  = m[:lstar].value
    c::Vector{Float64}    = m[:cstar].value
    μ::Vector{Float64}    = m[:μstar].value
    β::Float64            = m[:βstar].value

    T::Float64     = m[:Tstar].value
    ω::Float64     = m[:ωstar].value
    xstar::Float64 = m[:xstar].value
    ystar::Float64 = m[:ystar].value
    Rk::Float64    = m[:Rkstar].value
    kstar::Float64 = m[:kstar].value


    xgrid::Vector{Float64} = m.grids[:xgrid].points
    xwts::Vector{Float64}  = m.grids[:xgrid].weights
    sgrid::Vector{Float64} = m.grids[:sgrid].points
    swts::Vector{Float64}  = m.grids[:sgrid].weights
    fgrid::Matrix{Float64} = m.grids[:fgrid]
    xswts = kron(swts,xwts)

    zlo::Float64 = m[:zlo].value
    zhi::Float64 = m[:zhi].value

    nx::Int = get_setting(m, :nx)
    ns::Int = get_setting(m, :ns)
    nxns = nx*ns

    qp(z) = dmollifier_hetdsgegovdebt(z, zhi, zlo)
    qfunction_hetdsgegovdebt(x) = mollifier_hetdsgegovdebt(x, zhi, zlo) #/sumz

    unc = 1 ./ ell .<= repeat(xgrid,ns) .+ η

    euler_equation_hetdsgegovdebt_eq(nx, ns, qp, qfunction_hetdsgegovdebt, xgrid, sgrid, fgrid, unc, xswts,
                               R, γ, β, η, ell, T, ω, H)

    # KF Equation
    kolmogorov_fwd_hetdsgegovdebt(nx, ns, qfunction_hetdsgegovdebt, qp, xgrid, sgrid, fgrid, unc, xswts,
                               R, γ, ell, μ, η, T, ω, H, ee)

    setup_indices!(m)
    normalize_model_state_indices!(m)

    nvars = get_setting(m, :nvars)
    # Make the Jacobian
    JJ = zeros(nvars, 2*nvars)

    function sfunc(x::Float64) = (spp/2)*x^2
    function sfunc_prime(x::Float64) = spp*x

    transfer_eq = endo[:capreturn_t]*endo[:k_t]*exp(-endo[:z_t]) - endo[:I_t] + (1-endo[:mc_t])*endo[:y_t] - endo[:tg_t] - endo[:t_t]'
    other_eq = endo[:margutil_T]*endo[:q_t]*endo[:μ_t]*(1-s_func(endo[:I_t]/endo[:I_t1]*exp(-endo[:z_t]))*endo[:q_t]*endo[:μ′]*sfunc_prime(endo[:I_t′]/endo[:I_t]*endo[:z′_t])*(endo[:I_t′]/endo[:I_t]*endo[:z′_t])^2 - endo[:margutil_t](1+endo[:q_t]*endo[:μ_t]*sfunc_prime(endo[:I_t]/endo[:I_t1]*exp(endo[:z_t]))*endo[:I_t]/endo[:I_t1]*exp(endo[:z_t]))

    tobinq_eq = (1-δ)*endo[:k_t]*exp(-endo[:z_t]) + μ*(-xstar/kstar)*xstar - endo[:q_t]
    capital_eq = (1-δ)*endo[:k_t]*exp(-endo[:z_t]) + m[:μ_t]*(1-sfunc(endo[:I_t]/endo[:I_t1] * exp(γ)))*endo[:I_t] - endo[:k′_t]
    wage_inflation_eq = #figure out how to deal with kappa_w and kappa_p coefficients
    inflation_eq = (endo[:λ_w_t]* # see above

    marg_cost_eq = ((1-α)^(1-α)*α^α)^(-1)*endo[:w_t]^(1-α)*endo[:capreturn_t]^α - endo[:mc_t]
    gdp_eq = exp(-α*endo[:z_t])*endo[:k_t]^α*endo[:L_t]^(1-α) - endo[:y_t]

    cap_return_eq = α/(1-α)*endo[:L_t]/endo[:k_t]*exp(-endo[:z_t]) - endo[:capreturn_t]/endo[:w_t]
    investment_eq = ((1+endo[:i_t])/(1+iii))^ρR*((endo[:π]/π_star)^ψπ * (endo[:y_t]/endo[:y_t1]*exp(-γ))^ψy)^(1-ρR)*exp(endo[:rm′_t]) - (1+endo[:i_t])/(1+iii)
    other_eq =
    wage_eq = endo[:π_t]*exp(endo[:z_t])*endo[:w_t]/endo[:w_t1]
    T_g_eq = δb(endo[:bg_t]*exp(-endo[:z_t]) + (1-1/endo[:g_t])*endo[:y_t] - bg/(1+endo[:R_t])) + (1-δ)*Tgstar - endo[:Tg_t]
    b_g_eq = endo[:bg_t]*exp(-endo[:z_t]) + endo[:g_t] - endo[:tg_t] - endo[:bg′_t]/(1+endo[:R_t])
    budget_constraint_eq = endo[:y_t]/endo[:g_t] - endo[:c_t] - endo[:I_t]

    return [transfer_eq; other_eq; tobinq_eq; capital_eq; wage_inflation_eq; inflation_eq;
            inflation_eq; marg_cost_eq; gdp_eq; cap_return_eq; investment_eq; other_eq2;
            wage_eq; T_g_eq; b_g_eq; budget_constraint_eq]

end


function euler_equation_hetdsgegovdebt_eq(nx::Int, ns::Int,
                                qp::Function, qfunction::Function,
                                xgrid::Vector{Float64}, sgrid::Vector{Float64},
                                fgrid::Matrix{Float64},
                                unc::BitArray,
                                xswts::Vector{Float64},
                                R::Float64, γ::Float64, β::Float64,
                                η::Float64, ell::Vector{Float64}, T::Float64,
                                ω::Float64, H::Float64)

    qfunction(x) # => mollifier_hetdsgegovdebt(x, zhi, zlo) #g: pdf of the transitory skill shock
    nxns = nx*ns

    ell_euler = 0.
    ee  = zeros(nxns,nxns) # ee[i,j] takes you from i to j

    cfunc(a, s) = minimum(1/ell[i], a)

    for iss=1:ns
        for ia=1:nx
            i  = nx*(iss-1)+ia
            for isp=1:ns
                for iap=1:nx
                    ip = nx*(isp-1)+iap
                    ee[i,ip] = (xgrid[iap] - R*(exp(-γ))*max(xgrid[ia]-1/ell[i], -η) - T)/(ω*H*sgrid[isp])
                    cfunc = mininum(1/ell[i], η)
                    ell_euler += (exp(-γ)/cfunc) * qfunction(ee[i,ip])*(fgrid[iss,isp])/(ω*H*sgrid[isp])
                end
            end
        end
    end
    ell_euler = β*b*R*ell_euler
    return ell_euler
end

function kolmogorov_fwd_hetdsgegovdebt_eq(nx::Int, ns::Int,
                                qfunction::Function, qp::Function,
                                xgrid::Vector{Float64}, sgrid::Vector{Float64},
                                fgrid::Matrix{Float64}, unc::BitArray,
                                xswts::Vector{Float64},
                                R::Float64, γ::Float64,
                                ell::Vector{Float64}, μ::Vector{Float64},
                                η::Float64, T::Float64, ω::Float64, H::Float64, ee::Matrix{Float64})
    nxns = nx*ns

    for isp=1:ns
        for iap=1:nx
            ip  = nx*(isp-1)+iap
            for iss=1:ns
                for ia=1:nx
                    i = nx*(iss-1)+ia

                    ee[i,ip] = (xgrid[iap] - R*(exp(-γ))*max(xgrid[ia]-1/ell[i], -η) - T)/(ω*H*sgrid[isp])
                    # EM: I think this is wrong since this is equation for ω_i^a ω_j^s m(a_i, s_j^p) in the derivatives (lineared KF) but we actually want m_t(a_i, s). Ask Keshav?
                    mmm = xswts[i]*μ[i]
                    m_kolmogorov += mmm * qfunction(ee[i,ip])*(fgrid[iss,isp])/(ω*H*sgrid[isp])

                end
            end
        end
    end
    return m_kolmogorov
end

function consumption_eq(ell, η, xswts, μ)
    for iss=1:ns
        for ia=1:nx
            i  = nx*(iss-1)+ia
            for isp=1:ns
                for iap=1:nx
                    ip = nx*(isp-1)+iap
                    cfunc = mininum(1/ell[i], η)
                    # EM: I think this is wrong since this is equation for ω_i^a ω_j^s m(a_i, s_j^p) in the derivatives (lineared KF) but we actually want m_t(a_i, s). Ask Keshav?
                    mmm = xswts[i]*μ[i]
                    con_eq += = cfunc*mmm
                end
            end
        end
    end
    return con_eq
end

function lambda_eq(ell, η, xswts, μ)
    for iss=1:ns
        for ia=1:nx
            i  = nx*(iss-1)+ia
            for isp=1:ns
                for iap=1:nx
                    ip = nx*(isp-1)+iap
                    cfunc = mininum(1/ell[i], η)
                    # EM: I think this is wrong since this is equation for ω_i^a ω_j^s m(a_i, s_j^p) in the derivatives (lineared KF) but we actually want m_t(a_i, s). Ask Keshav?
                    mmm = xswts[i]*μ[i]
                    lam_eq += = (1/cfunc)*mmm
                end
            end
        end
    end
    return lam_eq
end



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
