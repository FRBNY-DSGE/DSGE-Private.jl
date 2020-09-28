# TODO: refactor to reduce number of allocations further, e.g. C_Final, c_pol_in, etc.
#       e.g. refactor to not allocate KF_in if using kf_eigen or transition_mat if using kf_anderson
function steadystate!(m::HetDSGEGovDebt;
                      βlo::S = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                      βhi::S = min(exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), 0.999999),
                      excess::S = 5000., tol::S = 1e-4, maxit::Int64 = 20, βband::S = 1e-2,
                      euler_anderson::Bool = true, kf_anderson::Bool = false, kf_eigen::Bool = true,
                      doplots::Bool = false, verbose::Symbol = :none) where {S <: Real}
    @assert !(kf_eigen && kf_anderson) "Only one of kf_anderson and kf_eigen can be true"

    # If we have already solved for βstar (i.e. it's not NaN) and we only want to
    # estimate the non steady state parameters, there's no need to recompute
    # elo/ehi, etc.
    if !isnan(m[:βstar].value) && get_setting(m, :estimate_only_non_steady_state_parameters)
        return
    else
        # Initialize g function
        gfunc(x) = gfunc_default(x) 
        if get_setting(m, :gfunc_type) == :mollifier
            gfunc(x) = mollifier_hetdsgegovdebt(x, m[:ehi].value, m[:elo].value)
            gfunc_wbounds(x, y, z) = mollifier_hetdsgegovdebt(x, y, z) # need this for generate_us_and_es
        elseif get_setting(m, :gfunc_type) == :lognormal
            # calculate mu (do we need to do this every steady state if ehi and elo are fixed?
            function nl_mean!(F, x)
                F[1] = DSGE.lognormal_mean(m[:ehi].value, m[:elo].value, x[1], m[:σ_e].value) - 1
            end
            m[:μ_e].value = nlsolve(nl_mean!, [-m[:σ_e].value^2/2]).zero[1]
            gfunc(x) = lognormal_hetdsgegovdebt(x, m[:ehi].value, m[:elo].value, m[:μ_e].value, m[:σ_e].value)
            gfunc_wbounds(x, y, z) = lognormal_hetdsgegovdebt(x, y, z, m[:μ_e].value, m[:σ_e].value)
        end

        reset_grids!(m; gfunc = gfunc)
        na = get_setting(m, :na)
        ns = get_setting(m, :ns)
        ne = get_setting(m, :ne)
       
        if get_setting(m, :calibrate_income_targets)
            m[:sH_over_sL], m[:elo], m[:ehi] = compute_income_process_parameters(m, gfunc_wbounds)
        else
            m[:varlinc], m[:vardlinc] = skill_moments(m[:sH_over_sL].value, m[:elo].value,
                                                      m[:pLH].value, m[:pHL].value,
                                                      get_setting(m, :us),
                                                      get_setting(m, :es), get_setting(m, :n_calibration_iters))
        end

        # Parameters
        ω = m[:ωstar].value
        H = m[:H].value
        T = m[:Tstar].value
        γ = m[:γ].scaledvalue
        R  = 1 + m[:r].scaledvalue
        η  = m[:η].value
        bg = m[:bg].value

        # Construct Markov transition matrix for skill
        pLH = m[:pLH].value
        pHL = m[:pHL].value
        f = [[1-pLH pLH];[pHL 1-pHL]] # f1[i,j] is prob of going from i to j
        sH_over_sL = m[:sH_over_sL].value
     
        # Construct sgrid
        sgrid, swts, sscale = construct_sgrid(pHL, pLH, sH_over_sL, ns)
        m.grids[:sgrid] = Grid(sgrid, swts, sscale)

        # Construct egrid
        egrid, ewts, g_of_e = construct_egrid(m[:ehi].value, m[:elo].value, ne, gfunc)
        # m.grids[:egrid] = Grid(egrid, ewts)


        # Run loop expanding the agrid if a CashOnHandError is caught
        ahi_guesses = if haskey(get_settings(m), :ahi_incs) # Construct guesses for the upper bound of agrid
            get_setting(m, :ahi_incs)
        elseif haskey(get_settings(m), :ahi_inc)
            [get_setting(m, :ahi_inc)]
        else
            [NaN]
        end
        m_anderson = haskey(get_settings(m), :m_anderson) ? get_setting(m, :m_anderson) : 5
        β_anderson = haskey(get_settings(m), :β_anderson) ? get_setting(m, :β_anderson) : 1.
        for ahi_guess in ahi_guesses
            try
                # Construct agrid
                agrid, awts, ascale = construct_agrid(sgrid, egrid, ω, H, R, η, γ, T, na; ahi_inc = ahi_guess)

                m <= Setting(:alo, agrid[1])
                m <= Setting(:ahi, agrid[end])
                m <= Setting(:ascale, ascale)

                # Once have updated grids, can call steady state and compute other two moments
                find_steadystate!(m, na, ns, ne, egrid, ewts, g_of_e,
                                  f, agrid, sgrid, R, H, η, γ, ω, T, bg, gfunc;
                                  βlo = βlo, βhi = βhi, excess = excess, tol = tol, maxit = maxit,
                                  βband = βband, doplots = doplots, m_anderson = m_anderson, β_anderson = β_anderson,
                                  transition_mat = kf_eigen ? Matrix{S}(undef, na * ns, na * ns) : Matrix{S}(undef, 0, 0),
                                  euler_anderson = euler_anderson, kf_anderson = kf_anderson, kf_eigen = kf_eigen,
                                  verbose = verbose)

                if verbose == :high
                    println("The distribution D(a, s) integrates to $(round(sum(m[:Dstar].value), digits = 3)).")

                    # Calculate some summary statistics
                    print_summary_stats(m)
                end

                if doplots
                    plot_steadystate_asgrid(m, agrid)
                end

                # use ones for now before refactoring since no longer using quadrature over s
                m[:mpc] = ave_mpc(m[:Dstar].value,   m[:cstar].value, agrid, na, ns)
                m[:pc0] = frac_zero(m[:Dstar].value, m[:cstar].value, agrid, ns)
                m.grids[:agrid] = Grid(agrid, awts, ascale) # Save final agrid

                break
            catch e
                if ahi_guess == ahi_guesses[end]
                    rethrow(e)
                elseif isa(e, CashOnHandError) # Try to address this problem by increasing ahi
                    continue
                else
                    rethrow(e)
                end
            end
        end
    end
    m
end

function find_steadystate!(m::HetDSGEGovDebt, na::Int, ns::Int, ne::Int,
                           egrid::Vector{Float64}, ewts::Vector{Float64}, g_of_e::Vector{Float64},
                           f::Matrix{Float64},
                           agrid::Vector{Float64}, sgrid::Vector{Float64},
                           R::Float64, H::Float64, η::Float64, γ::Float64, ω::Float64, T::Float64, bg::Float64, gfunc;
                           βlo::S = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                           βhi::S = min(exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), 0.9999999),
                           excess::S = 5000., tol::S = 1e-4, maxit::Int64 = 20, βband::S = 1e-2,
                           euler_anderson::Bool = false, kf_anderson::Bool = false, kf_eigen::Bool = false,
                           m_anderson::Int = 5, β_anderson::S = 1.,
                           transition_mat::Matrix{S} = Matrix{S}(undef, 0, 0),
                           doplots::Bool = false, verbose::Symbol = :high) where {S <: Real}

    na_c = get_setting(m, :na_c)

    βlo_temp = βlo
    βhi_temp = βhi

    reject  = false
    counter = 0

    # Initial guess
    β         = NaN # Need to define β here so it's accessible outside of while lopp
    c_pol_in  = (R - 1) * repeat(agrid, 1, ns, ne) .+ ω * H * repeat(sgrid', na, 1, ne)
    KF_in     = fill(1.0 / (ns * na), na, ns)

    # If want to keep the last β/don't recompute
    if get_setting(m, :use_last_βstar) && !isnan(m[:βstar].value)

        # This short-circuits computation of the policy function
        βlo = βhi = m[:βstar].value
        # If you have a β saved, start from there (if you don't, βhi and βlo are defined as they are passed in to function)
    elseif !isnan(m[:βstar].value)

        # If one has computed β* before, then we first bisect into a neighborhood around it and check if the signs are the opposite
        βlo_temp = m[:βstar].value - βband
        βhi_temp = m[:βstar].value + βband
        c_pol_out, c_as, bp, KF, reject = policy_hetdsgegovdebt(na, ns, ne, na_c, βlo_temp, R, ω, H, η, T, γ,
                                                                m[:ehi].value, m[:elo].value,
                                                                agrid, sgrid, c_pol_in, KF_in,
                                                                f, egrid, ewts, g_of_e, transition_mat, gfunc,
                                                                maxit = get_setting(m, :policy_maxit),
                                                                m_anderson = m_anderson,
                                                                β_anderson = β_anderson,
                                                                euler_anderson = euler_anderson,
                                                                kf_anderson = kf_anderson,
                                                                kf_eigen = kf_eigen,
                                                                eigen_method = haskey(get_settings(m), :eigen_method) ?
                                                                get_setting(m, :eigen_method) : :krylov,
                                                                cas_fixedpoint = haskey(get_settings(m), :cas_fixedpoint) ?
                                                                get_setting(m, :cas_fixedpoint) : false,
                                                                cas_learning_rate = haskey(get_settings(m), :cas_learning_rate) ?
                                                                get_setting(m, :cas_learning_rate) : .3,
                                                                euler_tol = get_setting(m, :euler_tol),
                                                                kf_tol = get_setting(m, :kf_tol),
                                                                eigen_tol = get_setting(m, :eigen_tol),
                                                                C_tol = get_setting(m, :C_tol))

        excess_lo = compute_excess(KF, bp, bg)

        if excess_lo < 0 && abs(excess_lo) > tol
            βlo = βlo_temp
            c_pol_out, c_as, bp, KF, reject = policy_hetdsgegovdebt(na, ns, ne, na_c, βlo_temp, R, ω, H, η, T, γ,
                                                                    m[:ehi].value, m[:elo].value,
                                                                    agrid, sgrid, c_pol_in, KF_in,
                                                                    f, egrid, ewts, g_of_e, transition_mat, gfunc,
                                                                    maxit = get_setting(m, :policy_maxit),
                                                                    m_anderson = m_anderson,
                                                                    β_anderson = β_anderson,
                                                                    euler_anderson = euler_anderson,
                                                                    kf_anderson = kf_anderson,
                                                                    kf_eigen = kf_eigen,
                                                                    eigen_method = haskey(get_settings(m), :eigen_method) ?
                                                                    get_setting(m, :eigen_method) : :krylov,
                                                                    cas_fixedpoint = haskey(get_settings(m), :cas_fixedpoint) ?
                                                                    get_setting(m, :cas_fixedpoint) : false,
                                                                    cas_learning_rate = haskey(get_settings(m), :cas_learning_rate) ?
                                                                    get_setting(m, :cas_learning_rate) : .3,
                                                                    euler_tol = get_setting(m, :euler_tol),
                                                                    kf_tol = get_setting(m, :kf_tol),
                                                                    eigen_tol = get_setting(m, :eigen_tol),
                                                                    C_tol = get_setting(m, :C_tol))

            # Update the guess for c(b, s, e) only if the Euler iteration converged
            if !reject
                c_pol_in = c_pol_out
            end

            excess_hi = compute_excess(KF, bp, bg)

            if excess_hi > 0
                βhi = βhi_temp
            end
        elseif excess_lo >= 0
            βhi = βlo_temp
        end
    end

    while abs(excess) > tol && counter <= maxit # clearing markets
        β = (βlo + βhi) / 2.0
        c_pol_out, c_as, bp, KF, reject = policy_hetdsgegovdebt(na, ns, ne, na_c, β, R, ω, H, η, T, γ,
                                                                m[:ehi].value, m[:elo].value,
                                                                agrid, sgrid, c_pol_in, KF_in,
                                                                f, egrid, ewts, g_of_e, transition_mat, gfunc,
                                                                maxit = get_setting(m, :policy_maxit),
                                                                m_anderson = m_anderson,
                                                                β_anderson = β_anderson,
                                                                euler_anderson = euler_anderson,
                                                                kf_anderson = kf_anderson,
                                                                kf_eigen = kf_eigen,
                                                                eigen_method = haskey(get_settings(m), :eigen_method) ?
                                                                get_setting(m, :eigen_method) : :krylov,
                                                                cas_fixedpoint = haskey(get_settings(m), :cas_fixedpoint) ?
                                                                get_setting(m, :cas_fixedpoint) : false,
                                                                cas_learning_rate = haskey(get_settings(m), :cas_learning_rate) ?
                                                                get_setting(m, :cas_learning_rate) : .3,
                                                                euler_tol = get_setting(m, :euler_tol),
                                                                kf_tol = get_setting(m, :kf_tol),
                                                                eigen_tol = get_setting(m, :eigen_tol),
                                                                C_tol = get_setting(m, :C_tol))

        # Update the guess for c(b, s, e) only if the Euler iteration converged
        if !reject
            c_pol_in = c_pol_out
        end

        excess = compute_excess(KF, bp, bg)
        if verbose == :high
            println("On (i, β) = ($(counter), $(round(β, digits = 6))), excess bonds are $(excess).")
        end

        # bisection
        if excess > 0 || reject # Consumption policy iterations typically do not converge when β is too close to 1
            βhi = β
        elseif excess < 0
            βlo = β
        end
        counter += 1
        if get_setting(m, :use_last_βstar) && !isnan(m[:βstar].value)
            break
        end
    end

    if counter == maxit
        @warn "Bisection for β does not converge to a market-clearing value."
        reject = true
        # TODO: maybe do not auto-update c_pol_in unless you have euler iteration covnergence.
        #       c_pol_in should instead be set back to initial guess
    end

    bgrid, agrid_big = construct_bgrid_agrid_big(agrid, sgrid, egrid, na, ns, ne, γ, ω, H, T, R)

    if doplots
        plot_steadystate_bsegrid(bgrid, egrid, c_pol_in)
    end

    # Calculate euler equation error
    euler_err, euler_dev = if (haskey(get_settings(m), :cas_fixedpoint) ? get_setting(m, :cas_fixedpoint) : false)
        cas_euler_err(reshape(c_as, na, ns), β, R, γ, T, ω, H, na, ns, agrid, sgrid, f, gfunc)
    else
        cbse_euler_err(c_pol_in, ns, ne, na_c, f, ewts, g_of_e, bgrid, bgrid, sgrid,
                       egrid, β, R, γ, ω, H, T)
    end

    # If policy function does not converge, we signal to likelihood that should reject
    m <= Setting(:auto_reject, reject)
    m[:lstar]     = 1 ./ c_as
    m[:cstar]     = c_as
    m[:Dstar]     = KF
    m[:βstar]     = β
    m[:β_save]    = β
    m[:euler_err] = euler_err
    m[:euler_deviation] = euler_dev

    m
end

function policy_hetdsgegovdebt(na::Int, ns::Int, ne::Int, na_c::Int, β::S, R::S, ω::S, H::S, η::S,
                               T::S, γ::S, ehi::S, elo::S, agrid::Vector{S},
                               sgrid::Vector{S}, c_pol::AbstractArray{S, 3}, KF_in::AbstractArray{S, 2},
                               f::Matrix{S}, egrid::Vector{Float64}, ewts::Vector{Float64},
                               g_of_e::Vector{Float64}, transition_mat::AbstractMatrix{S}, gfunc, dist::S = 1.;
                               maxit::Int64 = 1000, m_anderson::Int = 5, β_anderson::S = 1.,
                               euler_anderson::Bool = false, kf_anderson::Bool = false, kf_eigen::Bool = false,
                               eigen_method::Symbol = :krylov, cas_fixedpoint::Bool = false, cas_learning_rate::Float64 = .3,
                               euler_tol::S = 1e-10, kf_tol::S = 1e-10,
                               eigen_tol::S = 2e-1, C_tol::S = -1e-8) where {S <: Real}

    colors = [:red, :blue, :green, :yellow, :purple]
    counter = 0
    reject = false

    # Constrainted consumption
    c_constrained = Matrix{Float64}(undef, ns, ne)
    for ie in 1:ne
        e = egrid[ie]
        c_constrained[:, ie] = sgrid .* (ω * e * H) .+ T
    end

    bgrid, agrid_big = construct_bgrid_agrid_big(agrid, sgrid, egrid, na, ns, ne, γ, ω, H, T, R)

    # Allocate memory here
    bp       = bgrid # bp = b' = bprime, only to make it clear that we're working with b', not b
    sum_term = zeros(S, length(bp))

    # Fixed point problem for c_poli
    if euler_anderson
        out = nlsolve((F_c_pol, c_pol) -> fixedpoint_cbse_nlsolve!(F_c_pol, c_pol, ns, ne, na_c, sum_term, f, ewts, g_of_e,
                                                                   bp, bgrid, sgrid, egrid, c_constrained, β, R, γ, ω, H, T),
                      c_pol, ftol = euler_tol, iterations = maxit, m = m_anderson, beta = β_anderson, method = :anderson)
        if !out.f_converged
            @warn "Euler iteration did not converge. The final distance is $(out.residual_norm)"
            reject = true
        end
        c_pol = out.zero
    else
        c_poli = similar(c_pol)
        while dist > euler_tol && counter <= maxit
            fixedpoint_cbse!(c_poli, c_pol, ns, ne, na_c, sum_term, f, ewts, g_of_e,
                             bp, bgrid, sgrid, egrid, c_constrained, β, R, γ, ω, H, T)

            dist     = maximum(abs.(c_pol - c_poli)) # Inf norm
            c_pol   .= c_poli
            counter += 1

            if counter == maxit
                @warn "Euler iteration did not converge. The final distance is $(dist)"
                reject = true
            end
        end
    end

    if any(agrid_big - c_pol .< -1e-14)
        throw(CashOnHandError("Consumption is not always less than cash on hand. " *
                              "The maximum excess is $(maximum(c_pol - agrid_big))."))
    end

    # Calculate c(a, s)
    c_as = integrate_out_e(agrid, agrid_big, bgrid, sgrid, c_pol, tol = C_tol)

    # Solve Euler equation in (a, s)?
    if cas_fixedpoint # TODO: wrap inside its own function
        agrη = repeat(agrid, 1, ns)
        ell_pol_in = similar(c_as)
        euler_equation_hetdsgegovdebt!(ell_pol_in, β, 1., R, γ, T, ω, H, c_as, c_as, na, ns, (agrid[end] - agrid[1]) / na,
                                       agrid, sgrid, f, agrη, gfunc)
        c_as .= min.(agrη, 1 ./ ell_pol_in)
        ell_pol_out = similar(ell_pol_in)
        dist = 1.
        while dist > euler_tol && counter <= maxit
            euler_equation_hetdsgegovdebt!(ell_pol_out, β, 1., R, γ, T, ω, H, c_as, c_as, na, ns, (agrid[end] - agrid[1]) / na,
                                           agrid, sgrid, f, agrη, gfunc)

            ell_pol_out .= cas_learning_rate .* ell_pol_out + (1 - cas_learning_rate) .* ell_pol_in
            dist         = maximum(abs.(ell_pol_in - ell_pol_out))
            ell_pol_in  .= ell_pol_out
            c_as        .= min.(agrη, 1 ./ ell_pol_out)

            counter += 1

            if counter == maxit
                @warn "Euler iteration did not converge for the second fixed point over the (a, s) grid. The final distance is $(dist)"
                reject = true
            end
        end
    end

    # Calculate bp(a, s)
    bp = (R * exp(-γ)) .* (agrid .- c_as)

    # Fixed point problem for D(a, s)
    if kf_anderson
        out  = nlsolve((F_D, D_as_in) -> fixedpoint_KF_nlsolve!(F_D, D_as_in, c_as, agrid, sgrid, f, gfunc, ω, H, R, γ, T),
                       KF_in, ftol = kf_tol, iterations = maxit, method = :anderson, m = m_anderson, beta = β_anderson)
        D_as = vec(out.zero) # out.zero already allocated so just re-assign D_as
    elseif kf_eigen
        D_as = stationary_KF!(transition_mat, c_as, agrid, sgrid, f, gfunc, ω, H, R, γ, T, (agrid[end] - agrid[1]) / length(agrid);
                              tol = eigen_tol, method = eigen_method)
    else
        D_as  = deepcopy(KF_in) # do not want to alter the initial guess
        D′_as = similar(D_as)
        counter = 0
        dif     = 1.
        while dif > kf_tol && counter <= maxit
            fixedpoint_KF!(D′_as, D_as, c_as, agrid, sgrid, f, gfunc, ω, H, R, γ, T)

            # check convergence
            dif      = maximum(abs.(D′_as - D_as))
            D_as    .= D′_as
            counter += 1
        end
        D_as = vec(D_as)
    end

    return c_pol, vec(c_as), vec(bp), D_as, reject
end

function fixedpoint_cbse!(c_poli::AbstractArray{S, 3}, c_pol::AbstractArray{S, 3}, ns::Int, ne::Int, na_c::Int,
                          sum_term::AbstractVector{S}, f::AbstractMatrix{S},
                          ewts::AbstractVector{S}, g_of_e::AbstractVector{S},
                          bp::AbstractVector{S}, bgrid::AbstractVector{S}, sgrid::AbstractVector{S},
                          egrid::AbstractVector{S}, c_constrained::AbstractMatrix{S},
                          β::S, R::S, γ::S, ω::S, H::S, T::S) where {S <: Real}

    @inbounds for is in 1:ns
        @inbounds for ie in 1:ne
            # Sums
            verify_one = 0.0
            @inbounds for iep in 1:ne     # Inner integral over e' (done first b/c Julia is column major)
                @inbounds for isp in 1:ns # Outer sum over s'
                    #               p(s'|s)    * iota(e')  *      g(e')    * c(b′, s', e′)^{-1}
                    # Note that f[is, isp] is prob going froms sgrid[is] to sgrid[isp], so we do want to
                    # iterate over the second element rather than the first, despite the p(s'|s) notation
                    sum_term .+= (f[is, isp] * ewts[iep] * g_of_e[iep]) ./ c_pol[:, isp, iep]
                end
            end

            # Compute ell(b′, s, e) = β * R * exp(-γ) * (Σₛₚ∫ₑₚ) = β * R * exp(-γ) * sum_term
            # Note that b′ is a choice variable today because it is how much an agent chooses to save
            # today to obtain b′ assets tomorrow. Thus, agents do not have any uncertainty about b′
            # once they know what s and e today are, hence the quadrature for the expectation
            # only needs to occur over (s′, e′). It follows that consumption today as a function
            # of savings b′ tomorrow is c(b′, s, e) = 1 / ell(b′, s, e).
            #
            # Note that c(b′, s, e) is the quantity of consumption today that will get you to b′ tomorrow,
            # not the quantity of consumption chosen if an agent has b′ assets today.
            # Once you know c(b′, s, e), you can calculate b(b′), i.e. assets today as a function of b′:
            # b′ / R = exp(-γ) b - c + ω * s * e * H + T ⇒ b = exp(γ) * (b′ / R - ω * s * e * H - T + c)
            #
            # We now have c = c(b′, s, e) and b = b(b′), where b′ = bgrid, i.e. it is a fixed grid.
            # Since c(b′, s, e) and b(b′) are functions, c(b′, s, e) and b(b′) map b′ to a unique
            # choice of consumption today and assets today. In other words, given b′ tomorrow, there is
            # no other pair (c, b) that implies b′ tomorrow. Thus, c(b(b′), s, e) = c(b′, s, e)
            c = 1 ./ ((β * R * exp(-γ)) .* sum_term) # c(b′, s, e)
            b = vec(exp(γ) * ((bp ./ R) .- ω * sgrid[is] * egrid[ie] * H .- T .+ c)) # b(b′)
            sum_term .= 0. # Reset the values to zero (done with it for this loop)
            if all(b .< 0.)
                # Increasing upper bound of agrid and number of egrid points are generally the best options
                throw(CashOnHandError("All elements of b (assets today) are negative. " *
                                      "The last element is b=$(round(b[end], digits = 3)). " *
                                      "Try increasing the upper bound of the agrid, increasing " *
                                      "the number of egrid points, increasing the number of agrid points, and/or lowering β"))
            end

            # Handle constrained consumption today: for b < 0, need to reset c(b′, s, e) such that b today is exactly 0
            c[b .< 0.] = -(bp[b .< 0] ./ R) .+  ω * sgrid[is] * egrid[ie] * H .+ T
            b         .= vec(exp(γ) * ((bp ./ R) .- ω * sgrid[is] * egrid[ie] * H .- T .+ c))

            # If b[1] is positive, then people aren't using a c(0, s, e) policy (i.e. save nothing and set b′ = 0)
            # but also must have some spare assets they consume. So need to handle some edge cases
            if b[1] > 0. # This code block assumes b_c > 0 for all i in 2:length(b)
                if c[1] < c_constrained[is, ie] # Check consuming more than implied by constrained rule if b[1] > 0
                    # Increasing upper bound of agrid and number of egrid points are generally the best options
                    throw(CashOnHandError("c[1] - c_constrained[is, ie]=$(c[1] - c_constrained[is, ie]), " *
                                          "try increasing the upper bound of the agrid, " *
                                          "the number of egrid points, or the number of agrid points."))
                end
                c_c = collect(range(c_constrained[is, ie], c[1], length = na_c)) # constrained rule is an equal spacing
                b_c = exp(γ) * (-ω * sgrid[is] * egrid[ie] * H - T .+ c_c)       # between no-assets rule and c[1]
                if -1e-14 < b_c[1] < 0.
                    b_c[1] = 0. # Force the first point to be zero (in case of floating point errors)
                end
                if any(b_c .< -1e-14)
                    # Increasing upper bound of agrid and number of egrid points are generally the best options
                    throw(CashOnHandError("Some assets today for constrained agents are negative. Try increasing the " *
                                          "upper bound of the agrid, the number of egrid points, or the number of agrid points."))
                end
                cutoff = findlast(b_c .< b[1])
                b = vcat(b_c[1:cutoff], b) # Add additional points for constrained people
                c = vcat(c_c[1:cutoff], c) # so that, for the new b, the condition b[1] ≈ 0 holds
            end

            # Nearest points, linear interpolation
            c_poli[:, is, ie] = try
                interp_one(b, c, bgrid)
            catch e
                if !issorted(b)
                    bsorted_inds = sortperm(b)
                    interp_one(b[bsorted_inds], c[bsorted_inds], bgrid)
                else
                    rethrow(e)
                end
            end
        end
    end
end

# See fixedpoint_cbse! for comments. This function produces the residual c_pol - F(c_pol)
function fixedpoint_cbse_nlsolve!(F_c_pol::AbstractArray{S, 3}, c_pol::AbstractArray{S, 3}, ns::Int, ne::Int, na_c::Int,
                                  sum_term::AbstractVector{S}, f::AbstractMatrix{S},
                                  ewts::AbstractVector{S}, g_of_e::AbstractVector{S},
                                  bp::AbstractVector{S}, bgrid::AbstractVector{S}, sgrid::AbstractVector{S},
                                  egrid::AbstractVector{S}, c_constrained::AbstractMatrix{S},
                                  β::S, R::S, γ::S, ω::S, H::S, T::S) where {S <: Real}
    @inbounds for is in 1:ns
        @inbounds for ie in 1:ne
            verify_one = 0.0
            @inbounds for iep in 1:ne
                @inbounds for isp in 1:ns
                    sum_term   .+= (f[is, isp] * ewts[iep] * g_of_e[iep]) ./ c_pol[:, isp, iep]
                    verify_one  +=  f[is, isp] * ewts[iep] * g_of_e[iep]
                end
            end
            @test isapprox(verify_one, 1.0, atol = 1e-5)

            c = 1 ./ ((β * R * exp(-γ)) .* sum_term)
            b = vec(exp(γ) * ((bp ./ R) .- ω * sgrid[is] * egrid[ie] * H .- T .+ c))
            sum_term .= 0.
            if all(b .< 0.)
                throw(CashOnHandError("All elements of b (assets today) are negative. " *
                                      "The last element is b=$(round(b[end], digits = 3)). " *
                                      "Try increasing the upper bound of agrid, increasing " *
                                      "the number of agrid points, and/or lowering β"))
            end

            c[b .< 0.] = -(bp[b .< 0] ./ R) .+  ω * sgrid[is] * egrid[ie] * H .+ T
            b         .= vec(exp(γ) * ((bp ./ R) .- ω * sgrid[is] * egrid[ie] * H .- T .+ c))

            if b[1] > 0.
                if c[1] < c_constrained[is, ie]
                    throw(CashOnHandError("c[1] - c_constrained[is, ie]=$(c[1] - c_constrained[is, ie]), " *
                                          "try increasing the upper bound of the agrid " *
                                          "or the number of agrid points."))
                end
                c_c = collect(range(c_constrained[is, ie], c[1], length = na_c))
                b_c = exp(γ) * (-ω * sgrid[is] * egrid[ie] * H - T .+ c_c)
                if -1e-10 < b_c[1] < 0.
                    b_c[1] = 0.
                end
                if any(b_c .< -1e-10)
                    throw(CashOnHandError("Some assets today for constrained agents are negative."))
                end
                cutoff = findlast(b_c .< b[1])
                b = vcat(b_c[1:cutoff], b)
                c = vcat(c_c[1:cutoff], c)
            end

            F_c_pol[:, is, ie] = try
                interp_one(b, c, bgrid) - c_pol[:, is, ie]
            catch e
                if !issorted(b)
                    bsorted_inds = sortperm(b)
                    interp_one(b[bsorted_inds], c[bsorted_inds], bgrid) - c_pol[:, is, ie]
                else
                    rethrow(e)
                end
            end
        end
    end
end

@inline function euler_equation_hetdsgegovdebt!(ret_ell::AbstractMatrix{S0}, β::S1, b_t::S0,
                                                R_t::S0, z′_t::S0, t′_t::S0, ω′_t::S0, H′_t::S0, # recall Hₜ ≡ Lₜ
                                                cfunc::AbstractMatrix{S0}, cfunc′::AbstractMatrix{S0},
                                                na::Int, ns::Int, ι::S1,
                                                agrid::AbstractVector{S1}, sgrid::AbstractVector{S1}, fgrid::AbstractMatrix{S1},
                                                agrη::AbstractMatrix{S1}, g_fn::Function) where {S0 <: Real, S1 <: Real}

    enegzpt = exp(-z′_t)
    Renegzpt = R_t * enegzpt
    ω′_H′ = ω′_t * H′_t

    # Loop over s
    @inbounds @simd for iss = 1:ns
        # Loop over a
        @inbounds @simd for ia = 1:na
            # Index for outcome
            ell_euler = 0.

            # Sum over s'
            @inbounds @simd for isp = 1:ns
                ω′_s′_H′ = ω′_H′ * sgrid[isp]

                # Sum over a'
                @inbounds @simd for iap = 1:na
                    # ee = (a' - Rₜ * exp(-z') (a - cₜ(a, s)) - T') / (ω's'H')
                    ee = (agrid[iap] - Renegzpt *
                          (agrid[ia] - cfunc[ia, iss]) - t′_t) / ω′_s′_H′

                    # ι * ∑∑(exp(-z'ₜ)/c'(a', s')) * g(ee) * p(s'|s) /(ω's'H') # f[iss, isp] = p(s'|s)
                    ell_euler += (enegzpt / cfunc′[iap, isp]) * g_fn(ee) * fgrid[iss, isp] / ω′_s′_H′
                end
            end
            ret_ell[ia, iss] = max((ι * β * b_t * R_t) * ell_euler, 1 / agrη[ia, iss])
        end
    end

    return ret_ell
end

function fixedpoint_KF!(D′_as::AbstractArray{S, 2}, D_as::AbstractArray{S, 2}, C_pol::AbstractArray{S, 2},
                        agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S},
                        gfunc, ω::S, H::S, R::S, γ::S, T::S) where {S <: Real}

    D′_as .= 0.

    # Calculate D′(a, s) from  D(a, s) by using the KF equation
    na, ns = size(D_as)
    @inbounds for isp in 1:ns
        @inbounds for iap in 1:na
            @inbounds for is in 1:ns
                @inbounds for ia in 1:na
                    # Technically calculating m′(a, s) from D(a, s) in this step
                    D′_as[iap, isp] += D_as[ia, is] * f[is, isp] / (ω * H * sgrid[isp]) *
                        gfunc((agrid[iap] - R * exp(-γ) * (agrid[ia] - C_pol[ia, is]) - T) / (ω * H * sgrid[isp]))
                end
            end
        end
    end

    D′_as ./= sum(D′_as) # ensure sum(D′(a, s)) = 1. This step maps m′(a, s) -> m(a, s)
end

function fixedpoint_KF_nlsolve!(F_D::AbstractArray{S, 2}, D_as::AbstractArray{S, 2}, C_pol::AbstractArray{S, 2},
                                agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S},
                                gfunc, ω::S, H::S, R::S, γ::S, T::S) where {S <: Real}

    # Zero out
    F_D .= 0.

    # Calculate D′(a, s) from  D(a, s) by using the KF equation
    na, ns = size(D_as)
    @inbounds for isp in 1:ns
        @inbounds for iap in 1:na
            @inbounds for is in 1:ns
                @inbounds for ia in 1:na
                    F_D[iap, isp] += D_as[ia, is] * f[is, isp] / (ω * H * sgrid[isp]) *
                        gfunc((agrid[iap] - R * exp(-γ) * (agrid[ia] - C_pol[ia, is]) - T) / (ω * H * sgrid[isp]))
                end
            end
        end
    end

    F_D ./= sum(F_D) # Normalize to 1
    F_D .-= D_as     # Subtract D(a, s) from F_D in place ⇒ F_D = D′(a, s) - D(a, s)
end

function stationary_KF!(transition_mat::AbstractMatrix{S}, C_as::AbstractMatrix{S},
                        agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S}, gfunc,
                        ω::S, H::S, R::S, γ::S, T::S, ι::S; tol::S = 2e-1, method::Symbol = :krylov) where {S <: Real}
    # Check D(a, s) solves the KF equation by constructing the transition matrix A and solving the eigenvalue problem.
    # Note that the Kolmogorov forward equation we use is defined for m(a, s) = D(a, s) / ι, hence
    # vec(m′(a, s))     = A * ι * vec(m(a, s)), which reduces to
    # vec(D′(a, s)) / ι = A * vec(D(a, s))
    # vec(D′(a, s))     = A * ι * vec(D(a, s))
    construct_transition!(transition_mat, C_as, agrid, sgrid, f, gfunc, ω, H, R, γ, T)

    # Find largest eigenvalue and associated eigenvector
    # For computational reasons, we solve D(a, s) / ι = A * D(a, s) instead of D(a, s) = A * ι * D(a, s)
    ~, μ = stationary_eigenvector(sparse(transition_mat), ι, method; tol = tol)
    μ ./= sum(μ) # normalize to 1

    # return reshape(μ, na, ns)
    return μ
end

function construct_transition!(transition_mat::AbstractMatrix{S}, C_as::AbstractMatrix{S},
                               agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S},
                               gfunc, ω::S, H::S, R::S, γ::S, T::S) where {S <: Real}
    na = length(agrid)
    ns = length(sgrid)
    @inbounds for is in 1:ns
        @inbounds for ia in 1:na
            @inbounds for isp in 1:ns
                @inbounds for iap in 1:na
                    transition_mat[na * (isp - 1) + iap, na * (is - 1) + ia] = f[is, isp] / (ω * H * sgrid[isp]) *
                        gfunc((agrid[iap] - R * exp(-γ) * (agrid[ia] - C_as[ia, is]) - T) / (ω * H * sgrid[isp]))
                end
            end
        end
    end
end

# Maps c(b, s, e) -> c(a, s)
function integrate_out_e(agrid::AbstractVector{S1}, agrid_big::AbstractArray{S2, 3},
                         bgrid::AbstractVector{S1}, sgrid::AbstractVector{S1}, c_pol::AbstractArray{S3, 3};
                         tol::Float64 = -1e-8) where {S1 <: Real, S2 <: Real, S3 <: Real}
    # Map c(b, s, e) -> c(a, s, e)
    na, ns, ne = size(agrid_big)
    C_as    = Matrix{S3}(undef, na, ns)
    @inbounds for is in 1:ns
        # Sort the a's, given the skill level
        vec_agrid_big_is = vec(agrid_big[:, is, :])
        sorted_inds = sortperm(vec_agrid_big_is)

        # agrid_big is the grid of a implied by bgrid and exogenous states (s, e),
        # whereas agrid is the grid of a that's passed in.
        # Linear interpolation with the sorted agrid. This approach works b/c mapping b into a is
        # really the mapping (b, s, e) -> (a(b, s, e), s, e). Thus, the value of a implicitly already
        # accounts for e. Intuitively, having a higher e or more b is the same when mapped into
        # cash-on-hand a. Sorting by a therefore maintains the desired monotonicity of the consumption
        # policy with cash-on-hand, so a linear interpolation over the sorted indices is an effective way to
        # "integrate out" the egrid.
        C_as[:, is] = interp_one(vec_agrid_big_is[sorted_inds], vec(c_pol[:, is, :])[sorted_inds], agrid)
        if !any(agrid - C_as[:, is] .< -1e-16)
            if any(agrid  - C_as[:, is] .< tol)
                throw(CashOnHandError("For skill s=$(round(sgrid[is], digits = 3)), " *
                                      "max(C(a, s) - a) = $(maximum(C_as[:, is] - agrid)). Try increasing the upper bound " *
                                      "of the agrid or decreasing the number of agrid points."))
            end

            inds           = agrid .< C_as[:, is]
            C_as[inds, is] = agrid[inds]             # Require consumption policy to equal agrid
        end
    end

    return C_as
end

@inline function compute_excess(μ::AbstractVector{S}, bp::AbstractVector{S}, bg::S) where {S <: Real}
    return dot(μ, bp) - bg # Compute excess supply of savings, which is a fn of w
end

@inline function cas_euler_err(c_as::AbstractMatrix{S0},
                               β::S0, R::S0, γ::S0, T::S0, ω::S0, H::S0, na::Int, ns::Int,
                               agrid::AbstractVector{S1}, sgrid::AbstractVector{S1},
                               f::AbstractMatrix{S1}, gfunc) where {S0 <: Real, S1 <: Real}
    ell_pol_in = similar(c_as)
    agrη = repeat(agrid, 1, ns)
    euler_equation_hetdsgegovdebt!(ell_pol_in, β, 1., R, γ, T, ω, H, c_as, c_as, na, ns, (agrid[end] - agrid[1]) / na,
                                   agrid, sgrid, f, agrη, gfunc)
    # return maximum(abs.((1 ./ ell_pol_in - c_as) ./ c_as))
    return maximum(abs.((1 ./ (ell_pol_in .* c_as) .- 1.))), maximum(abs.(1 ./ ell_pol_in - c_as))
end

@inline function cbse_euler_err(cbse::AbstractArray{S, 3}, ns::Int, ne::Int, na_c::Int,
                                f::AbstractMatrix{S}, ewts::AbstractVector{S}, g_of_e::AbstractVector{S},
                                bp::AbstractVector{S}, bgrid::AbstractVector{S}, sgrid::AbstractVector{S},
                                egrid::AbstractVector{S}, β::S, R::S, γ::S, ω::S, H::S, T::S) where {S <: Real}

    c_constrained = Matrix{Float64}(undef, ns, ne)
    for ie in 1:ne
        e = egrid[ie]
        c_constrained[:, ie] = sgrid .* (ω * e * H) .+ T
    end

    cbse_out = similar(cbse)
    sum_term = zeros(size(bgrid))
    fixedpoint_cbse!(cbse_out, cbse, ns, ne, na_c, sum_term, f, ewts, g_of_e,
                     bp, bgrid, sgrid, egrid, c_constrained, β, R, γ, ω, H, T)
    return maximum(abs.(cbse_out ./ cbse .- 1.)), maximum(abs.(cbse_out - cbse))
end



function plot_steadystate_asgrid(m::HetDSGEGovDebt, agrid::AbstractVector{S}) where {S <: Real}
    na = get_setting(m, :na)
    p = plot(fit(Histogram, agrid, Weights(m[:Dstar].value[1:na]), nbins = na),
             label = "low skill", color = :blue)
    plot!(fit(Histogram, agrid, Weights(m[:Dstar].value[(na + 1):end]), nbins = na),
          label = "high skill", color = :red)
    savefig(p, "agrid_vs_D_beta=$(string(round(m[:βstar].value, digits = 3))).pdf")

    p = plot(agrid, m[:cstar].value[1:na], label = "low skill")
    plot!(p, agrid, m[:cstar].value[(na + 1):end], label = "high skill")
    savefig(p, "agrid_vs_C_beta=$(string(round(m[:βstar].value, digits = 3))).pdf")

    p = plot(agrid, agrid - m[:cstar].value[1:na], label = "low skill")
    plot!(p, agrid, agrid - m[:cstar].value[(na + 1):end], label = "high skill")
    savefig(p, "agrid_vs_Saving_beta=$(string(round(m[:βstar].value, digits = 3))).pdf")
end

function plot_steadystate_bsegrid(bgrid::AbstractVector{S}, egrid::AbstractVector{S}, c_pol_in::AbstractArray{S, 3}) where {S <: Real}
    ns = size(c_pol_in, 2)
    ne = size(c_pol_in, 3)
    for is in 1:ns
        p = plot()
        for ie in 1:ne
            plot!(bgrid, c_pol_in[:, is, ie], label = "e=$(egrid[ie])")
        end
        savefig(p, "bgrid_vs_C_varye_s=$(sgrid[is]).pdf")
    end
end
