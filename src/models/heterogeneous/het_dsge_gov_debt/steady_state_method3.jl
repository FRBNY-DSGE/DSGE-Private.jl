function steadystate!(m::HetDSGEGovDebt;
                              βlo::S = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                              βhi::S = min(exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), 0.999999),
                              excess::S = 5000., tol::S = 1e-4, maxit::Int64 = 20, βband::S = 1e-2,
                              roots_algorithm = nothing,
                              euler_anderson::Bool = true,
                              kf_anderson::Bool = true,
                              doplots::Bool = false, verbose::Symbol = :none) where {S <: Real}
    # If we have already solved for βstar (i.e. it's not NaN) and we only want to
    # estimate the non steady state parameters, there's no need to recompute
    # elo/ehi, etc.
    if !isnan(m[:βstar].value) && get_setting(m, :estimate_only_non_steady_state_parameters)
        return
    else
        reset_grids!(m)
        na = get_setting(m, :na)
        ns = get_setting(m, :ns)
        ne = get_setting(m, :ne)

        m[:sH_over_sL], m[:elo], m[:ehi] = compute_income_process_parameters(m)

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
        egrid, ewts, g_of_e = construct_egrid(m[:ehi].value, m[:elo].value, ne)

        # Run loop expanding the agrid if a CashOnHandError is caught
        ahi_guesses = if haskey(get_settings(m), :ahi_incs) # Construct guesses for the upper bound of agrid
            get_setting(m, :ahi_incs)
        elseif haskey(get_settings(m), :ahi_inc)
            [get_setting(m, :ahi_inc)]
        else
            [NaN]
        end
        m_anderson = haskey(get_settings(m), :m_anderson) ? get_setting(m, :m_anderson) : 5 # Do this calculation here
        for ahi_guess in ahi_guesses
            try
                # Construct agrid
                agrid, awts, ascale = cash_grid(sgrid, egrid, ω, H, R, η, γ, T, na; ahi_inc = ahi_guess)

                m <= Setting(:alo, agrid[1])
                m <= Setting(:ahi, agrid[end])
                m <= Setting(:ascale, ascale)

                # Once have updated grids, can call steady state and compute other two moments
                method3_find_steadystate!(m, na, ns, ne,
                                          egrid, ewts, g_of_e,
                                          f, agrid, sgrid,
                                          R, H, η, γ, ω, T, bg;
                                          βlo = βlo, βhi = βhi,
                                          excess = excess, tol = tol, maxit = maxit,
                                          βband = βband, doplots = doplots,
                                          m_anderson = m_anderson,
                                          euler_anderson = euler_anderson,
                                          kf_anderson = kf_anderson,
                                          roots_algorithm = roots_algorithm,
                                          verbose = verbose)

                if verbose == :high
                    println("The distribution μ(a, s) integrates to $(round(sum(m[:μstar].value), digits = 3)).")
                end

                if doplots
                    p = plot(fit(Histogram, agrid, Weights(m[:μstar].value[1:na]), nbins = na),
                             label = "low skill", color = :blue)
                    plot!(fit(Histogram, agrid, Weights(m[:μstar].value[(na + 1):end]), nbins = na),
                          label = "high skill", color = :red)
                    savefig(p, "method3_agrid_vs_D_beta=$(string(round(m[:βstar].value, digits = 3))).pdf")

                    p = plot(agrid, m[:cstar].value[1:na], label = "low skill")
                    plot!(p, agrid, m[:cstar].value[(na + 1):end], label = "high skill")
                    savefig(p, "method3_agrid_vs_C_beta=$(string(round(m[:βstar].value, digits = 3))).pdf")
                end

                m[:mpc] = ave_mpc(m[:μstar].value,   m[:cstar].value, agrid, kron(swts, awts), na, ns)
                m[:pc0] = frac_zero(m[:μstar].value, m[:cstar].value, agrid, kron(swts, awts), ns)

                break
            catch e
                if isa(e, CashOnHandError) && ahi_guess != ahi_guesses[end]
                    continue
                else
                    rethrow(e)
                end
            end
        end
    end

    nothing
end

function method3_find_steadystate!(m::HetDSGEGovDebt, na::Int, ns::Int, ne::Int,
                                   egrid::Vector{Float64}, ewts::Vector{Float64}, g_of_e::Vector{Float64},
                                   f::Matrix{Float64},
                                   agrid::Vector{Float64}, sgrid::Vector{Float64},
                                   R::Float64, H::Float64, η::Float64, γ::Float64, ω::Float64, T::Float64, bg::Float64;
                                   βlo::S = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                                   βhi::S = min(exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), 0.9999999),
                                   excess::S = 5000., tol::S = 1e-4, maxit::Int64 = 20, βband::S = 1e-2,
                                   euler_anderson::Bool = false, kf_anderson::Bool = false,
                                   m_anderson::Int, roots_algorithm = nothing,
                                   doplots::Bool = false, verbose::Symbol = :high) where {S <: Real}

    na_c = get_setting(m, :na_c)

    βlo_temp = βlo
    βhi_temp = βhi

    reject  = false
    counter = 0

    # Initial guess
    β        = NaN # Need to define β here so it's accessible outside of while lopp
    c_pol_in = (R - 1) * repeat(agrid, 1, ns, ne) .+ ω * H * repeat(sgrid', na, 1, ne)
    KF_in    = fill(1.0 / (ns * na * ne), na, ns, ne)

    # If want to keep the last β/don't recompute
    if get_setting(m, :use_last_βstar) && !isnan(m[:βstar].value)

        # This short-circuits computation of the policy function
        βlo = βhi = m[:βstar].value
    # If you have a β saved, start from there (if you don't, βhi and βlo are defined as they are passed in to function)
    elseif !isnan(m[:βstar].value)

        # If one has computed β* before, we first bisect into a neighborhood around it
        βlo_temp = m[:βstar].value - βband
        βhi_temp = m[:βstar].value + βband

        c_pol_in, bp, KF, reject = method3_policy_hetdsgegovdebt(na, ns, ne, na_c, βlo_temp, R, ω, H, η, T, γ,
                                                                 m[:ehi].value, m[:elo].value, agrid, sgrid, c_pol_in, KF_in,
                                                                 f, egrid, ewts, g_of_e,
                                                                 maxit = get_setting(m, :policy_maxit),
                                                                 m_anderson = m_anderson,
                                                                 euler_anderson = euler_anderson,
                                                                 kf_anderson = kf_anderson,
                                                                 euler_tol = get_setting(m, :euler_tol),
                                                                 kf_tol = get_setting(m, :kf_tol))

        excess_lo = compute_excess(KF, bp, bg)

        if excess_lo < 0 && abs(excess_lo) > tol
            βlo = βlo_temp
            c_pol_in, bp, KF, reject = method3_policy_hetdsgegovdebt(na, ns, ne, na_c, βhi_temp, R, ω, H, η, T, γ, m[:ehi].value,
                                                                     m[:elo].value, agrid, sgrid, c_pol_in, KF_in,
                                                                     f, egrid, ewts, g_of_e,
                                                                     maxit = get_setting(m, :policy_maxit),
                                                                     m_anderson = m_anderson,
                                                                     euler_anderson = euler_anderson,
                                                                     kf_anderson = kf_anderson,
                                                                     euler_tol = get_setting(m, :euler_tol),
                                                                     kf_tol = get_setting(m, :kf_tol))

            excess_hi = compute_excess(KF, bp, bg)

            if excess_hi > 0
                βhi = βhi_temp
            end
        elseif excess_lo >= 0
            βhi = βlo_temp
        end
    end

    # If you don't have a β, guess a β
    if isnothing(roots_algorithm)
        while abs(excess) > tol && counter <= maxit # clearing markets
            β = (βlo + βhi) / 2.0

            c_pol_in, bp, KF, reject = method3_policy_hetdsgegovdebt(na, ns, ne, na_c, β, R, ω, H, η, T, γ,
                                                                     m[:ehi].value, m[:elo].value,
                                                                     agrid, sgrid, c_pol_in, KF_in,
                                                                     f, egrid, ewts, g_of_e,
                                                                     maxit = get_setting(m, :policy_maxit),
                                                                     m_anderson = m_anderson,
                                                                     euler_anderson = euler_anderson,
                                                                     kf_anderson = kf_anderson,
                                                                     euler_tol = get_setting(m, :euler_tol),
                                                                     kf_tol = get_setting(m, :kf_tol))

            excess = compute_excess(KF, bp, bg)
            if verbose == :high
                println("On (i, β) = ($(counter), $(round(β, digits = 6))), excess bonds are $(excess).")
            end

            # bisection
            if excess > 0
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
            @warn "Euler iteration does not converge"
            reject = true
        end
    elseif isa(roots_algorithm, AbstractBracketing)
        β = find_zero(β -> bisect_β(β, na, ns, ne, na_c, R, ω, H, η, T, γ, m[:ehi].value, m[:elo].value, agrid, sgrid, c_pol_in,
                                    KF_in, f, egrid, ewts, g_of_e, bg, get_setting(m, :policy_maxit), m_anderson,
                                    euler_anderson, kf_anderson, get_setting(m, :euler_tol), get_setting(m, :kf_tol)),
                      (βlo, βhi), roots_algorithm, maxevals = maxit, atol = tol)
        c_pol_in, bp, KF, reject = method3_policy_hetdsgegovdebt(na, ns, ne, na_c, β, R, ω, H, η, T, γ, m[:ehi].value, m[:elo].value,
                                                                 agrid, sgrid, c_pol_in, KF_in,
                                                                 f, egrid, ewts, g_of_e,
                                                                 maxit = get_setting(m, :policy_maxit),
                                                                 m_anderson = m_anderson,
                                                                 euler_anderson = euler_anderson,
                                                                 kf_anderson = kf_anderson,
                                                                 euler_tol = get_setting(m, :euler_tol),
                                                                 kf_tol = get_setting(m, :kf_tol))
    elseif isa(roots_algorithm, AbstractSecant)
        β = find_zero(β -> bisect_β(β, na, ns, ne, na_c, R, ω, H, η, T, γ, m[:ehi].value, m[:elo].value, agrid, sgrid, c_pol_in,
                                    KF_in, f, egrid, ewts, g_of_e, bg, get_setting(m, :policy_maxit), m_anderson,
                                    euler_anderson, kf_anderson, get_setting(m, :euler_tol), get_setting(m, :kf_tol)),
                      (βlo + βhi) / 2., roots_algorithm, maxevals = maxit, atol = tol)
        c_pol_in, bp, KF, reject = method3_policy_hetdsgegovdebt(na, ns, ne, na_c, β, R, ω, H, η, T, γ, m[:ehi].value, m[:elo].value,
                                                                 agrid, sgrid, c_pol_in, KF_in,
                                                                 f, egrid, ewts, g_of_e,
                                                                 maxit = get_setting(m, :policy_maxit),
                                                                 m_anderson = m_anderson,
                                                                 euler_anderson = euler_anderson,
                                                                 kf_anderson = kf_anderson,
                                                                 euler_tol = get_setting(m, :euler_tol),
                                                                 kf_tol = get_setting(m, :kf_tol))
    else
        error("Cannot use the Roots algorithm $(typeof(roots_algorithm))")
    end

    # Integrate out the e dimension
    bgrid, agrid_big = construct_bgrid_agrid_big(agrid, sgrid, egrid, na, ns, ne, γ, ω, H, T)
    if doplots
        for is in 1:ns
            p = plot()
            for ie in 1:ne
                plot!(bgrid, c_pol_in[:, is, ie], label = "e=$(egrid[ie])")
            end
            savefig(p, "method3_bgrid_vs_C_varye_s=$(sgrid[is]).pdf")
        end

        for is in 1:ns
            p = plot()
            for ie in 1:ne
                plot!(fit(Histogram, bgrid, Weights(KF[:, is, ie]), nbins = na), label = "e=$(egrid[ie])")
            end

            savefig(p, "method3_bgrid_vs_D_varye_s=$(sgrid[is]).pdf")
        end
    end
    C_Final, KF_Final, ell = integrate_out_e(agrid, agrid_big, bgrid, sgrid, c_pol_in, KF, ω, H, T, γ;
                                             tol = get_setting(m, :C_tol))

    # If policy function does not converge, we signal to likelihood that should reject
    m <= Setting(:auto_reject, reject)
    m[:lstar]  = ell
    m[:cstar]  = C_Final
    m[:μstar]  = KF_Final
    m[:βstar]  = β
    m[:β_save] = β

    nothing
end

function bisect_β(β::S, na::Int, ns::Int, ne::Int, na_c::Int, R::S, ω::S, H::S, η::S, T::S, γ::S,
                  ehi::S, elo::S, agrid::AbstractVector{S}, sgrid::AbstractVector{S},
                  c_pol_in::AbstractArray{S, 3}, KF_in::AbstractArray{S, 3}, f::AbstractMatrix{S},
                  egrid::AbstractVector{S}, ewts::AbstractVector{S}, g_of_e::AbstractVector{S}, bg::S,
                  maxit::Int, m_anderson::Int, euler_anderson::Bool, kf_anderson::Bool, euler_tol::S, kf_tol::S) where {S <: Real}
    c_pol_in, bp, KF, reject = method3_policy_hetdsgegovdebt(na, ns, ne, na_c, β, R, ω, H, η, T, γ, ehi, elo,
                                                             agrid, sgrid, c_pol_in, KF_in,
                                                             f, egrid, ewts, g_of_e,
                                                             maxit = maxit,
                                                             m_anderson = m_anderson,
                                                             euler_anderson = euler_anderson,
                                                             kf_anderson = kf_anderson,
                                                             euler_tol = euler_tol,
                                                             kf_tol = kf_tol)
    excess = compute_excess(KF, bp, bg)
    return excess
end

function method3_policy_hetdsgegovdebt(na::Int, ns::Int, ne::Int, na_c::Int, β::S, R::S, ω::S, H::S, η::S,
                                       T::S, γ::S, ehi::S, elo::S, agrid::Vector{S},
                                       sgrid::Vector{S}, c_pol::Array{S}, KF_in::Array{S},
                                       f::Matrix{S}, egrid::Vector{Float64}, ewts::Vector{Float64},
                                       g_of_e::Vector{Float64}, dist::S = 1.; maxit::Int64 = 1000,
                                       m_anderson::Int = 5,
                                       euler_anderson::Bool = false, kf_anderson::Bool = false,
                                       euler_tol::S = 1e-10, kf_tol::S = 1e-10) where {S <: Real}

    colors = [:red, :blue, :green, :yellow, :purple]
    counter = 0
    reject = false

    # Constrainted consumption
    c_constrained = Matrix{Float64}(undef, ns, ne)
    for ie in 1:ne
        e = egrid[ie]
        c_constrained[:, ie] = sgrid .* (ω * e * H) .+ T
    end

    bgrid, agrid_big = construct_bgrid_agrid_big(agrid, sgrid, egrid, na, ns, ne, γ, ω, H, T)

    # Allocate memory here
    bp       = bgrid # bp = b' = bprime, only to make it clear that we're working with b', not b
    sum_term = zeros(S, length(bp))

    # Fixed point problem for c_poli
    if euler_anderson
        out = nlsolve((F_c_pol, c_pol) -> fixedpoint_c_policy_nlsolve!(F_c_pol, c_pol, ns, ne, na_c, sum_term, f, ewts, g_of_e,
                                                                       bp, bgrid, sgrid, egrid, c_constrained, β, R, γ, ω, H, T),
                      c_pol, ftol = euler_tol, iterations = maxit, m = m_anderson, method = :anderson)
        if !out.f_converged
            @warn "Euler iteration did not converge. The final distance is $(out.residual_norm)"
            reject = true
        end
        c_pol = out.zero
    else
        c_poli = similar(c_pol)
        while dist > euler_tol && counter <= maxit
            fixedpoint_c_policy!(c_poli, c_pol, ns, ne, na_c, sum_term, f, ewts, g_of_e,
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

    # Construct weights for calculating distribution over (b, s, e), see the KF loop to undersatnd how these Arrays are used
    bpgrid_big, ibp_pol, wts_big = construct_KF_weights(c_pol, bgrid, sgrid, egrid, na, ns, ne, R, ω, H, T, γ)

    # Iterate asset transition matrix starting from KF_in
    dif = 1
    pd  = deepcopy(KF_in) # D(b, s, e)
    pdi = similar(pd)     # D'(b', s', e')
    if kf_anderson
        out = nlsolve((F_pd, pd) -> fixedpoint_KF_nlsolve!(F_pd, pd, na, ns, ne, ibp_pol, wts_big, ewts, g_of_e, f),
                      pd, ftol = kf_tol, iterations = maxit, method = :anderson, m = m_anderson)
        pd  = out.zero
    else
        counter = 0
        while dif > kf_tol && counter <= maxit
            fixedpoint_KF!(pdi, pd, na, ns, ne, ibp_pol, wts_big, ewts, g_of_e, f)

            # check convergence
            dif      = maximum(abs.(pdi - pd))
            pd      .= pdi
            counter += 1
        end
    end

    return c_pol, bgrid, pd, reject
end

function fixedpoint_c_policy!(c_poli::AbstractArray{S, 3}, c_pol::AbstractArray{S, 3}, ns::Int, ne::Int, na_c::Int,
                              sum_term::AbstractVector{S}, f::AbstractMatrix{S},
                              ewts::AbstractVector{S}, g_of_e::AbstractVector{S},
                              bp::AbstractVector{S}, bgrid::AbstractVector{S}, sgrid::AbstractVector{S},
                              egrid::AbstractVector{S}, c_constrained::AbstractMatrix{S},
                              β::S, R::S, γ::S, ω::S, H::S, T::S) where {S <: Real}
    for is in 1:ns
        for ie in 1:ne
            # Sums
            verify_one = 0.0
            for iep in 1:ne     # Inner integral over e' (done first b/c Julia is column major)
                for isp in 1:ns # Outer sum over s'
                    #               p(s'|s)    * iota(e')  *      g(e')    * c(a, s')^{-1}
                    # Note that f[is, isp] is prob going froms sgrid[is] to sgrid[isp], so we do want to
                    # iterate over the second element rather than the first, despite the p(s'|s) notation
                    sum_term   .+= (f[is, isp] * ewts[iep] * g_of_e[iep]) ./ c_pol[:, isp, iep]
                end
            end

            # Compute ell(a, s) = β * R * exp(-γ) * (Σₛ∫ₑ)
            # l = β * R * exp(-γ) * sum_term
            # Compute consumption today: c(b', s) = 1/l(a, s) and then
            # assets today: b = exp(γ) * (b' / R - w * s * e * H - T + c)
            c = 1 ./ ((β * R * exp(-γ)) .* sum_term)
            b = vec(exp(γ) * ((bp ./ R) .- ω * sgrid[is] * egrid[ie] * H .- T .+ c))
            sum_term .= 0. # Reset the values to zero (done with it for this loop)
            if all(b .< 0.)
                throw(CashOnHandError("All elements of b (assets today) are negative. " *
                                      "The last element is b=$(round(b[end], digits = 3)). " *
                                      "Try increasing the upper bound of the agrid, increasing " *
                                      "the number of agrid points, and/or lowering β"))
            end

            # Handle constrained consumption today: for b < 0, need to reset c such that b is exactly 0. See line 254
            c[b .< 0.] = -(bp[b .< 0] ./ R) .+  ω * sgrid[is] * egrid[ie] * H .+ T
            b         .= vec(exp(γ) * ((bp ./ R) .- ω * sgrid[is] * egrid[ie] * H .- T .+ c))

            # If b[1] is positive, then people aren't using a c(0, s, e) policy but also must have some spare b
            # that they additionally consume. This code block assumes "relative" monotonicity of c
            if b[1] > 0.
                if c[1] < c_constrained[is, ie]
                    # Check consuming more than implied by constrained rule if b[1] > 0
                    throw(CashOnHandError("c[1] - c_constrained[is, ie]=$(c[1] - c_constrained[is, ie]), " *
                                          "try increasing the upper bound of the agrid " *
                                          "or the number of agrid points."))
                end
                c_c = collect(range(c_constrained[is, ie], c[1], length = na_c)) # constrained rule is an equal spacing
                b_c = exp(γ) * (-ω * sgrid[is] * egrid[ie] * H - T .+ c_c)       # between no-assets rule and c[1]
                if -1e-14 < b_c[1] < 0.
                    b_c[1] = 0. # Force the first point to be zero (in case of floating point errors)
                end
                if any(b_c .< -1e-14)
                    throw(CashOnHandError("Some assets today for constrained agents are negative."))
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

function fixedpoint_c_policy_nlsolve!(F_c_pol::AbstractArray{S, 3}, c_pol::AbstractArray{S, 3}, ns::Int, ne::Int, na_c::Int,
                                      sum_term::AbstractVector{S}, f::AbstractMatrix{S},
                                      ewts::AbstractVector{S}, g_of_e::AbstractVector{S},
                                      bp::AbstractVector{S}, bgrid::AbstractVector{S}, sgrid::AbstractVector{S},
                                      egrid::AbstractVector{S}, c_constrained::AbstractMatrix{S},
                                      β::S, R::S, γ::S, ω::S, H::S, T::S) where {S <: Real}
    for is in 1:ns
        for ie in 1:ne
            # Sums
            verify_one = 0.0
            for iep in 1:ne     # Inner integral over e' (done first b/c Julia is column major)
                for isp in 1:ns # Outer sum over s'
                    #               p(s'|s)    * iota(e')  *      g(e')    * c(a, s')^{-1}
                    # Note that f[is, isp] is prob going froms sgrid[is] to sgrid[isp], so we do want to
                    # iterate over the second element rather than the first, despite the p(s'|s) notation
                    sum_term   .+= (f[is, isp] * ewts[iep] * g_of_e[iep]) ./ c_pol[:, isp, iep]
                    verify_one  +=  f[is, isp] * ewts[iep] * g_of_e[iep]
                end
            end
            @test isapprox(verify_one, 1.0, atol = 1e-5)

            # Compute ell(a, s) = β * R * exp(-γ) * (Σₛ∫ₑ)
            # l = β * R * exp(-γ) * sum_term
            # Compute consumption today: c(b', s) = 1/l(a, s) and then
            # assets today: b = exp(γ) * (b' / R - w * s * e * H - T + c)
            c = 1 ./ ((β * R * exp(-γ)) .* sum_term)
            b = vec(exp(γ) * ((bp ./ R) .- ω * sgrid[is] * egrid[ie] * H .- T .+ c))
            sum_term .= 0. # Reset the values to zero (done with it for this loop)
            if all(b .< 0.)
                throw(CashOnHandError("All elements of b (assets today) are negative. " *
                                      "The last element is b=$(round(b[end], digits = 3)). " *
                                      "Try increasing the upper bound of agrid, increasing " *
                                      "the number of agrid points, and/or lowering β"))
            end

            # Handle constrained consumption today: for b < 0, need to reset c such that b is exactly 0. See line 254
            c[b .< 0.] = -(bp[b .< 0] ./ R) .+  ω * sgrid[is] * egrid[ie] * H .+ T
            b         .= vec(exp(γ) * ((bp ./ R) .- ω * sgrid[is] * egrid[ie] * H .- T .+ c))

            # If b[1] is positive, then people aren't using a c(0, s, e) policy but also must have some spare b
            # that they additionally consume. This code block assumes "relative" monotonicity of c
            if b[1] > 0.
                if c[1] < c_constrained[is, ie]
                    # Check consuming more than implied by constrained rule if b[1] > 0
                    throw(CashOnHandError("c[1] - c_constrained[is, ie]=$(c[1] - c_constrained[is, ie]), " *
                                          "try increasing the upper bound of the agrid " *
                                          "or the number of agrid points."))
                end
                c_c = collect(range(c_constrained[is, ie], c[1], length = na_c)) # constrained rule is an equal spacing
                b_c = exp(γ) * (-ω * sgrid[is] * egrid[ie] * H - T .+ c_c)       # between no-assets rule and c[1]
                if -1e-10 < b_c[1] < 0.
                    b_c[1] = 0. # Force the first point to be zero (in case of floating point errors)
                end
                if any(b_c .< -1e-10)
                    throw(CashOnHandError("Some assets today for constrained agents are negative."))
                end
                cutoff = findlast(b_c .< b[1])
                b = vcat(b_c[1:cutoff], b) # Add additional points for constrained people
                c = vcat(c_c[1:cutoff], c) # so that, for the new b, the condition b[1] ≈ 0 holds
            end

            # Nearest points, linear interpolation
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

function construct_KF_weights(c_pol::AbstractArray{S, 3}, bgrid::AbstractVector{S}, sgrid::AbstractVector{S},
                              egrid::AbstractVector{S}, na::Int, ns::Int, ne::Int,
                              R::S, ω::S, H::S, T::S, γ::S) where {S <: Real}
    bpgrid_big = Array{S}(undef, na, ns, ne)   # bp implied by consumption policy
    ibp_pol    = Array{Int}(undef, na, ns, ne) # index of bgrᵢ bgrid node, where bgrᵢ <= bp <= bgrᵢ₊₁
    wts_big    = Array{S}(undef, na, ns, ne)   # Weight for bp for allocating mass to bgrᵢ and bgrᵢ₊₁
    for ie in 1:ne
        for is in 1:ns
            bpgrid_big[:, is, ie] = R .* ((ω * egrid[ie] * sgrid[is] * H + T) .+ exp(-γ) .* bgrid .- c_pol[:, is, ie])
            for ia in 1:na
                idx = findlast(bpgrid_big[ia, is, ie] .>= bgrid)
                if isnothing(idx) # hot fix, assign mass to boundary
                    ibp_pol[ia, is, ie] = 1
                    wts_big[ia, is, ie] = 1. #(bmin - bpgrid_big[ia, is, ie]) / (bgrid[2] - bmin)
                elseif idx == na  # hot fix, assign mass to boundary
                    ibp_pol[ia, is, ie] = na - 1
                    wts_big[ia, is, ie] = 0. #(bpgrid_big[ia, is, ie] - bmax) / (bmax - bgrid[end - 1])
                else
                    ibp_pol[ia, is, ie] = idx
                    wts_big[ia, is, ie] = (bpgrid_big[ia, is, ie] - bgrid[idx]) / (bgrid[idx + 1] - bgrid[idx])
                end
            end
        end
    end

    return bpgrid_big, ibp_pol, wts_big
end


function fixedpoint_KF!(pdi::AbstractArray{S, 3}, pd::AbstractArray{S, 3}, na::Int, ns::Int, ne::Int,
                        ibp_pol::AbstractArray{Int, 3}, wts_big::AbstractArray{S, 3},
                        ewts::AbstractVector{S}, g_of_e::AbstractVector{S}, f::AbstractMatrix{S}) where {S <: Real}
    pdi .= 0.

    for ie in 1:ne
        for is = 1:ns
            for iep = 1:ne
                for isp = 1:ns
                    for ib = 1:na
                        ibp = ibp_pol[ib, is, ie] # Given (b, s, e), get index of bgrᵢ to approximate bp

                        # True evolution: D'(b'(b), s', e') += ewts(e') * g(e') * p(s'|s) * D(b, s, e), but
                        # have to interpolate from b'(b) to b, so we distribute mass to bgrᵢ <= b'(b) <= bgrᵢ₊₁

                        # D'(bgrᵢ, s', e'  ) += wt(b, s, e)     * ewts(e') * g(e') * p(s'|s) * D(b, s, e)
                        pdi[ibp, isp, iep]     += wts_big[ib, is, ie] *
                            ewts[iep] * g_of_e[iep] * f[is, isp] * pd[ib, is, ie] # recall, f = prob going to isp from is

                        # D'(bgrᵢ₊₁, s', e') += (1-wt(b, s, e)) * ewts(e') * g(e') * p(s'|s) * D(b, s, e)
                        pdi[ibp + 1, isp, iep] += (1 - wts_big[ib, is, ie]) *
                            ewts[iep] * g_of_e[iep] * f[is, isp] * pd[ib, is, ie]
                    end
                end
            end
        end
    end

    return pdi
end

function fixedpoint_KF_nlsolve!(F_pd::AbstractArray{S, 3}, pd::AbstractArray{S, 3}, na::Int, ns::Int, ne::Int,
                                ibp_pol::AbstractArray{Int, 3}, wts_big::AbstractArray{S, 3},
                                ewts::AbstractVector{S}, g_of_e::AbstractVector{S}, f::AbstractMatrix{S}) where {S <: Real}
    # Zero out
    F_pd .= 0.

    # Add in place to F_pd ⇒ D' = F_pd
    for ie in 1:ne
        for is = 1:ns
            for iep = 1:ne
                for isp = 1:ns
                    for ib = 1:na
                        ibp = ibp_pol[ib, is, ie] # Given (b, s, e), get index of bgrᵢ to approximate bp

                        F_pd[ibp, isp, iep]     += wts_big[ib, is, ie] *
                            ewts[iep] * g_of_e[iep] * f[is, isp] * pd[ib, is, ie]
                        F_pd[ibp + 1, isp, iep] += (1 - wts_big[ib, is, ie]) *
                            ewts[iep] * g_of_e[iep] * f[is, isp] * pd[ib, is, ie]
                    end
                end
            end
        end
    end

    # Subtract D from F_pd in place ⇒ F_pd = D' - D
    F_pd .-= pd
end

# KF is D(a, s, e), bp is the bprime grid, and bg is the government's supply of bonds
@inline function compute_excess(KF::AbstractArray{S, 3}, bp::AbstractVector{S}, bg::S) where {S <: Real}
    # Return ∑ᵢ ∑ₛ ∑ₑ D(bᵢ, s, e) bᵢ - β
    return dot(sum(KF, dims = (2, 3)), bp) - bg # Summing over (s, e) first speed things up, fewer multiplications and can use dot
end

function transform_ab(a::S, b::S, grid::AbstractVector{S}) where {S <: Real}
    xs = ((b-a)/2) .* grid .+ (a+b)/2
    return xs
end

# Maps c(b, s, e) -> c(a, s) and D(b, s, e) -> D(a, s)
function integrate_out_e(agrid::AbstractVector{S}, agrid_big::AbstractArray{S, 3},
                         bgrid::AbstractVector{S}, sgrid::AbstractVector{S}, c_pol::AbstractArray{S, 3}, D_bse::AbstractArray{S, 3},
                         ω::S, H::S, T::S, γ::S; tol::Float64 = -1e-8) where {S <: Real}
    # Map c(b, s, e) -> c(a, s, e)
    na, ns, ne = size(agrid_big)
    C_Final    = Matrix{S}(undef, na, ns)
    for is in 1:ns
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
        C_Final[:, is] = interp_one(vec_agrid_big_is[sorted_inds], vec(c_pol[:, is, :])[sorted_inds], agrid)
        if !any(agrid - C_Final[:, is] .< -1e-16)
            if any(agrid  - C_Final[:, is] .< -1e-14)
                throw(CashOnHandError("For skill s=$(round(sgrid[is], digits = 3)), " *
                                      "max(C(a, s) - a) = $(maximum(C_Final[:, is] - agrid)). Try increasing the upper bound " *
                                      "of the agrid or decreasing the number of agrid points."))
            end

            inds              = agrid .< C_Final[:, is]
            C_Final[inds, is] = agrid[inds]             # Require consumption policy to equal agrid
        end
    end

    #  D(b, s, e) -> D(a, s)
    D_as = zeros(S, na, ns)
    for ie in 1:ne
        for is in 1:ns
            for ia in 1:na
                ind = findlast(agrid_big[ia, is, ie] .>= agrid) # Find i s.t. agridᵢ <= agrid_big[ia, is, ie] <= agridᵢ₊₁
                if isnothing(ind) # Just add the remaining weight onto agrid₁
                    ind = 1
                    weight = 1.
                elseif ind == na # Just add the remaining weight onto agridₙ, where n is the last point
                    ind = na - 1
                    weight = 0.
                else
                    weight = (agrid_big[ia, is, ie] - agrid[ind]) / (agrid[ind + 1] - agrid[ind])
                end

                # Note agrid_big = a(bgrid, sgrid, egrid) ⇒ D_as is defined over an (agrid, sgrid) mesh
                D_as[ind, is]     += weight       * D_bse[ia, is, ie]
                D_as[ind + 1, is] += (1 - weight) * D_bse[ia, is, ie]
            end
        end
    end

    ell = 1 ./ vec(C_Final)

    return vec(C_Final), vec(D_as), ell
end


@inline function mollifier_hetdsgegovdebt(e::S, ehi::S, elo::S) where {S <: Real}
    In = 0.443993816237631
    if e<ehi && e>elo
        temp = -1.0 + 2.0 * (e - elo) / (ehi - elo)
        return (2.0 / (ehi - elo)) * exp(-1.0 / (1.0 - temp^2)) / In
    end
    return 0.0
end

@inline function dmollifier_hetdsgegovdebt(x::S, ehi::S, elo::S) where {S <: Real}
    In = 0.443993816237631
    if x<ehi && x>elo
        temp = (-1.0 + 2.0*(x-elo)/(ehi-elo))
        out  = -(2*temp./((1 - temp.^2).^2)).*(2/(ehi-elo)) .*
            mollifier_hetdsgegovdebt(x, ehi, elo)
    else
        out = 0.0
    end
    return out
end

function calibrate_pLH_pHL(m::HetDSGEGovDebt)
    target_mpc, target_pc0 = get_setting(m, :targets)
    σt_mpc, σt_pc0         = get_setting(m, :target_σt)

    function minimize_penalty(m, pLHpHL::Vector{Float64})
        m[:pLH] = pLHpHL[1]
        m[:pHL] = pLHpHL[2]
        steadystate!(m)
        if get_setting(m, :auto_reject)
            m <= Setting(:auto_reject, false)
            @show pLHpHL, Inf
            return Inf
        end
        @show pLHpHL, m[:mpc].value, m[:pc0].value
        println(0.5*((log(m[:mpc])-log(target_mpc))^2/σt_mpc +
                    (log(m[:pc0].value)-log(target_pc0))^2/σt_pc0^2))
        return 0.5*((log(m[:mpc])-log(target_mpc))^2/σt_mpc +
                    (log(m[:pc0].value)-log(target_pc0))^2/σt_pc0^2)
    end
    minimize_me(x) = minimize_penalty(m, x)
    res = optimize(minimize_me, [0.005, 0.005], [0.095, 0.095], [0.01125, 0.03],
                   Fminbox(Optim.NelderMead()), Optim.Options(f_calls_limit = 300))
    println(res)
    return res
end

function ave_mpc(m::AbstractArray, c::AbstractArray, agrid::AbstractArray,
                 aswts::AbstractArray, na::Int, ns::Int)
	mpc = 0.
	for iss=1:ns
		i = na*(iss-1)+1
		mpc += aswts[i]*m[i]*(c[i+1] - c[i])/(agrid[2] - agrid[1])
		for ia=2:na
			i = na*(iss-1)+ia
			mpc += aswts[i]*m[i]*(c[i] - c[i-1])/(agrid[ia] - agrid[ia-1])
		end
	end
	return mpc
end

function frac_zero(m::AbstractArray, c::AbstractArray, agrid::AbstractArray,
                   aswts::AbstractArray, ns::Int)
	return sum(aswts .* m .* (c .== repeat(agrid, ns)))
end

function ssample(us::Matrix{S}, P::Matrix{S}, πss::AbstractArray,
                 ni::Int) where {S <: Real}
	shist = ones(Int,ni,8)
	for i=1:ni
		if us[i,1] > πss[1]
			shist[i,1] = 2
		end
		for t=2:8
			if us[i,t] > P[shist[i,t-1],1]
				shist[i,t] = 2
			end
		end
	end
	return shist
end

function ln_annual_inc(ehist::Matrix{S}, us::Matrix{S}, elo::S, P::Matrix{S},
                       πss::AbstractArray, sgrid::AbstractArray,
                       ni::Int) where {S <: Real}
  	s_inds = ssample(us,P,πss,ni)
	linc1 = zeros(ni)
	linc2 = zeros(ni)
	for i=1:ni
		inc1 = 0.
		for t=1:4
			eshock = 1. + (1. - elo)*(ehist[i,t]-1.)
			sshock = (sgrid[1] + (sgrid[2] - sgrid[1])*(s_inds[i,t] - 1.))
			inc1 += eshock*sshock
		end
		inc2 = 0.
		for t=5:8
			eshock = 1. + (1. - elo)*(ehist[i,t]-1.)
			sshock = (sgrid[1] + (sgrid[2] - sgrid[1])*(s_inds[i,t] - 1.))
			inc2 += eshock*sshock
		end
		linc1[i] += log(inc1)
     	linc2[i] += log(inc2)
	end
	return linc1, linc2
end

function skill_moments(sH_over_sL::Real, elo::Real, pLH::S, pHL::S, us::Matrix{S},
                       es::Matrix{S}, ni::Int = 10000) where {S <: Real}
	πL    = pHL / (pLH + pHL)
	πss   = [πL; 1.0-πL]
	P     = [[1.0-pLH pLH]; [pHL 1.0-pHL]]
	slo   = 1.0 / (πL + (1-πL) * sH_over_sL)
	shi   = sH_over_sL * slo
	sgrid = [slo; shi]
	linc1, linc2 = ln_annual_inc(es, us, elo, P, πss, sgrid, ni)
	return var(linc1), var(linc2 - linc1)
end

"""
```
function loss(x::Vector{S}, target::Vector{S}) where {S <: Real}
```

Computes for given vector and target vector, the sum of absolute loss.
"""
loss(x::Vector{S}, target::Vector{S}) where {S <: Real} = sum(abs.(x-target))

"""
```
function best_fit(pLH::S, pHL::S, target::Vector{S}, lower::Vector{S}, upper::Vector{S},
                  us::Matrix{S}, es::Matrix{S}, max_iter::Int = 20,
                  initial_guess::Vector{S} = [6.3, 0.03]) where {S <: Real}
```

Uses Nelder-Mead (gradient descent algorithm) to optimize for income moments.
"""
function best_fit(pLH::S, pHL::S, target::Vector{S}, lower::Vector{S}, upper::Vector{S},
                  us::Matrix{S}, es::Matrix{S}, max_iter::Int = 20,
                  initial_guess::Vector{S} = [6.3, 0.03]) where {S <: Real}

    skill_moments_f(x) = loss(collect(skill_moments(x[1], x[2], pLH, pHL, us, es)), target)

    res = optimize(skill_moments_f, lower, upper, initial_guess, Fminbox(NelderMead()),
                   Optim.Options(f_calls_limit = max_iter))

    sH_over_sL_argmin, elo_argmin = Optim.minimizer(res)
    min_varlinc, min_vardlinc = skill_moments(sH_over_sL_argmin, elo_argmin, pLH,
                                              pHL, us, es)
    return sH_over_sL_argmin, elo_argmin, min_varlinc, min_vardlinc
end

function compute_income_process_parameters(m::AbstractDSGEModel)
    # Determines whether random or not
    us, es = if get_setting(m, :fix_random_matrices)
        get_setting(m, :us), get_setting(m, :es)
    else
        generate_us_and_es(ni, ne)
    end

    target = get_setting(m, :calibration_targets)
    lower  = get_setting(m, :calibration_targets_lb)
    upper  = get_setting(m, :calibration_targets_ub)

    sH_over_sL, elo, _, _ = best_fit(m[:pLH].value, m[:pHL].value,
                                     target, lower, upper, us, es)
    ehi = 2.0 - m[:elo].value
    return sH_over_sL, elo, ehi
end

function esample(ue::Matrix{S}, egrid::AbstractArray, ecdf::AbstractArray,
                 ni::Int, ne::Int) where {S <: Real}
    eave = 0.5*egrid[1:ne-1]+0.5*egrid[2:ne]
	es = zeros(ni,8)
	for i=1:ni
		for t=1:8
			for ie=1:ne-1
				if ecdf[ie] < ue[i,t] <= ecdf[ie+1]
					es[i,t] = eave[ie]
				end
			end
		end
	end
	return es
end

function construct_sgrid(pHL::S, pLH::S, sH_over_sL::S, ns::Int) where {S <: Real}
    ss_skill_distr = [pHL / (pLH + pHL); pLH / (pLH + pHL)]
    slo            = 1.0 / (ss_skill_distr' * [1; sH_over_sL])
    sgrid          = slo * [1; sH_over_sL]
    sscale         = sgrid[2] - sgrid[1]
    swts           = fill(sscale / ns, ns) # Quadrature weights

    return sgrid, swts, sscale
end

function construct_egrid(ehi::S, elo::S, ne::Int) where {S <: Real}
    # Construct egrid
    egrid, ewts = gausslegendre(ne)
    egrid      .= transform_ab(elo, ehi, egrid)

    # Normalize egrid, ewts so that ∫ g(e) de ≈ ∑ᵢ g(eᵢ) * wᵢ = 1
    g_of_e      = map(x -> mollifier_hetdsgegovdebt(x, ehi, elo), egrid) # g(eᵢ)
    ewts      ./= dot(g_of_e, ewts) # normalize wᵢ to w̃ᵢ so that ∑ᵢ w̃ᵢ * gᵢ = 1
    egrid     ./= dot(g_of_e .* egrid, ewts) # Normalize egrid so that mean ∫ e g(e) de = 1, should be a small adjustment

    return egrid, ewts, g_of_e
end

function cash_grid(sgrid::AbstractVector{S}, egrid::AbstractVector{S},
                   ω::S, H::S, R::S, η::S, γ::S, T::S, na::Int;
                   ahi_inc::S = NaN) where {S <: Real}
    smin = minimum(sgrid)                            # lowest possible skill
    emin = minimum(egrid)                            # lowest possible realization of idiosyncratic shock
    alo  = ω * smin * emin * H - R * η * exp(-γ) + T # lowest SS possible cash on hand
    ahi  = if isnan(ahi_inc)
        ahi = max(alo * 2., alo + 20.0)              # upper bound on cash on hand
    else
        ahi = alo + ahi_inc
    end

    ascale = (ahi - alo)                                  # size of w grids
    agrid  = collect(range(alo, stop = ahi, length = na)) # Evenly spaced grid
    awts   = fill(ascale / na, na)                        # Quadrature weights, sum up to 12

    return agrid, awts, ascale
end

function construct_bgrid_agrid_big(agrid::AbstractVector{S}, sgrid::AbstractVector{S}, egrid::AbstractVector{S},
                                   na::Int, ns::Int, ne::Int, γ::S, ω::S, H::S, T::S) where {S <: Real}

    bmin = 0.0
    bmax = exp(γ) * (maximum(agrid)- ω * maximum(sgrid) * maximum(egrid) * H) # NOTE egrid is not exactly in [elo, ehi] b/c
    bgrid = bmin .+ ((1:na) / na).^2 * (bmax - bmin)                          # slightly renormalized to ensure ∫e g(e) de = 1

    # a grid (cash on hand) implied by bgrid (assets) is: map b into a using equation at top of page 4
    agrid_big = Array{Float64}(undef, na, ns, ne)
    for ie in 1:ne
        for is in 1:ns
            agrid_big[:, is, ie] = (ω * sgrid[is] * egrid[ie] * H + T) .+ exp(-γ) .* bgrid
        end
    end

    return bgrid, agrid_big
end
