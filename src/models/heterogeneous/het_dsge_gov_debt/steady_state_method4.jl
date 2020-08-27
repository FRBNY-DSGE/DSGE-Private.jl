# TODO: refactor to reduce number of allocations further, e.g. C_Final, c_pol_in, etc.
function method4_steadystate!(m::HetDSGEGovDebt;
                              βlo::S = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                              βhi::S = min(exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), 0.999999),
                              excess::S = 5000., tol::S = 1e-4, maxit::Int64 = 20, βband::S = 1e-2,
                              roots_algorithm = nothing,
                              euler_anderson::Bool = true,
                              kf_anderson::Bool = false,
                              kf_eigen::Bool = true,
                              doplots::Bool = false, verbose::Symbol = :none) where {S <: Real}
    @assert !(kf_eigen && kf_anderson) "Only one of kf_anderson and kf_eigen can be true"

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

        if get_setting(m, :calibrate_income_targets)
            m[:sH_over_sL], m[:elo], m[:ehi] = compute_income_process_parameters(m)
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
        egrid, ewts, g_of_e = construct_egrid(m[:ehi].value, m[:elo].value, ne)

        # Initialize mollifier function
        qfunc(x) = mollifier_hetdsgegovdebt(x, m[:ehi].value, m[:elo].value)

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
                method4_find_steadystate!(m, na, ns, ne,
                                          egrid, ewts, g_of_e,
                                          f, agrid, sgrid,
                                          R, H, η, γ, ω, T, bg, qfunc;
                                          βlo = βlo, βhi = βhi,
                                          excess = excess, tol = tol, maxit = maxit,
                                          βband = βband, doplots = doplots,
                                          m_anderson = m_anderson,
                                          β_anderson = β_anderson,
                                          transition_mat = kf_eigen ? Matrix{S}(undef, na * ns, na * ns) : Matrix{S}(undef, 0, 0),
                                          euler_anderson = euler_anderson,
                                          kf_anderson = kf_anderson,
                                          kf_eigen = kf_eigen,
                                          roots_algorithm = roots_algorithm,
                                          verbose = verbose)

                if verbose == :high
                    println("The distribution D(a, s) integrates to $(round(sum(m[:Dstar].value), digits = 3)).")

                    # Calculate some summary statistics
                    los_save = sum(m[:Dstar].value[1:na] .* (agrid - m[:cstar].value[1:na]))
                    his_save = sum(m[:Dstar].value[1+na:end] .* (agrid - m[:cstar].value[1+na:end]))
                    los_C    = sum(m[:Dstar].value[1:na] .* m[:cstar].value[1:na])
                    his_C    = sum(m[:Dstar].value[1+na:end] .* m[:cstar].value[1+na:end])
                    println("Aggregate savings by low and high skill workers (resp.):            " *
                            "($(round(los_save, digits = 3)), " * "$(round(his_save, digits = 3)))")
                    println("Aggregate consumption by low and high skill workers (resp.):        " *
                            "($(round(los_C, digits = 3)), " * "$(round(his_C, digits = 3)))")

                end

                if doplots
                    p = plot(fit(Histogram, agrid, Weights(m[:Dstar].value[1:na]), nbins = na),
                             label = "low skill", color = :blue)
                    plot!(fit(Histogram, agrid, Weights(m[:Dstar].value[(na + 1):end]), nbins = na),
                          label = "high skill", color = :red)
                    savefig(p, "method4_agrid_vs_D_beta=$(string(round(m[:βstar].value, digits = 3))).pdf")

                    p = plot(agrid, m[:cstar].value[1:na], label = "low skill")
                    plot!(p, agrid, m[:cstar].value[(na + 1):end], label = "high skill")
                    savefig(p, "method4_agrid_vs_C_beta=$(string(round(m[:βstar].value, digits = 3))).pdf")

                    p = plot(agrid, agrid - m[:cstar].value[1:na], label = "low skill")
                    plot!(p, agrid, agrid - m[:cstar].value[(na + 1):end], label = "high skill")
                    savefig(p, "method4_agrid_vs_Saving_beta=$(string(round(m[:βstar].value, digits = 3))).pdf")
                end

                # use ones for now before refactoring since no longer using quadrature over s
                m[:mpc] = ave_mpc(m[:Dstar].value,   m[:cstar].value, agrid, kron(ones(size(swts)), awts), na, ns)
                m[:pc0] = frac_zero(m[:Dstar].value, m[:cstar].value, agrid, kron(ones(size(swts)), awts), ns)
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

    nothing
end

# TODO: refactor to not allocate KF_in if using kf_eigen or transition_mat if using kf_anderson
function method4_find_steadystate!(m::HetDSGEGovDebt, na::Int, ns::Int, ne::Int,
                                   egrid::Vector{Float64}, ewts::Vector{Float64}, g_of_e::Vector{Float64},
                                   f::Matrix{Float64},
                                   agrid::Vector{Float64}, sgrid::Vector{Float64},
                                   R::Float64, H::Float64, η::Float64, γ::Float64, ω::Float64, T::Float64, bg::Float64, qfunc;
                                   βlo::S = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                                   βhi::S = min(exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue), 0.9999999),
                                   excess::S = 5000., tol::S = 1e-4, maxit::Int64 = 20, βband::S = 1e-2,
                                   euler_anderson::Bool = false, kf_anderson::Bool = false, kf_eigen::Bool = false,
                                   m_anderson::Int = 5, β_anderson::S = 1.,
                                   transition_mat::Matrix{S} = Matrix{S}(undef, 0, 0), roots_algorithm = nothing,
                                   doplots::Bool = false, verbose::Symbol = :high) where {S <: Real}

    na_c = get_setting(m, :na_c)

    βlo_temp = βlo
    βhi_temp = βhi

    reject  = false
    counter = 0

    # Initial guess
    β              = NaN # Need to define β here so it's accessible outside of while lopp
    c_pol_in       = (R - 1) * repeat(agrid, 1, ns, ne) .+ ω * H * repeat(sgrid', na, 1, ne)
    KF_in          = fill(1.0 / (ns * na), na, ns)

    # If want to keep the last β/don't recompute
    if get_setting(m, :use_last_βstar) && !isnan(m[:βstar].value)

        # This short-circuits computation of the policy function
        βlo = βhi = m[:βstar].value
    # If you have a β saved, start from there (if you don't, βhi and βlo are defined as they are passed in to function)
    elseif !isnan(m[:βstar].value)

        # If one has computed β* before, then we first bisect into a neighborhood around it and check if the signs are the opposite
        βlo_temp = m[:βstar].value - βband
        βhi_temp = m[:βstar].value + βband
        c_pol_out, c_as, bp, KF, reject = method4_policy_hetdsgegovdebt(na, ns, ne, na_c, βlo_temp, R, ω, H, η, T, γ,
                                                                        m[:ehi].value, m[:elo].value,
                                                                        agrid, sgrid, c_pol_in, KF_in,
                                                                        f, egrid, ewts, g_of_e, transition_mat, qfunc,
                                                                        maxit = get_setting(m, :policy_maxit),
                                                                        m_anderson = m_anderson,
                                                                        β_anderson = β_anderson,
                                                                        euler_anderson = euler_anderson,
                                                                        kf_anderson = kf_anderson,
                                                                        kf_eigen = kf_eigen,
                                                                        eigen_method = haskey(get_settings(m), :eigen_method) ?
                                                                        get_setting(m, :eigen_method) : :krylov,
                                                                        euler_tol = get_setting(m, :euler_tol),
                                                                        kf_tol = get_setting(m, :kf_tol),
                                                                        eigen_tol = get_setting(m, :eigen_tol),
                                                                        C_tol = get_setting(m, :C_tol))

        excess_lo = method4_compute_excess(KF, bp, bg)

        if excess_lo < 0 && abs(excess_lo) > tol
            βlo = βlo_temp
            c_pol_out, c_as, bp, KF, reject = method4_policy_hetdsgegovdebt(na, ns, ne, na_c, βlo_temp, R, ω, H, η, T, γ,
                                                                            m[:ehi].value, m[:elo].value,
                                                                            agrid, sgrid, c_pol_in, KF_in,
                                                                            f, egrid, ewts, g_of_e, transition_mat, qfunc,
                                                                            maxit = get_setting(m, :policy_maxit),
                                                                            m_anderson = m_anderson,
                                                                            β_anderson = β_anderson,
                                                                            euler_anderson = euler_anderson,
                                                                            kf_anderson = kf_anderson,
                                                                            kf_eigen = kf_eigen,
                                                                            eigen_method = haskey(get_settings(m), :eigen_method) ?
                                                                            get_setting(m, :eigen_method) : :krylov,
                                                                            euler_tol = get_setting(m, :euler_tol),
                                                                            kf_tol = get_setting(m, :kf_tol),
                                                                            eigen_tol = get_setting(m, :eigen_tol),
                                                                            C_tol = get_setting(m, :C_tol))

            # Update the guess for c(b, s, e) only if the Euler iteration converged
            if !reject
                c_pol_in = c_pol_out
            end

            excess_hi = method4_compute_excess(KF, bp, bg)

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
            c_pol_out, c_as, bp, KF, reject = method4_policy_hetdsgegovdebt(na, ns, ne, na_c, β, R, ω, H, η, T, γ,
                                                                            m[:ehi].value, m[:elo].value,
                                                                            agrid, sgrid, c_pol_in, KF_in,
                                                                            f, egrid, ewts, g_of_e, transition_mat, qfunc,
                                                                            maxit = get_setting(m, :policy_maxit),
                                                                            m_anderson = m_anderson,
                                                                            β_anderson = β_anderson,
                                                                            euler_anderson = euler_anderson,
                                                                            kf_anderson = kf_anderson,
                                                                            kf_eigen = kf_eigen,
                                                                            eigen_method = haskey(get_settings(m), :eigen_method) ?
                                                                            get_setting(m, :eigen_method) : :krylov,
                                                                            euler_tol = get_setting(m, :euler_tol),
                                                                            kf_tol = get_setting(m, :kf_tol),
                                                                            eigen_tol = get_setting(m, :eigen_tol),
                                                                            C_tol = get_setting(m, :C_tol))

            # Update the guess for c(b, s, e) only if the Euler iteration converged
            if !reject
                c_pol_in = c_pol_out
            end

            excess = method4_compute_excess(KF, bp, bg)
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
            @warn "Euler iteration does not converge"
            reject = true
            # TODO: maybe do not auto-update c_pol_in unless you have euler iteration covnergence.
            #       c_pol_in should instead be set back to initial guess
        end
    elseif isa(roots_algorithm, AbstractBracketing)
        β = find_zero(β -> bisect_β(β, na, ns, ne, na_c, R, ω, H, η, T, γ, m[:ehi].value, m[:elo].value, agrid, sgrid, c_pol_in,
                                    KF_in, f, egrid, ewts, g_of_e, transition_mat, qfunc, bg, get_setting(m, :policy_maxit), m_anderson,
                                    euler_anderson, kf_anderson, kf_eigen, haskey(get_settings(m), :eigen_method) ?
                                    get_setting(m, :eigen_method) : :krylov,
                                    get_setting(m, :euler_tol), get_setting(m, :kf_tol), get_setting(m, :eigen_tol), get_setting(m, :C_tol)),
                      (βlo, βhi), roots_algorithm, maxevals = maxit, atol = tol)
        c_pol_in, c_as, bp, KF, reject = method4_policy_hetdsgegovdebt(na, ns, ne, na_c, β, R, ω, H, η, T, γ,
                                                                       m[:ehi].value, m[:elo].value,
                                                                       agrid, sgrid, c_pol_in, KF_in,
                                                                       f, egrid, ewts, g_of_e, transition_mat, qfunc,
                                                                       maxit = get_setting(m, :policy_maxit),
                                                                       m_anderson = m_anderson,
                                                                       β_anderson = β_anderson,
                                                                       euler_anderson = euler_anderson,
                                                                       kf_anderson = kf_anderson,
                                                                       kf_eigen = kf_eigen,
                                                                       eigen_method = haskey(get_settings(m), :eigen_method) ?
                                                                       get_setting(m, :eigen_method) : :krylov,
                                                                       euler_tol = get_setting(m, :euler_tol),
                                                                       kf_tol = get_setting(m, :kf_tol),
                                                                       eigen_tol = get_setting(m, :eigen_tol),
                                                                       C_tol = get_setting(m, :C_tol))

    elseif isa(roots_algorithm, AbstractSecant)
        β = find_zero(β -> bisect_β(β, na, ns, ne, na_c, R, ω, H, η, T, γ, m[:ehi].value, m[:elo].value, agrid, sgrid, c_pol_in,
                                    KF_in, f, egrid, ewts, g_of_e, transition_mat, qfunc, bg, get_setting(m, :policy_maxit), m_anderson,
                                    β_anderson, euler_anderson, kf_anderson, kf_eigen, haskey(get_settings(m), :eigen_method) ?
                                    get_setting(m, :eigen_method) : :krylov,
                                    get_setting(m, :euler_tol), get_setting(m, :kf_tol), get_setting(m, :eigen_tol),
                                    get_setting(m, :C_tol)),
                      (βlo + βhi) / 2., roots_algorithm, maxevals = maxit, atol = tol)
        c_pol_in, c_as, bp, KF, reject = method4_policy_hetdsgegovdebt(na, ns, ne, na_c, β, R, ω, H, η, T, γ,
                                                                       m[:ehi].value, m[:elo].value,
                                                                       agrid, sgrid, c_pol_in, KF_in,
                                                                       f, egrid, ewts, g_of_e, transition_mat, qfunc,
                                                                       maxit = get_setting(m, :policy_maxit),
                                                                       β_anderson = β_anderson,
                                                                       m_anderson = m_anderson,
                                                                       euler_anderson = euler_anderson,
                                                                       kf_anderson = kf_anderson,
                                                                       kf_eigen = kf_eigen,
                                                                       eigen_method = haskey(get_settings(m), :eigen_method) ?
                                                                       get_setting(m, :eigen_method) : :krylov,
                                                                       euler_tol = get_setting(m, :euler_tol),
                                                                       kf_tol = get_setting(m, :kf_tol),
                                                                       eigen_tol = get_setting(m, :eigen_tol),
                                                                       C_tol = get_setting(m, :C_tol))
    else
        error("Cannot use the Roots algorithm $(typeof(roots_algorithm))")
    end

    bgrid, agrid_big = construct_bgrid_agrid_big(agrid, sgrid, egrid, na, ns, ne, γ, ω, H, T, R)
    if doplots
        for is in 1:ns
            p = plot()
            for ie in 1:ne
                plot!(bgrid, c_pol_in[:, is, ie], label = "e=$(egrid[ie])")
            end
            savefig(p, "method4_bgrid_vs_C_varye_s=$(sgrid[is]).pdf")
        end
    end

    # If policy function does not converge, we signal to likelihood that should reject
    m <= Setting(:auto_reject, reject)
    m[:lstar]  = 1 ./ c_as
    m[:cstar]  = c_as
    m[:Dstar]  = KF
    m[:βstar]  = β
    m[:β_save] = β

    nothing
end

function bisect_β(β::S, na::Int, ns::Int, ne::Int, na_c::Int, R::S, ω::S, H::S, η::S, T::S, γ::S,
                  ehi::S, elo::S, agrid::AbstractVector{S}, sgrid::AbstractVector{S},
                  c_pol_in::AbstractArray{S, 3}, KF_in::AbstractArray{S, 2}, f::AbstractMatrix{S},
                  egrid::AbstractVector{S}, ewts::AbstractVector{S}, g_of_e::AbstractVector{S},
                  transition_mat::AbstractMatrix{S}, qfunc, bg::S,
                  maxit::Int, m_anderson::Int, β_anderson::S, euler_anderson::Bool, kf_anderson::Bool, kf_eigen::Bool,
                  eigen_method::Symbol, euler_tol::S, kf_tol::S, eigen_tol::S, C_tol::S) where {S <: Real}
    ~, ~, bp, KF, ~ = method4_policy_hetdsgegovdebt(na, ns, ne, na_c, β, R, ω, H, η, T, γ,
                                                    ehi, elo, agrid, sgrid, c_pol_in, KF_in,
                                                    f, egrid, ewts, g_of_e, transition_mat, qfunc,
                                                    maxit = maxit, m_anderson = m_anderson,
                                                    β_anderson = β_anderson,
                                                    euler_anderson = euler_anderson,
                                                    kf_anderson = kf_anderson,
                                                    kf_eigen = kf_eigen,
                                                    eigen_method = eigen_method,
                                                    euler_tol = euler_tol,
                                                    kf_tol = kf_tol,
                                                    eigen_tol = eigen_tol,
                                                    C_tol = C_tol)
    excess = method4_compute_excess(KF, bp, bg)
    return excess
end

function method4_policy_hetdsgegovdebt(na::Int, ns::Int, ne::Int, na_c::Int, β::S, R::S, ω::S, H::S, η::S,
                                       T::S, γ::S, ehi::S, elo::S, agrid::Vector{S},
                                       sgrid::Vector{S}, c_pol::AbstractArray{S, 3}, KF_in::AbstractArray{S, 2},
                                       f::Matrix{S}, egrid::Vector{Float64}, ewts::Vector{Float64},
                                       g_of_e::Vector{Float64}, transition_mat::AbstractMatrix{S}, qfunc, dist::S = 1.;
                                       maxit::Int64 = 1000, m_anderson::Int = 5, β_anderson::S = 1.,
                                       euler_anderson::Bool = false, kf_anderson::Bool = false, kf_eigen::Bool = false,
                                       eigen_method::Symbol = :krylov, euler_tol::S = 1e-10, kf_tol::S = 1e-10,
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
        out = nlsolve((F_c_pol, c_pol) -> fixedpoint_c_policy_nlsolve!(F_c_pol, c_pol, ns, ne, na_c, sum_term, f, ewts, g_of_e,
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

    # Calculate c(a, s) and bp(a, s)
    c_as = integrate_out_e(agrid, agrid_big, bgrid, sgrid, c_pol, tol = C_tol)
    bp   = (R * exp(-γ)) .* (agrid .- c_as)

    # Fixed point problem for D(a, s)
    if kf_anderson
        out  = nlsolve((F_D, D_as_in) -> fixedpoint_KF_nlsolve!(F_D, D_as_in, c_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T),
                        KF_in, ftol = kf_tol, iterations = maxit, method = :anderson, m = m_anderson, beta = β_anderson)
        D_as = vec(out.zero) # out.zero already allocated so just re-assign D_as
    elseif kf_eigen
        D_as = stationary_KF!(transition_mat, c_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T, (agrid[end] - agrid[1]) / length(agrid);
                              tol = eigen_tol, method = eigen_method)
    else
        D_as  = deepcopy(KF_in) # do not want to alter the initial guess
        D′_as = similar(D_as)
        counter = 0
        dif     = 1.
        while dif > kf_tol && counter <= maxit
            fixedpoint_KF!(D′_as, D_as, c_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T)

            # check convergence
            dif      = maximum(abs.(D′_as - D_as))
            D_as    .= D′_as
            counter += 1
        end
        D_as = vec(D_as)
    end

    return c_pol, vec(c_as), vec(bp), D_as, reject
end

function fixedpoint_KF!(D′_as::AbstractArray{S, 2}, D_as::AbstractArray{S, 2}, C_pol::AbstractArray{S, 2},
                        agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S},
                        qfunc, ω::S, H::S, R::S, γ::S, T::S) where {S <: Real}

    D′_as .= 0.

    # Calculate D′(a, s) from  D(a, s) by using the KF equation
    na, ns = size(D_as)
    for isp in 1:ns
        for iap in 1:na
            for is in 1:ns
                for ia in 1:na
                    # Technically calculating m′(a, s) from D(a, s) in this step
                    D′_as[iap, isp] += D_as[ia, is] * f[is, isp] / (ω * H * sgrid[isp]) *
                        qfunc((agrid[iap] - R * exp(-γ) * (agrid[ia] - C_pol[ia, is]) - T) / (ω * H * sgrid[isp]))
                end
            end
        end
    end

    D′_as ./= sum(D′_as) # ensure sum(D′(a, s)) = 1. This step maps m′(a, s) -> m(a, s)
end

function fixedpoint_KF_nlsolve!(F_D::AbstractArray{S, 2}, D_as::AbstractArray{S, 2}, C_pol::AbstractArray{S, 2},
                                agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S},
                                qfunc, ω::S, H::S, R::S, γ::S, T::S) where {S <: Real}

    # Zero out
    F_D .= 0.

    # Calculate D′(a, s) from  D(a, s) by using the KF equation
    na, ns = size(D_as)
    for isp in 1:ns
        for iap in 1:na
            for is in 1:ns
                for ia in 1:na
                    F_D[iap, isp] += D_as[ia, is] * f[is, isp] / (ω * H * sgrid[isp]) *
                        qfunc((agrid[iap] - R * exp(-γ) * (agrid[ia] - C_pol[ia, is]) - T) / (ω * H * sgrid[isp]))
                end
            end
        end
    end

    F_D ./= sum(F_D) # Normalize to 1
    F_D .-= D_as     # Subtract D(a, s) from F_D in place ⇒ F_D = D′(a, s) - D(a, s)
end

function stationary_KF!(transition_mat::AbstractMatrix{S}, C_as::AbstractMatrix{S},
                        agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S}, qfunc,
                        ω::S, H::S, R::S, γ::S, T::S, ι::S; tol::S = 2e-1, method::Symbol = :krylov) where {S <: Real}
    # Check D(a, s) solves the KF equation by constructing the transition matrix A and solving the eigenvalue problem.
    # Note that the Kolmogorov forward equation we use is defined for m(a, s) = D(a, s) / ι, hence
    # vec(m′(a, s))     = A * ι * vec(m(a, s)), which reduces to
    # vec(D′(a, s)) / ι = A * vec(D(a, s))
    # vec(D′(a, s))     = A * ι * vec(D(a, s))
    construct_transition!(transition_mat, C_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T)

    # Find largest eigenvalue and associated eigenvector
    # For computational reasons, we solve D(a, s) / ι = A * D(a, s) instead of D(a, s) = A * ι * D(a, s)
    ~, μ = stationary_eigenvector(sparse(transition_mat), ι, method; tol = tol)
    μ ./= sum(μ) # normalize to 1

    # return reshape(μ, na, ns)
    return μ
end

function construct_transition!(transition_mat::AbstractMatrix{S}, C_as::AbstractMatrix{S},
                               agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S},
                               qfunc, ω::S, H::S, R::S, γ::S, T::S) where {S <: Real}
    na = length(agrid)
    ns = length(sgrid)
    @inbounds @simd for is in 1:ns
        @inbounds @simd for ia in 1:na
            @inbounds @simd for isp in 1:ns
                @inbounds @simd for iap in 1:na
                    transition_mat[na * (isp - 1) + iap, na * (is - 1) + ia] = f[is, isp] / (ω * H * sgrid[isp]) *
                        qfunc((agrid[iap] - R * exp(-γ) * (agrid[ia] - C_as[ia, is]) - T) / (ω * H * sgrid[isp]))
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

@inline function method4_compute_excess(μ::AbstractVector{S}, bp::AbstractVector{S}, bg::S) where {S <: Real}
    return dot(μ, bp) - bg # Compute excess supply of savings, which is a fn of w
end
