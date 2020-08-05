function method3_steadystate!(m::HetDSGEGovDebt;
                              βlo::S = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                              βhi::S = exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                              excess::S = 5000.,
                              tol::S = 1e-4,
                              maxit::Int64 = 20,
                              βband::S = 1e-2,
                              use_quadrature::Bool = false) where {S <: Real}
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
        ss_skill_distr = [pHL/(pLH+pHL); pLH/(pLH+pHL)]
        slo    = 1.0 / (ss_skill_distr'*[1;sH_over_sL])
        sgrid  = slo*[1;sH_over_sL]
        sscale = sgrid[2] - sgrid[1]
        swts   = (sscale/ns)*ones(ns) # Quadrature weights
        m.grids[:sgrid] = Grid(sgrid, swts, sscale)

        # @test isapprox(mean(f^1000*sgrid), 1.0, atol = 1e-3)

        # Construct egrid
        qfunction(x::Float64) = mollifier_hetdsgegovdebt(x, m[:ehi].value, m[:elo].value)
        # egrid_gk = gauss(ne) # Use FastGaussQuadrature, faster to do so
        egrid, ewts = gausslegendre(ne)
        egrid      .= transform_ab(m[:elo].value, m[:ehi].value, egrid)

        # Normalize egrid, ewts so that ∫ g(e) de ≈ ∑ᵢ g(eᵢ) * wᵢ = 1
        g_of_e      = map(x -> mollifier_hetdsgegovdebt(x, m[:ehi].value, m[:elo].value), egrid) # g(eᵢ)
        # ewts      ./= dot(qfunction.(egrid), ewts)
        ewts      ./= dot(g_of_e, ewts) # normalize wᵢ to w̃ᵢ so that ∑ᵢ w̃ᵢ * gᵢ = 1
        # egrid = egrid ./ dot(g_of_e .* egrid, ewts)
        egrid     ./= dot(g_of_e .* egrid, ewts) # Normalize egrid so that mean ∫ e g(e) de = 1, should be a small adjustment
        @show egrid
        # ẽᵢ = eᵢ / (∑ᵢ (gᵢ * eᵢ * w̃ᵢ)), hence
        # ∑ᵢ ̃eᵢ * gᵢ * w̃ᵢ = (∑ᵢ eᵢ * gᵢ * w̃ᵢ) / (∑ᵢ (gᵢ * eᵢ * w̃ᵢ)) = 1, but we could just change the weights

        # Construct agrid
        smin = minimum(sgrid) # * m[:elo].value                                   # lowest possible skill
        alo  = ω * smin * H - R * η * exp(-γ) + T # + sgrid[1]*ω*H*0.05 # lowest SS possible cash on hand
        ahi  = max(alo * 2., alo + 20.0)         # upper bound on cash on hand

        ascale = (ahi - alo)                  # size of w grids
        agrid  = collect(range(alo, stop = ahi, length = na)) # Evenly spaced grid
        awts   = fill(ascale / na, na)          # Quadrature weights, sum up to 12

        m <= Setting(:alo, alo)
        m <= Setting(:ahi, ahi)
        m <= Setting(:ascale, ascale)

        # Once have updated grids, can call steady state and compute other two moments
        method3_find_steadystate!(m, na, ns, ne,
                                  egrid, ewts, g_of_e,
                                  f,
                                  agrid, sgrid,
                                  R, H, η, γ, ω, T, bg;
                                  βlo = βlo, βhi = βhi,
                                  excess = excess, tol = tol, maxit = maxit,
                                  βband = βband, use_quadrature = use_quadrature)

        # TODO: Remove these lines
        @show sum(m[:μstar].value)
        p = plot(agrid, m[:μstar].value[1:300], label = "low skill")
        plot!(p, agrid, m[:μstar].value[301:600], label = "high skill")
        if use_quadrature
            savefig(p, "method3_quadrature_agrid_vs_D_beta=$(string(round(m[:βstar].value, digits = 3))).pdf")
        else
            savefig(p, "method3_sortinterp_agrid_vs_D_beta=$(string(round(m[:βstar].value, digits = 3))).pdf")
        end

        m[:mpc] = ave_mpc(m[:μstar].value,   m[:cstar].value, agrid, kron(swts, awts), na, ns)
        m[:pc0] = frac_zero(m[:μstar].value, m[:cstar].value, agrid, kron(swts, awts), ns)
    end

    nothing
end

function method3_find_steadystate!(m::HetDSGEGovDebt, na::Int, ns::Int, ne::Int,
                           egrid::Vector{Float64}, ewts::Vector{Float64}, g_of_e::Vector{Float64},
                           f::Matrix{Float64},
                           agrid::Vector{Float64}, sgrid::Vector{Float64},
                           R::Float64, H::Float64, η::Float64, γ::Float64, ω::Float64, T::Float64, bg::Float64;
                           βlo::S = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                           βhi::S = exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                           excess::S = 5000.,
                           tol::S = 1e-4,
                           maxit::Int64 = 20,
                           βband::S = 1e-2,
                           use_quadrature::Bool = false) where {S <: Real}

    na_c = get_setting(m, :na_c)

    βlo_temp = βlo
    βhi_temp = βhi

    reject  = false
    counter = 1

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
                                                         f, egrid, ewts, g_of_e, damp = get_setting(m, :policy_damp),
                                                         maxit = get_setting(m, :policy_maxit))

        excess_lo = compute_excess(KF, bp, bg)

        if excess_lo < 0 && abs(excess_lo) > tol
            βlo = βlo_temp
            c_pol_in, bp, KF, reject = method3_policy_hetdsgegovdebt(na, ns, ne, na_c, βhi_temp, R, ω, H, η, T, γ, m[:ehi].value,
                                                             m[:elo].value, agrid, sgrid, c_pol_in, KF_in,
                                                             f, egrid, ewts, g_of_e,
                                                             damp = get_setting(m, :policy_damp),
                                                             maxit = get_setting(m, :policy_maxit))
            excess_hi = compute_excess(KF, bp, bg)

            if excess_hi > 0
                βhi = βhi_temp
            end
        elseif excess_lo >= 0
            βhi = βlo_temp
        end
    end

    # If you don't have a β, guess a β
    while abs(excess) > tol && counter < maxit # clearing markets
        β = (βlo + βhi) / 2.0

        c_pol_in, bp, KF, reject = method3_policy_hetdsgegovdebt(na, ns, ne, na_c, β, R, ω, H, η, T, γ, m[:ehi].value, m[:elo].value,
                                                         agrid, sgrid, c_pol_in, KF_in,
                                                         f, egrid, ewts, g_of_e,
                                                         damp = get_setting(m, :policy_damp),
                                                         maxit = get_setting(m, :policy_maxit))
        excess = compute_excess(KF, bp, bg)
        @show counter, β, excess
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

    bmin = 0.0
    bmax = exp(γ) * (maximum(agrid)- ω * maximum(sgrid) * maximum(egrid) * H) # NOTE egrid is not in [elo, ehi] b/c transformed
    bgrid = bmin .+ ((1:na) / na).^2 * (bmax - bmin)                          # but makes quadrature work (normalize to 1)
    # a grid (cash on hand) implied by bgrid (assets) is: map b into a using equation at top of page 4
    agrid_big = Array{Float64}(undef, na, ns, ne)
    for ie in 1:ne
        for is in 1:ns
            agrid_big[:, is, ie] = (ω * sgrid[is] * egrid[ie] * H + T) .+ exp(-γ) .* bgrid
        end
    end
    C_Final, KF_Final, ell = integrate_out_e(agrid, agrid_big, bgrid, sgrid, c_pol_in, KF, ω, H, T, γ;
                                             use_quadrature = use_quadrature, g_of_e = g_of_e, ewts = ewts)

    # If policy function does not converge, we signal to likelihood that should reject
    m <= Setting(:auto_reject, reject)
    m[:lstar]  = ell
    m[:cstar]  = C_Final
    m[:μstar]  = KF_Final
    m[:βstar]  = β
    m[:β_save] = β

    nothing
end


function method3_policy_hetdsgegovdebt(na::Int, ns::Int, ne::Int, na_c::Int, β::S, R::S, ω::S, H::S, η::S,
                               T::S, γ::S, ehi::S, elo::S, agrid::Vector{S},
                               sgrid::Vector{S}, c_pol::Array{S}, KF_in::Array{S},
                               f::Matrix{S}, egrid::Vector{Float64}, ewts::Vector{Float64},
                               g_of_e::Vector{Float64},
                               dist::S = 1., tol::S = 1e-10;
                               maxit::Int64 = 500, damp::S = 0.5) where {S <: Real}

    colors = [:red, :blue, :green, :yellow, :purple]
    counter = 1
    reject = false

    # Constrainted consumption
    c_constrained = Matrix{Float64}(undef, ns, ne)
    for ie in 1:ne
        # for is in 1:ns
            # e = egrid[ie] #minimum([egrid[ie], 1.0])
        e = min(egrid[ie], 1.)
        c_constrained[:, ie] = sgrid .* (ω * e * H) .+ T # ω * s * e * H + T
        # end
    end

    c_poli = deepcopy(c_pol)

    bmin = 0.0
    bmax = exp(γ) * (maximum(agrid)- ω * maximum(sgrid) * maximum(egrid) * H) # NOTE egrid is not in [elo, ehi] b/c transformed
    bgrid = bmin .+ ((1:na) / na).^2 * (bmax - bmin)                          # but makes quadrature work (normalize to 1)
    # a grid (cash on hand) implied by bgrid (assets) is: map b into a using equation at top of page 4
    agrid_big = Array{Float64}(undef, na, ns, ne)
    for ie in 1:ne
        for is in 1:ns
            agrid_big[:, is, ie] = (ω * sgrid[is] * egrid[ie] * H + T) .+ exp(-γ) .* bgrid
        end
    end

    # Allocate memory here
    bp       = bgrid # bp = b' = bprime, only to make it clear that we're working with b', not b
    sum_term = zeros(S, length(bp))

    # Fixed point problem for c_poli
    while dist > tol && counter < maxit # TODO: rewrap the consumption function loop inside a function b/c fixed point problem
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

                # Handle constrained consumption today: for b < 0, need to reset c such that b is exactly 0. See line 254
                c[b .< 0.] = -(bp[b .< 0] ./ R) .+  ω * sgrid[is] * egrid[ie] * H .+ T
                b         .= vec(exp(γ) * ((bp ./ R) .- ω * sgrid[is] * egrid[ie] * H .- T .+ c))

                # If b[1] is positive, then people aren't using a c(0, s, e) policy but also must have some spare b
                # that they additionally consume. This code block assumes "relative" monotonicity of c
                if b[1] > 0.
                    @assert c_constrained[is, ie] < c[1] # Check consuming more than implied by constrained rule if b[1] > 0
                    c_c = collect(range(c_constrained[is, ie], c[1], length = na_c)) # constrained rule is an equal spacing
                    b_c = exp(γ) * (-ω * sgrid[is] * egrid[ie] * H - T .+ c_c)       # between no-assets rule and c[1]
                    if -1e-14 < b_c[1] < 0.
                        b_c[1] = 0. # Force the first point to be zero (in case of floating point errors)
                    end
                    @assert all(b_c .>= -1e-14)    # Make sure implied b is non-negative
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

        dist  = maximum(abs.(c_pol - c_poli)) # Inf norm
        c_pol = deepcopy(c_poli)

        counter += 1
        if counter == maxit
            @warn "Euler iteration did not converge. The final distance is $(dist)"
            reject = true
        end
    end
    @assert all(agrid_big - c_pol .>= -1e16) "Consumption is not always less than cash-on-hand."
    println("Cpolicy counter: $(counter)")

    # Construct weights for calculating distribution over (b, s, e), see the KF loop to undersatnd how these Arrays are used
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

    # Iterate asset transition matrix starting from KF_in
    dif = 1
    pd  = deepcopy(KF_in)      # D(b, s, e)
    pdi = zeros(S, na, ns, ne) # D'(b', s', e')
    counter = 0
    while dif > tol
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

        # check convergence
        dif = maximum(abs.(pdi - pd))

        # Make sure that distribution integrates to 1
        @assert isapprox(sum(pdi), 1.0, atol = 1e-6)
        pd   = deepcopy(pdi)
        pdi .= 0. # Reset to zero

      #=  p = plot(agrid, pd[:, 1], label = "low skill")
        plot!(p, agrid, pd[:, 2], label = "high skill")
        savefig(p, "agrid_vs_D_first.png") =#

        counter += 1
    end
    println("KF counter: $(counter)")

    return c_pol, bgrid, pd, reject
end

# KF is D(a, s, e), bp is the bprime grid, and bg is the government's supply of onds
@inline function compute_excess(KF::AbstractArray{S, 3}, bp::AbstractVector{S}, bg::S) where {S <: Real}
    # Return ∑ᵢ ∑ₛ ∑ₑ D(bᵢ, s, e) bᵢ - β
    return sum(sum(KF, dims = (2, 3)) .* bp) - bg # Summing over (s, e) first speed things up, fewer multiplication operations
end

function transform_ab(a::S, b::S, grid::AbstractVector{S}) where {S <: Real}
    xs = ((b-a)/2) .* grid .+ (a+b)/2
    return xs
end

# Maps c(b, s, e) -> c(a, s) and D(b, s, e) -> D(a, s)
function integrate_out_e(agrid::AbstractVector{S}, agrid_big::AbstractArray{S, 3},
                         bgrid::AbstractVector{S}, sgrid::AbstractVector{S}, c_pol::AbstractArray{S, 3}, D_bse::AbstractArray{S, 3},
                         ω::S, H::S, T::S, γ::S;
                         use_quadrature::Bool = false, g_of_e::AbstractVector{S} = Vector{S}(undef, 0),
                         ewts::AbstractVector{S} = Vector{S}(undef, 0)) where {S <: Real}
    # Map c(b, s, e) -> c(a, s, e)
    na, ns, ne = size(agrid_big)
    C_Final    = Matrix{S}(undef, na, ns)
    if use_quadrature && !isempty(g_of_e) && !isempty(ewts)
        C_b_conds = Vector{S}(undef, na)
        for is in 1:ns
            implied_agrid = ((ω * H) * sgrid[is] + T) .+ exp(-γ) .* bgrid # True mean b/c ∫ e * g(e) de should integrate to 1.
            for ia in 1:na
                # c(b, s, e) -> c(b, s) via c(b, s) = ∫ g(e) c(b, s, e) de ≈ ∑ᵢ g(eᵢ) c(b, s, eᵢ) wᵢ
                C_b_conds[ia] = sum(g_of_e .* c_pol[ia, is, :]     .* ewts)
            end
            C_Final[:, is] = interp_one(implied_agrid, C_b_conds, agrid)
            @test all(agrid .- C_Final[:, is] .>= -1e16)
        end
    else
        for is in 1:ns
            # Sort the a's, given the skill level
            vec_agrid_big_is = vec(agrid_big[:, is, :])
            sorted_inds = sortperm(vec_agrid_big_is)

            # agrid_big is the grid of a implied by bgrid and exogenous states (s, e),
            # whereas agrid is the grid of a that's paseed in.
            # Linear interpolation
            C_Final[:, is] = interp_one(vec_agrid_big_is[sorted_inds], vec(c_pol[:, is, :])[sorted_inds], agrid)
            @test all(agrid .- C_Final[:, is] .>= -1e16)
        end
    end

    #  D(b, s, e) -> D(a, s)
    D_as = zeros(S, na, ns)
    for ie in 1:ne
        for is in 1:ns
            for ia in 1:na
                ind = findlast(agrid_big[ia, is, ie] .>= agrid) # Find i s.t. agridᵢ <= agrid_big[ia, is, ie] <= agridᵢ₊₁
                if isnothing(ind)
                    # @show (ia, is, ie), agrid_big[ia, is, ie], agrid[1], D_bse[ia, is, ie]
                    ind = 1
                    weight = 1.
                elseif ind == na
                    # @show (ia, is, ie), agrid_big[ia, is, ie], agrid[end], D_bse[ia, is, ie]
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
                   Fminbox(Optim.NelderMead()), Optim.Options(f_calls_limit=300))
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


####### OLD CODE for policy iteration


#=
    if β < 0.0
        p = plot()
        for is = 1:ns
            for ie = 1:ne
                plot!(p, bgrid, ω*sgrid[is]*egrid[ie]*H .+ T .+ exp(-γ)*bgrid - c_pol[:, is, ie], label = "is = $(is), ie = $(ie)")
            end
        end
        savefig(p, "bgrid_vs_expression.png")
        p = plot()
        for is = 1:ns
            for ie = 1:ne
                lab = (is == 1 ? "Low skill" : "High skill") * "ie = $(ie)"
                plot!(p, bgrid, c_pol[:, is, ie], label = lab, linestyle = is==1 ? :solid : :dash, color = colors[ie])
            end
        end
        savefig(p, "bgrid_vs_cpol.png")

        p = plot()
        for is = 1:ns
            for ie = 1:ne
                lab = (is == 1 ? "Low skill" : "High skill") * "ie = $(ie)"
                plot!(p, agrid_big[:, is, ie], c_pol[:, is, ie], label = lab, linestyle = is==1 ? :solid : :dash, color = colors[ie])
            end
        end
        savefig(p, "agrid_big_vs_cpol.png")

    end
=#
    # Interpolate the (a, s, e) grid back to the (a, s) grid.
#=    C_Final = Matrix{Float64}(undef, na, ns)
    for is in 1:ns
        # Sort the a's and use those for everything
        sorted_inds = sortperm(vec(agrid_big[:, is, :]))
        # agrid_big is the grid of a implied by bgrid, whereas xgrid is the grid of a that's paseed in
        C_Final[:, is] = interp_one(vec(agrid_big[:, is, :])[sorted_inds], vec(c_pol[:, is, :])[sorted_inds], agrid)
        @test all(agrid .- C_Final[:, is] .>= -1e16)
    end
=#
  #=  if β < .75
        sorted_inds = sortperm(vec(agrid_big[:, 1, :]))
        p_l = plot(vec(agrid_big[:, 1, :])[sorted_inds], vec(c_pol[:, 1, :])[sorted_inds], label = "cpol", legend = :bottomright)
        sorted_inds = sortperm(vec(agrid_big[:, 2, :]))
        p_h  = plot(vec(agrid_big[:, 2, :])[sorted_inds], vec(c_pol[:, 2, :])[sorted_inds], label = "cpol", legend = :bottomright)
       # savefig(p, "cpol_agrid.png")
        plot!(p_l, agrid, vec(C_Final[:, 1]), label = "C Final")
        plot!(p_h, agrid, vec(C_Final[:, 2]), label = "C Final")
        savefig(p_l, "lowskill.png")
        savefig(p_h, "highskill.png")
    end =#


    # Compute a' implied by interpolated C_Final (back on the usual grid of a)
#=    ap = Array{Float64}(undef, na, ns, ne, ns)
    for is in 1:ns
        for isp in 1:ns
            for iep in 1:ne
                ap[:, is, iep, isp] = ω*sgrid[isp]*egrid[iep]*H .+ R*exp(-γ)*(agrid - C_Final[:, is])
            end
        end
    end

    b_grid_implied = Matrix{Float64}(undef, na, ns)
    for is in 1:ns
        b_grid_implied[:, is] = exp(γ)*agrid .- ω*sgrid[is]*H .- T
    end
@show minimum(b_grid_implied)

    # Finding the ergodic distribution
    # Assign weights to adjacent grid points paproportionally to distance
#    @show ap
#    @show agrid
    ib_pol, wei = histc(ap, agrid)
=#
#=
  if β < 0.0
      for is in 1:2
          for isp = 1:2
              p = plot()
              for ie = 1:ne
                  lab = "a' " * " ie = $(ie)"
                  plot!(p, agrid, ap[:, isp, ie, is], left_margin = 10mm, label = lab, legend = :bottomright, color = colors[ie])
                  lab = "agrid[ib_pol] " * " ie = $(ie)"
                  plot!(agrid, agrid[ib_pol[:, isp, ie, is]], left_margin = 10mm, label = lab, linestyle = :dash, color = colors[ie])
              end
              savefig(p, "ap_is=$(is)_isp=$(isp).png")
          end
      end
      #end
      p2 = plot(agrid, agrid - C_Final[:, 1], label = "low skill")
      plot!(p2, agrid, agrid - C_Final[:, 2], label = "high skill")
      savefig(p2, "agrid_minus_CFinal.png")
  end
=#

    # Check ib_pol and weights worked
   #= for ia in 1:na
        for is in 1:ns
            for ie in 1:ne
                for isp in 1:ns
                    # Make sure ap lies between the two nearest grid points on either side
                    if ap[ia, is, ie, isp] < minimum(agrid)
                        @test ib_pol[ia, is, ie, isp] == 1
                        @test wei[ia, is, ie, isp] == 1.0
                    elseif ap[ia, is, ie, isp] > maximum(agrid)
                        @test ib_pol[ia, is, ie, isp] == length(agrid) - 1
                        @test wei[ia, is, ie, isp] == 1.0

                    else
                        @test agrid[(ib_pol[ia, is, ie, isp])] <= ap[ia, is, ie, isp] <=
                            agrid[(ib_pol[ia, is, ie, isp] + 1)]
                        # If closer to left grid point, weight should be greater than 1-weight
                        if abs(agrid[(ib_pol[ia, is, ie, isp])] - ap[ia, is, ie, isp]) <
                            abs(ap[ia, is, ie, isp] - agrid[(ib_pol[ia, is, ie, isp] + 1)])
                            @test wei[ia, is, ie, isp] > (1-wei[ia, is, ie, isp])
                            # If closer to right grid point, weight should be less than 1-weight
                        elseif abs(agrid[(ib_pol[ia, is, ie, isp])] - ap[ia, is, ie, isp]) <
                            abs(ap[ia, is, ie, isp] - agrid[(ib_pol[ia, is, ie, isp] + 1)])
                            @test wei[ia, is, ie, isp] < (1-wei[ia, is, ie, isp])
                        end
                    end
                end
            end
        end
    end =#

#=    # Iterate asset transition matrix starting from KF_in
    dif = 1
    pd  = deepcopy(KF_in)
    while dif > tol
        pdi = zeros(S, na, ns)
        for i = 1:na
            for s = 1:ns
                sum = 0.0
                for iep = 1:ne
                    for si = 1:ns
                        pdi[ib_pol[i,s,iep,si], si] = wei[i,s,iep,si] * f[s,si] * g_of_e[iep] *
                            ewts[iep] * pd[i,s] .+ pdi[ib_pol[i,s,iep,si], si]

                        pdi[ib_pol[i,s,iep,si] + 1, si] = (1-wei[i,s,iep,si]) * f[s,si] * g_of_e[iep] *
                            ewts[iep] * pd[i,s] .+ pdi[ib_pol[i,s,iep,si] + 1, si]

                        sum += wei[i,s,iep,si]  * f[s,si] * g_of_e[iep] * ewts[iep] +
                               (1-wei[i,s,iep,si]) * f[s,si] * g_of_e[iep] * ewts[iep]
                    end
                end
                @test isapprox(sum, 1.0, atol = 1e-5)
            end
        end

        # check convergence
        dif = maximum(abs.(pdi - pd))

        # Make sure that distribution integrates to 1
        @test isapprox(sum(pdi), 1.0, atol = 1e-6)
        pd = deepcopy(pdi) #pd = deepcopy(pdi / sum(pdi))

      #=  p = plot(agrid, pd[:, 1], label = "low skill")
        plot!(p, agrid, pd[:, 2], label = "high skill")
        savefig(p, "agrid_vs_D_first.png") =#

        counter += 1
    end=#
#=
    if β < 0.0
        p = plot(agrid, pd[:, 1], label = "low skill")
        plot!(p, agrid, pd[:, 2]*20, label = "high skill (x20)")
        savefig(p, "agrid_vs_D.png")
    end

    if β < 0.0
        p = plot(bgrid, pd[:, 1], label = "low skill")
        plot!(p, bgrid, pd[:, 2]*20, label = "high skill (x20)")
        savefig(p, "bgrid_vs_D.png")
    end
=#
