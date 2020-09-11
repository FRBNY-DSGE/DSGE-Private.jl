function steadystate!(m::HetDSGEGovDebt;
                      βlo::S = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                      βhi::S = exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                      excess::S = 5000.,
                      tol::S = 1e-4,
                      maxit::Int64 = 20,
                      βband::S = 1e-2) where {S<:Real}
    # If we have already solved for βstar (i.e. it's not NaN) and we only want to
    # estimate the non steady state parameters, there's no need to recompute
    # elo/ehi, etc.
    if !isnan(m[:βstar].value) && get_setting(m, :estimate_only_non_steady_state_parameters)
        return
    else
        reset_grids!(m)
        nx = get_setting(m, :nx)
        ns = get_setting(m, :ns)
        ni = get_setting(m, :ni)
        ne = get_setting(m, :ne)

        # Determines whether random or not
        us, es = if get_setting(m, :fix_random_matrices)
            get_setting(m, :us), get_setting(m, :es)
        else
            generate_us_and_es(ni, ne)
        end

        # This is a test setting; TODO: Remove because now we can just fix the randomness
        if get_setting(m, :steady_state_only)
            find_steadystate!(m; βlo = βlo, βhi = βhi, excess = excess, tol = tol, maxit = maxit,
                              βband = βband)
            return
        else
            # Do you want to calibrate for matching the income moments?
            if get_setting(m, :calibrate_income_targets)

                target = get_setting(m, :calibration_targets)
                lower  = get_setting(m, :calibration_targets_lb)
                upper  = get_setting(m, :calibration_targets_ub)
                m[:sH_over_sL], m[:elo], _, _ = best_fit(m[:pLH].value, m[:pHL].value,
                                                         target, lower, upper, us, es)
            else
                m[:varlinc], m[:vardlinc] = skill_moments(m[:sH_over_sL].value, m[:elo].value,
                                                          m[:pLH].value, m[:pHL].value,
                                                          us, es, ni)
            end
            # Whether one calibrates or not, ehi needs to be updated here
            m[:ehi] = 2.0 - m[:elo].value

            # Construct Markov transition matrix for skill
            f, sgrid, swts, sscale = persistent_skill_process(m[:sH_over_sL].value, m[:pLH].value,
                                                              m[:pHL].value, get_setting(m, :ns))
            m.grids[:sgrid] = Grid(sgrid, swts, sscale)
            m.grids[:fgrid] = f

            xgrid, xwts, xlo, xhi, xscale = cash_grid(sgrid, m[:ωstar].value, m[:H].value,
                                                      m[:r].scaledvalue, m[:η].value,
                                                      m[:γ].scaledvalue,
                                                      m[:Tstar].value, m[:elo].value, nx)

            m.grids[:xgrid] = Grid(uniform_quadrature(xscale), xlo, xhi, nx, scale = xscale)
            m <= Setting(:xlo, xlo)
            m <= Setting(:xhi, xhi)
            m <= Setting(:xscale, xscale)
            xswts = kron(swts, xwts)
            m.grids[:weights_total] = xswts

            # Once have updated grids, can call steady state and compute other two moments
            find_steadystate!(m; βlo = βlo, βhi = βhi,
                              excess = excess, tol = tol, maxit = maxit,
                              βband = βband)
            m[:mpc] = ave_mpc(m[:μstar].value,   m[:cstar].value, xgrid, xswts, nx, ns)
            m[:pc0] = frac_zero(m[:μstar].value, m[:cstar].value, xgrid, xswts, ns)
        end
    end
end

function find_steadystate!(m::HetDSGEGovDebt;
                           βlo::S = 0.5*exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                           βhi::S = exp(m[:γ].scaledvalue)/(1 + m[:r].scaledvalue),
                           excess::S = 5000.,
                           tol::S = 1e-4,
                           maxit::Int64 = 100, #20,
                           βband::S = 1e-2) where {S<:Real}
    # Load settings
    nx = get_setting(m, :nx)
    ns = get_setting(m, :ns)
    ne = get_setting(m, :ne)

    xlo = get_setting(m, :xlo)
    xhi = get_setting(m, :xhi)

    elo = m[:elo].value
    ehi = m[:ehi].value

    pLH = m[:pLH].value
    pHL = m[:pHL].value

    # Load parameters
    R  = 1 + m[:r].scaledvalue
    H  = m[:H].value
    η  = m[:η].value
    γ  = m[:γ].scaledvalue
    ω  = m[:ωstar].value
    T  = m[:Tstar].value
    bg = m[:bg].value

    # Load grids
    xgrid = m.grids[:xgrid].points
    sgrid = m.grids[:sgrid].points
    xswts = m.grids[:weights_total]
    f     = m.grids[:fgrid]

    βlo_temp = βlo
    βhi_temp = βhi

    reject  = false
    counter = 1

    n  = ns*nx
    c  = zeros(n)
    bp = zeros(n)
    KF = zeros(n,n)
    μ  = zeros(n)

    # Initial guess
    β        = 1.0
    c_pol_in = (R-1) * repeat(xgrid, 1, ns, ne) .+ ω * H * repeat(sgrid', nx, 1, ne) # consume steady state (interest + labor) income
    KF_in    = fill(1.0/(ns*nx), nx, ns) # Uniform distribution

    # e shock, qfunction stuff
    qfunction(x::Float64) = mollifier_hetdsgegovdebt(x, ehi, elo) # g in the paper
    # egrid_gk = gauss(ne) # Changed to use FastGaussQuadrature instead
    egrid_gk = gausslegendre(ne)
    egrid    = transform_ab(elo, ehi, egrid_gk[1])
    # ι: iota
    ewts = egrid_gk[2]
    ewts = ewts / dot(qfunction.(egrid), ewts)
    g_of_e = qfunction.(egrid) # save after deciding grid instead of recomputing
    egrid = egrid ./ dot(g_of_e .* egrid, ewts)

    if get_setting(m, :use_last_βstar) && !isnan(m[:βstar].value)

        # This short-circuits computation of the policy function
        βlo = βhi = m[:βstar].value

    elseif !isnan(m[:βstar].value)

        # If one has computed β* before, we first bisect into a neighborhood around it
        βlo_temp = m[:βstar].value - βband
        βhi_temp = m[:βstar].value + βband

        c, c_pol_in, bp, ell, KF, reject = policy_hetdsgegovdebt(nx, ns, ne, βlo_temp, R, ω, H, η, T, γ,
                                                                 ehi, elo, xgrid, sgrid, xswts, c_pol_in, KF_in,
                                                                 pLH, pHL, f, egrid, ewts, g_of_e,
                                                                 damp = get_setting(m, :policy_damp),
                                                                 maxit = get_setting(m, :policy_maxit))
        excess_lo, μ = compute_excess(xswts, KF, bp, bg)

        if excess_lo < 0 && abs(excess_lo) > tol
            βlo = βlo_temp
            c, c_pol_in, bp, ell, KF, reject = policy_hetdsgegovdebt(nx, ns, ne, βhi_temp, R, ω, H, η, T, γ, ehi,
                                                                     elo, xgrid, sgrid, xswts, c_pol_in, KF_in,
                                                                     pLH, pHL, f, egrid, ewts, g_of_e,
                                                                     damp = get_setting(m, :policy_damp),
                                                                     maxit = get_setting(m, :policy_maxit))
            excess_hi, μ = compute_excess(xswts, KF, bp, bg)

            if excess_hi > 0
                βhi = βhi_temp
            end
        elseif excess_lo >= 0
            βhi = βlo_temp
        end
    end

    while abs(excess) > tol && counter < maxit # clearing markets
        @show counter, β
        β = (βlo + βhi) / 2.0 # Note that b/c while loop is inside a function, scoping permits β to be changed
        c, c_pol_in, bp, ell, KF, reject = policy_hetdsgegovdebt(nx, ns, ne, β, R, ω, H, η, T, γ, ehi, elo,
                                                                 xgrid, sgrid, xswts, c_pol_in, KF_in,
                                                                 pLH, pHL, f, egrid, ewts, g_of_e,
                                                                 damp = get_setting(m, :policy_damp),
                                                                 maxit = get_setting(m, :policy_maxit))
        excess, μ = compute_excess(xswts, KF, bp, bg)

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

    # If policy function does not converge, we signal to likelihood that should reject
    m <= Setting(:auto_reject, reject)
    m[:lstar]  = ell
    m[:cstar]  = c
    m[:μstar]  = μ
    m[:βstar]  = β
    m[:β_save] = β
    nothing
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

# Draw a sample of nodes from the skill distribution
function ssample(us::Matrix{S}, P::Matrix{S}, πss::AbstractArray,
                 ni::Int) where {S<:Real}
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
                       ni::Int) where {S<:Real}
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
                       es::Matrix{S}, ni::Int = 10000) where {S<:Real}
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
function loss(x::Vector{S}, target::Vector{S}) where {S<:Real}
```

Computes for given vector and target vector, the sum of absolute loss.
"""
loss(x::Vector{S}, target::Vector{S}) where {S<:Real} = sum(abs.(x-target))

"""
```
function best_fit(pLH::S, pHL::S, target::Vector{S}, lower::Vector{S}, upper::Vector{S},
                  us::Matrix{S}, es::Matrix{S}, max_iter::Int = 20,
                  initial_guess::Vector{S} = [6.3, 0.03]) where {S<:Real}
```

Uses Nelder-Mead (gradient descent algorithm) to optimize for income moments.
"""
function best_fit(pLH::S, pHL::S, target::Vector{S}, lower::Vector{S}, upper::Vector{S},
                  us::Matrix{S}, es::Matrix{S}, max_iter::Int = 20,
                  initial_guess::Vector{S} = [6.3, 0.03]) where {S<:Real}

    skill_moments_f(x) = loss(collect(skill_moments(x[1], x[2], pLH, pHL, us, es)), target)

    res = optimize(skill_moments_f, lower, upper, initial_guess, Fminbox(NelderMead()),
                   Optim.Options(f_calls_limit = max_iter))

    sH_over_sL_argmin, elo_argmin = Optim.minimizer(res)
    min_varlinc, min_vardlinc = skill_moments(sH_over_sL_argmin, elo_argmin, pLH,
                                              pHL, us, es)
    return sH_over_sL_argmin, elo_argmin, min_varlinc, min_vardlinc
end

function esample(ue::Matrix{S}, egrid::AbstractArray, ecdf::AbstractArray,
                 ni::Int, ne::Int) where {S<:Real}
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

@inline function compute_excess(xswts::Vector{S}, KF::Matrix{S}, bp::Matrix{S},
                                bg::S, print_warning::Bool = false,
                                tol::S = 2e-1) where {S<:Float64}
  #=  LPMKF = xswts[1] * KF

    # Find eigenvalue closest to 1
    (D,V) = (eigen(LPMKF)...,)
    max_D = argmax(abs.(D))
    D = D[max_D]

    if abs(D - 1) > tol && print_warning
        @warn "Your eigenvalue is too far from 1, something is wrong."
    end

    # Pick eigenvector associated w/ largest eigenvalue\beta and moving it back to values
    μ = real(V[:,max_D])
    μ = μ ./ dot(xswts, μ) # Scale of eigenvectors not determinate: rescale to integrate to 1
    excess = dot(xswts, (μ .* bp)) - bg # Compute excess supply of savings, which is a fn of w
    return excess, μ =#
    @show sum(KF .* bp) - bg
    return sum(KF .* bp) - bg, vec(KF) #return sum(sum(KF, dims = 2) .* bp) - bg, vec(KF) #dot(vec(KF), bp) - bg
end

function transform_ab(a::Float64, b::Float64, grid::Vector{Float64})
    xs = ((b-a)/2) .* grid .+ (a+b)/2
    return xs
end


function policy_hetdsgegovdebt(nx::Int, ns::Int, ne::Int, β::S, R::S, ω::S, H::S, η::S,
                               T::S, γ::S, ehi::S, elo::S, xgrid::Vector{S},
                               sgrid::Vector{S}, xswts::Vector{S}, c_pol::Array{S}, KF_in::Array{S},
                               pLH::S, pHL::S, f::Matrix{S}, egrid::Vector{Float64}, ewts::Vector{Float64}, g_of_e::Vector{Float64},
                               dist::S = 1., tol::S = 1e-10;
                               maxit::Int64 = 500, damp::S = 0.5) where {S<:Real}
    n  = nx*ns
    nx_c = 25

    counter = 1
    reject = false

    # Constrainted consumption
    c_constrained = Matrix{Float64}(undef, ns, ne)
    for is in 1:ns
        for ie in 1:ne
            # min of this (below) and e=1)
            e = egrid[ie] # minimum([egrid[ie], 1.0])
            c_constrained[is, ie] = ω*sgrid[is]*e*H + T
        end
    end

    c_poli = deepcopy(c_pol)
    dist = 1

    bmin = 0.0
    bmax = maximum(exp(γ)*xgrid)
    bgrid = bmin .+ ((1:nx)/nx).^2 * (bmax - bmin) # Inefficient, should construct bgrid in advance
    # a grid implied by bgrid is: map b into a using equation at top of page 4
    agrid = Array{Float64}(undef, nx, ns, ne)
    for is in 1:ns
        for ie in 1:ne
            agrid[:, is, ie] = ω*sgrid[is]*egrid[ie]*H .+ T .+ exp(-γ)*bgrid
        end
    end


    f = [[1-pLH pLH];[pHL 1-pHL]] # f1[i,j] is prob of going from i to j

    while dist>tol && counter<maxit
        for is in 1:ns # TODO: switch to use reflect column major order, doesn't seem like it'd be a problem to do so
            for ie in 1:ne
                # Keep only non-constrained
                non_c_inds = vec(exp(γ)*((bgrid ./ R) .- ω*sgrid[is]*egrid[ie]*H .-
                                         T .+ c_pol[:, is, ie])) .> 0.0
                bp = bgrid[non_c_inds, :]

                # Sums
                sum_term = zeros(length(bp)) # Have to create here b/c depend on bp
                verify_one = 0.0

                ## Outer sum over s'
#=
                for isp in 1:ns
                    ## Inner integral over e'
                    for iep in 1:ne
                        #              p(s'|s)    *  iota(e') * g(e')         * c(a, s')⁻¹
                        sum_term  .+= (f[is, isp] * ewts[iep] * g_of_e[iep]) ./ c_pol[non_c_inds, isp, iep]
                        verify_one += f[is, isp]  * ewts[iep] * g_of_e[iep]
                    end
                end
=#
                # But faster to do in reverse (note that it's just a double sum)
                for iep in 1:ne
                    for isp in 1:ns
                        sum_term  .+= (f[is, isp] * ewts[iep] * g_of_e[iep]) ./ c_pol[non_c_inds, isp, iep]
                        verify_one += f[is, isp]  * ewts[iep] * g_of_e[iep]
                    end
                end

                @test isapprox(verify_one, 1.0, atol = 1e-5) # TODO: ADD REMOVE CHECK OPTION

                # Compute ell(a, s) = β * R * exp(-γ) * (Σₛ∫ₑ)
                # l = β * R * exp(-γ) * sum_term        # Calculate below directly to save allocations

                # Compute consumption today: c(b', s) = 1/l(a, s)
                c = 1. / (β * R * exp(-γ) * sum_term)
                b = vec(exp(γ) * ((bp ./ R) .- ω * sgrid[is] * egrid[ie] * H .- T .+ c)) #_pol[:, is, ie]))

                if b[1] > 0. && c[1] > c_constrained[is, ie]
                    @test c_constrained[is, ie] < c[1] # TODO: ADD REMOVE CHECK OPTION
                    c_c = collect(range(c_constrained[is, ie], c[1], length = nx_c))
                    b_c = exp(γ) .* ((-ω * sgrid[is] * egrid[ie] * H - T) .+ c_c)
                    b = vcat(b_c[1:nx_c - 1], b)
                    c = vcat(c_c[1:nx_c - 1], c)
                end

                # Nearest points, linear interpolation
                c_poli[:, is, ie] = interp_one(b, c, bgrid)
            end
        end

        dist = maximum(abs.(c_pol - c_poli))

        c_pol = deepcopy(c_poli)
        counter += 1

        if counter == maxit
            @warn "Euler iteration did not converge"
            reject = true
          #  return Vector{Float64}(undef, nx*ns), Array{Float64}(undef, nx, ns, ne), Matrix{Float64}(undef, nx, ns),
          #  Array{Float64}(undef, nx, ns, ns), Matrix{Float64}(undef, nx, ns), reject
        end
    end

    # Interpolate the (a, s, e) grid back to the (a, s) grid.
    C_Final = Matrix{Float64}(undef, nx, ns) # even faster to allocate this ahead of time
    for is in 1:ns
        # Sort the a's and use those for everything
        sorted_inds = sortperm(vec(agrid[:, is, :]))

        # agrid is the grid of a implied by bgrid, whereas xgrid is the grid of a that's paseed in
        C_Final[:, is] = interp_one(vec(agrid[:, is, :])[sorted_inds], vec(c_pol[:, is, :])[sorted_inds], xgrid)
    end

    # Compute a' implied by interpolated C_Final (back on the usual grid of a)
    ap = Array{Float64}(undef, nx, ns, ne, ns)
    for is in 1:ns
        for isp in 1:ns # TODO: switch isp and iep loops
            for iep in 1:ne
                ap[:, isp, iep, is] = (ω * sgrid[isp] * egrid[iep] * H) .+ (R * exp(-γ)) .* (xgrid - C_Final[:, is])
            end
        end
    end
    ap_grid = uniform_quadrature(minimum(ap), maximum(ap) + .1, nx, scale = maximum(ap) + .1 - minimum(ap))[1] # TODO: REPLACE WITH JUST DIRECTLY CREATING THE GRID, AVOID EXTRA ALLOCATIONS
    b_grid_implied = Matrix{Float64}(undef, nx, ns)
    for is in 1:ns
        b_grid_implied[:, is] = exp(γ) .* (ap_grid .- ω * sgrid[is] * H .- T)
    end

    # Finding the ergodic distribution
    # Assign weights to adjacent grid points proportionally to distance
    ib_pol, wei = histc(ap, ap_grid) # TODO: REPLACE WITH fit(Histogram) if faster, but first fix use of global in histc

    # Check ib_pol and weights worked
    for ia in 1:nx # TODO: ASSORT INTO COLUMN MAJOR ORDER, this is row-major
        for is in 1:ns # TODO: ADD OPTION TO NOT CHECK
            for ie in 1:ne
                for isp in 1:ns
                    # Make sure ap lies between the two nearest grid points on either side
                    @test ap_grid[(ib_pol[ia, is, ie, isp])] <= ap[ia, is, ie, isp] <=
                        ap_grid[(ib_pol[ia, is, ie, isp] + 1)]
                    # If closer to left grid point, weight should be greater than 1-weight
                    if abs(ap_grid[(ib_pol[ia, is, ie, isp])] - ap[ia, is, ie, isp]) <
                        abs(ap[ia, is, ie, isp] - ap_grid[(ib_pol[ia, is, ie, isp] + 1)])
                        @test wei[ia, is, ie, isp] > (1-wei[ia, is, ie, isp])
                    # If closer to right grid point, weight should be less than 1-weight
                    elseif abs(ap_grid[(ib_pol[ia, is, ie, isp])] - ap[ia, is, ie, isp]) <
                        abs(ap[ia, is, ie, isp] - ap_grid[(ib_pol[ia, is, ie, isp] + 1)])
                        @test wei[ia, is, ie, isp] < (1-wei[ia, is, ie, isp])
                    end
                end
            end
        end
    end

    # Iterate asset transition matrix starting from uniform distribution to solve Kolmogorov forward equation
    dif = 1
    pd  = deepcopy(KF_in)
    while dif > tol
        pdi = zeros(nx, ns)
        for i = 1:nx
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
        pd = deepcopy(pdi / sum(pdi))
    end

    ell = 1 ./ C_Final
    return vec(C_Final), c_pol, b_grid_implied, ell, pd, reject
end

function policy_hetdsgegovdebt_old(nx::Int, ns::Int, β::S, R::S, ω::S, H::S, η::S,
                               T::S, γ::S, ehi::S, elo::S, xgrid::Vector{S},
                               sgrid::Vector{S}, xswts::Vector{S}, Win::Vector{S},
                               f::Matrix{S}, dist::S = 1., tol::S = 1e-4;
                               maxit::Int64 = 500, damp::S = 0.5) where {S<:Real}
    n    = nx*ns
    c    = zeros(n)                  # consumption
    bp   = Vector{Float64}(undef, n) # savings
    Wout = Vector{Float64}(undef, length(Win))
    counter = 1
    reject = false
    qfunction(x::Float64) = mollifier_hetdsgegovdebt(x, ehi, elo) # g in the paper

    while dist>tol && counter<maxit
        # compute c(w) given guess for Win = β*R*E[u'(c_{t+1})], Win is guess for marginal utility tomorrow
        for iss in 1:ns
            for ia in 1:nx
                c[nx*(iss-1)+ia] = min(1/Win[nx*(iss-1)+ia],xgrid[ia]+η) # c = min{1/ell_t(a, s), a) (page 7 paper).
                # consumption today is smaller of [inverse of marginal utility (unconstrained), cash on hand (constrained)]
            end
        end

        bp = R*(exp(-γ))*(repeat(xgrid, ns) - c)  # compute bp(w) given guess for Win, bp = (1+r_t)*exp(-e_{t+1})(a-c_t(a, s)) (thisi s inside the g function in defintion of elolo on page 7 of paper
        Wout = parameterized_expectations_hetdsgegovdebt(nx, ns, β, R, ω, H, T, γ,
                                                         qfunction, xgrid,
                                                         sgrid, xswts, c, bp, f)
        # W is ell_star
        dist = maximum(abs.(Wout - Win))
        Win  = damp*Wout + (1.0-damp) * Win
        counter += 1
    end
    if counter == maxit
        #@warn "Euler iteration did not converge"
        reject = true
        return c, bp, Wout, zeros(n,n), reject
    end
    tr = kolmogorov_fwd_hetdsgegovdebt(nx, ns, ω, H, T, R, γ, qfunction, xgrid, sgrid, bp, f)
    return c, bp, Wout, tr, reject
end

@inline function parameterized_expectations_hetdsgegovdebt(nx::Int, ns::Int, β::S,
                                                           R::S, ω::S, H::S, T::S, γ::S,
                                                           qfunc::Function, xgrid::Vector{S},
                                                           sgrid::Vector{S}, xswts::Vector{S},
                                                           c::Vector{S}, bp::Vector{S},
                                                           f::Matrix{S}) where {S<:Real}
    l_out = zeros(nx*ns)
    # Expectation at t (over s and a)
    for iss=1:ns
        for ia=1:nx
            sumn = 0.0
            # Sum over s'
            for isp=1:ns
                # Integral over a'
                for iap=1:nx
                    sumn += (xswts[nx*(isp-1)+iap]/c[nx*(isp-1)+iap]) *
                        #g(a'             - (1+r)exp(-gamma)(a-c) -T / w_{t+1}*H_{t+1}*s' p(s' | s)   / s' (the w and H are done below)
                        qfunc((xgrid[iap] - bp[nx*(iss-1)+ia] - T) / (ω*H*sgrid[isp])) * f[iss,isp] ./ sgrid[isp]
                end
            end
            # Stuff that pops ouit of integral/sum
            l_out[nx*(iss-1)+ia] = (β*R*(exp(-γ))/ω*H)*sumn
        end
    end
    return l_out
end

@inline function kolmogorov_fwd_hetdsgegovdebt(nx::Int, ns::Int, ω::S, H::S, T::S, R::S,
                                               γ::S, qfunc::Function, xgrid::Vector{S},
                                               sgrid::Vector{S}, bp::Vector{S},
                                               f::Matrix{S}) where {S<:Real}
    tr = zeros(nx*ns,nx*ns)
    for iss=1:ns
        for ia=1:nx
            for isp=1:ns
                for iap=1:nx
                    tr[nx*(isp-1)+iap, nx*(iss-1)+ia] = qfunc((xgrid[iap] -
                                                               bp[nx*(iss-1)+ia] - T) /
                                                              (ω*H*sgrid[isp])) * f[iss,isp] ./
                                                              (ω * H * sgrid[isp])
                end
            end
        end
    end
    return tr
end

@inline function mollifier_hetdsgegovdebt(e::S, ehi::S, elo::S) where {S<:Real}
    In = 0.443993816237631
    if e<ehi && e>elo
        temp = -1.0 + 2.0 * (e - elo) / (ehi - elo)
        return (2.0 / (ehi - elo)) * exp(-1.0 / (1.0 - temp^2)) / In
    end
    return 0.0
end

@inline function dmollifier_hetdsgegovdebt(x::S, ehi::S, elo::S) where {S<:Real}
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
