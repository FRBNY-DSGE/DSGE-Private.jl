function construct_gfunc!(m::HetDSGEGovDebt; recalculate_μ::Bool = true, check_egrid::Bool = false)
    gfunc_type = get_setting(m, :gfunc_type)
    if gfunc_type == :mollifier
        _mollifier_gfunc = (x) -> mollifier_hetdsgegovdebt(x, m[:ehi].value, m[:elo].value)

        return _mollifier_gfunc
    elseif gfunc_type == :lognormal
        # calculate mu (do we need to do this every steady state if ehi and elo are fixed?
        if recalculate_μ && !(m[:elo].fixed && m[:ehi].fixed)
            m[:μ_e].value = calc_lognormal_mu(m[:ehi].value, m[:elo].value, m[:σ_e].value)
        end
        _lognormal_gfunc = (x) -> lognormal_hetdsgegovdebt(x, m[:ehi].value, m[:elo].value, m[:μ_e].value, m[:σ_e].value)

        return _lognormal_gfunc
    else
        error("gfunc_type $(gfunc_type) is not recognized. It must be one of [:mollifier, :lognormal]")
    end
end

function construct_gfunc_wbounds!(m::HetDSGEGovDebt; recalculate_μ::Bool = true)
    gfunc_type = get_setting(m, :gfunc_type)
    if gfunc_type == :mollifier
        return (x, y, z) -> mollifier_hetdsgegovdebt(x, y, z)
    elseif gfunc_type == :lognormal
        # calculate mu (do we need to do this every steady state if ehi and elo are fixed?
        if recalculate_μ && !(m[:elo].fixed && m[:ehi].fixed)
            m[:μ_e].value = calc_lognormal_mu(m[:ehi].value, m[:elo].value, m[:σ_e].value)
        end

        return (x, y, z) -> lognormal_hetdsgegovdebt(x, y, z, m[:μ_e].value, m[:σ_e].value)
    else
        error("gfunc_type $(gfunc_type) is not recognized. It must be one of [:mollifier, :lognormal]")
    end
end

@inline function lognormal_hetdsgegovdebt(e::S0, ehi::S1, elo::S1, μ::S1, σ::S1) where {S0 <: Real, S1 <: Real}
    return pdf(truncated(LogNormal(μ, σ), elo, ehi), e)
end

#= @inline function lognormal_cdf(e::S0, ehi::S1, elo::S1, μ::S1, σ::S1) where {S0 <: Real, S1 <: Real}
    erf(x) = (2/π)*quadgk(exp(-x^2), 0, x, rtol=1e-3)
    return .5 + .5*erf((log(x)-μ)/(sqrt(2)*σ))
end =#

#= @inline function lognormal_pdf(e::S0, ehi::S1, elo::S1, μ::S1, σ::S1) where {S0 <: Real, S1 <: Real}
    return 1/(e*σ*sqrt(2*π))*exp(-(log(x)-μ)/(2*σ^2))
end =#

@inline function lognormal_mean(ehi::S1, elo::S1, μ::S1, σ::S1) where {S1 <: Real}
    Φ(x) = .5 + .5 * erf(x / sqrt(2.))
    x_0(x) = (log(x)-μ)/σ
    elo_0 = x_0(elo)
    ehi_0 = x_0(ehi)
    num1 = σ - elo_0
    num2 = σ - ehi_0

    return exp(μ + σ^2 / 2.) * (Φ(num1) - Φ(num2)) / (Φ(ehi_0) - Φ(elo_0))
end

@inline function calc_lognormal_mu(ehi::S1, elo::S1, σ::S1) where {S1 <: Real}
    function nl_mean!(F, x)
        F[1] = lognormal_mean(ehi, elo, x[1], σ) - 1.
    end
    results = nlsolve(nl_mean!, [σ^2 / 2.], iterations = 10000)
    if results.f_converged
        return results.zero[1]
    else
        throw(DomainError("Cannot find μ implying mean of 1 for truncated lognormal, given inputted ehi, elo, and σ"))
    end
end

@inline function mollifier_hetdsgegovdebt(e::S0, ehi::S1, elo::S1) where {S0 <: Real, S1 <: Real}
    In = 0.443993816237631
    if e<ehi && e>elo
        temp = -1.0 + 2.0 * (e - elo) / (ehi - elo)
        return (2.0 / (ehi - elo)) * exp(-1.0 / (1.0 - temp^2)) / In
    end
    return 0.0
end


@inline function dmollifier_hetdsgegovdebt(x::S0, ehi::S1, elo::S1) where {S0 <: Real, S1 <: Real}
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

# Save as above but assumes aswts = 1
function ave_mpc(m::AbstractArray, c::AbstractArray, agrid::AbstractArray,
                 na::Int, ns::Int)
    mpc = 0.
    for iss=1:ns
        i = na*(iss-1)+1
        mpc += m[i]*(c[i+1] - c[i])/(agrid[2] - agrid[1])
        for ia=2:na
            i = na*(iss-1)+ia
            mpc += m[i]*(c[i] - c[i-1])/(agrid[ia] - agrid[ia-1])
        end
    end
    return mpc
end

function frac_zero(m::AbstractArray, c::AbstractArray, agrid::AbstractArray,
                   aswts::AbstractArray, ns::Int; atol::Real = 1e-10)
    return sum(aswts .* m .* (isapprox.(c, repeat(agrid, ns); atol = atol)))
end

# same as above but assume aswts = 1
function frac_zero(m::AbstractArray, c::AbstractArray, agrid::AbstractArray,
                   ns::Int; atol::Real = 1e-10)
    return sum(m .* (isapprox.(c, repeat(agrid, ns); atol = atol)))
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

# For mollifier
function ln_annual_inc(ehist::Matrix{S}, us::Matrix{S}, elo::S, P::Matrix{S},
                       πss::AbstractArray, sgrid::AbstractArray,
                       ni::Int) where {S <: Real}
    s_inds = ssample(us,P,πss,ni)
    linc1 = zeros(ni)
    linc2 = zeros(ni)
    for i=1:ni
        inc1 = 0.
        for t=1:4
            eshock = 1. + (1. - elo)*(ehist[i,t]-1.) # this is just the actual eshock. Multiplying by (1 - elo) transforms ehist,
            sshock = (sgrid[1] + (sgrid[2] - sgrid[1])*(s_inds[i,t] - 1.)) # which was generated on a [0, 2] grid, onto [elo, 2].
            inc1 += eshock*sshock                    # Subtract 1 from ehist b/c re-centering
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

# For generic gfunc
function ln_annual_inc(ehist::Matrix{S}, us::Matrix{S}, elo::S, ehi::S, P::Matrix{S},
                       πss::AbstractArray, sgrid::AbstractArray,
                       ni::Int) where {S <: Real}
    s_inds = ssample(us,P,πss,ni)
    linc1 = zeros(ni)
    linc2 = zeros(ni)
    for i=1:ni
        inc1 = 0.
        for t=1:4
            eshock = ehist[i,t] # unlike mollifier, it is assumed that ehist is indeed the eshock
            sshock = (sgrid[1] + (sgrid[2] - sgrid[1])*(s_inds[i,t] - 1.))
            inc1 += eshock*sshock
        end
        inc2 = 0.
        for t=5:8
            eshock = ehist[i,t]
            sshock = (sgrid[1] + (sgrid[2] - sgrid[1])*(s_inds[i,t] - 1.))
            inc2 += eshock*sshock
        end
        linc1[i] += log(inc1)
        linc2[i] += log(inc2)
    end
    return linc1, linc2
end

# For mollifier
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

# Generic gfunc distribution for e shock
function skill_moments(sH_over_sL::S, elo::S, ehi::S, pLH::S, pHL::S, us::Matrix{S},
                       ue::Matrix{S}, gfunc_wbounds::Function, ne::Int, ni::Int = 10000) where {S <: Real}
    # Construct s transition dynamics
    πL    = pHL / (pLH + pHL)
    πss   = [πL; 1.0-πL]
    P     = [[1.0-pLH pLH]; [pHL 1.0-pHL]]
    slo   = 1.0 / (πL + (1-πL) * sH_over_sL)
    shi   = sH_over_sL * slo
    sgrid = [slo; shi]

    # Sample from gfunc for e shock
    egrid = collect(range(elo, stop = ehi, length = ne))
    eprob = [gfunc_wbounds(x, ehi, elo) / ne for x in egrid] # divide by ne b/c uniform quadrature
    eprob ./= sum(eprob)  # normalize discrete approximation to 1
    ecdf  = cumsum(eprob) # get CDF to use fact F(e) ∼ U[0, 1]
    es    = esample(ue, egrid, ecdf, ni, ne)

    linc1, linc2 = ln_annual_inc(es, us, elo, ehi, P, πss, sgrid, ni)
    return var(linc1), var(linc2 - linc1)
end

#=
"""
```
loss(x::Vector{S}, target::Vector{S}) where {S <: Real}
```

Computes for given vector and target vector, the sum of absolute loss.
"""
=#
loss(x::Vector{S}, target::Vector{S}) where {S <: Real} = sum(abs.(x-target))

#=
"""
```
best_fit(pLH::S, pHL::S, target::Vector{S}, lower::Vector{S}, upper::Vector{S},
         us::Matrix{S}, es::Matrix{S}, max_iter::Int = 20,
         initial_guess::Vector{S} = [6.3, 0.03]) where {S <: Real}
```

Uses Nelder-Mead (gradient descent algorithm) to optimize for income moments.
"""
=#
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

function compute_income_process_parameters(m::AbstractDSGEModel, gfunc::Function)
    # Determines whether random or not
    error("This function is not properly formed")
    us, es = if get_setting(m, :fix_random_matrices)
        get_setting(m, :us), get_setting(m, :es)
    else
        generate_us_and_es(ni, ne, gfunc)
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
    eave = 0.5*egrid[1:ne-1]+0.5*egrid[2:ne] # midpoint of egrid
    es = zeros(ni,8)
    for i=1:ni
        for t=1:8
            for ie=1:ne-1
                if ecdf[ie] < ue[i,t] <= ecdf[ie+1]
                    es[i,t] = eave[ie] # sample using fact F(e) ∼ U[0, 1] and assign draw to midpoint of egrid
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

function transform_ab(a::S, b::S, grid::AbstractVector{S}) where {S <: Real}
    xs = ((b-a)/2) .* grid .+ (a+b)/2
    return xs
end

# function construct_egrid(ehi::S, elo::S, ne::Int, gfunc::Function) where {S <: Real}
function construct_egrid(ehi::S, elo::S, ne::Int, gfunc_wbounds::Function) where {S <: Real}
    # Construct egrid
    gfunc(x)    = gfunc_wbounds(x, ehi, elo)
    egrid, ewts = gausslegendre(ne) # ne point gauss-legendre quadrature -> xᵢ, wᵢ
    egrid      .= transform_ab(elo, ehi, egrid) # xᵢ∈[-1, 1] -> eᵢ∈[elo, ehi]

    # Normalize egrid, ewts so that ∫ g(e) de ≈ ∑ᵢ g(eᵢ) * wᵢ = 1 b/c g(e) is a pdf
    g_of_e      = map(x -> gfunc(x), egrid) # g(eᵢ)
    if all(g_of_e .≈ 0.)
        throw(MalformedTruncationError("The gfunc is not properly formed. All quadrature points yield zero mass."))
    end
    ewts      ./= dot(g_of_e, ewts) # normalize wᵢ to w̃ᵢ so that ∑ᵢ w̃ᵢ * gᵢ = 1
    egrid     ./= dot(g_of_e .* egrid, ewts) # Normalize egrid so that mean ∫ e g(e) de = 1, should be a small adjustment
#=    if any(isnan.(egrid))
        @show "Calculated a NaN egrid"
        @show (ehi, elo, ne)
        @show ewts
        @show gausslegendre(ne)

        egrid, ewts = gausslegendre(ne) # ne point gauss-legendre quadrature -> xᵢ, wᵢ
        @show egrid
        egrid      .= transform_ab(elo, ehi, egrid) # xᵢ∈[-1, 1] -> eᵢ∈[elo, ehi]
        @show egrid

        # Normalize egrid, ewts so that ∫ g(e) de ≈ ∑ᵢ g(eᵢ) * wᵢ = 1 b/c g(e) is a pdf
        g_of_e      = map(x -> gfunc(x), egrid) # g(eᵢ)
        @show g_of_e
        pts = range(0., stop = 3., length = 1000)
        for pt in pts
            if !(gfunc(pt) ≈ 0)
                @show "Found lower point for which g_of_e is not zero"
                @show pt
                break
            end
        end
        for pt in reverse(pts)
            if !(gfunc(pt) ≈ 0)
                @show "Found upper point for which g_of_e is not zero"
                @show pt
                break
            end
        end
        ewts      ./= dot(g_of_e, ewts) # normalize wᵢ to w̃ᵢ so that ∑ᵢ w̃ᵢ * gᵢ = 1
        @show ewts
        egrid     ./= dot(g_of_e .* egrid, ewts) # Normalize egrid so that mean ∫ e g(e) de = 1, should be a small adjustment
    end

    if any(isnan.(egrid))
        @assert m[:ehi].value == ehi "ehi matches what is stored in m"
        @assert m[:elo].value == elo "elo matches what is stored in m"
        gfunc = construct_gfunc!(m; recalculate_μ = false)
        egrid, ewts = gausslegendre(ne) # ne point gauss-legendre quadrature -> xᵢ, wᵢ
        egrid      .= transform_ab(elo, ehi, egrid) # xᵢ∈[-1, 1] -> eᵢ∈[elo, ehi]

        # Normalize egrid, ewts so that ∫ g(e) de ≈ ∑ᵢ g(eᵢ) * wᵢ = 1 b/c g(e) is a pdf
        g_of_e      = map(x -> gfunc(x), egrid) # g(eᵢ)
        ewts      ./= dot(g_of_e, ewts) # normalize wᵢ to w̃ᵢ so that ∑ᵢ w̃ᵢ * gᵢ = 1
        egrid     ./= dot(g_of_e .* egrid, ewts) # Normalize egrid so that mean ∫ e g(e) de = 1, should be a small adjustment

        @assert !(any(isnan.(egrid))) "Even after reconstructing gfunc, we get NaNs in egrid"
    end
    @assert !(any(isnan.(egrid)))
=#

    return egrid, ewts, g_of_e
end

function construct_agrid(smin::S, emin::S, ω::S, H::S, R::S, η::S, γ::S, T::S, na::Int;
                         ahi_inc::S = NaN) where {S <: Real}

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

@inline function construct_bgrid_agrid_big(agrid::AbstractVector{S}, sgrid::AbstractVector{S}, egrid::AbstractVector{S},
                                           na::Int, ns::Int, ne::Int, γ::S, ω::S, H::S, T::S, R::S) where {S <: Real}

    bgrid = construct_bgrid(agrid, sgrid, egrid, na, γ, ω, H, T, R)
    agrid_big = construct_agrid_big(bgrid, sgrid, egrid, na, ns, ne, γ, ω, H, T, R)

    return bgrid, agrid_big
end

@inline function construct_agrid_big(bgrid::AbstractVector{S1}, sgrid::AbstractVector{S1}, egrid::AbstractVector{S1},
                                     na::Int, ns::Int, ne::Int, γ::S2, ω::S2, H::S2, T::S2, R::S2) where {S1 <: Real, S2 <: Real}

    # a grid (cash on hand) implied by bgrid (assets) is: map b into a using equation at top of page 4
    agrid_big = Array{S2}(undef, na, ns, ne)
    for ie in 1:ne
        for is in 1:ns
            agrid_big[:, is, ie] = (ω * sgrid[is] * egrid[ie] * H + T) .+ exp(-γ) .* bgrid
        end
    end

    return agrid_big
end

@inline function construct_bgrid(agrid::AbstractVector{S}, sgrid::AbstractVector{S}, egrid::AbstractVector{S},
                                 na::Int, γ::S, ω::S, H::S, T::S, R::S) where {S <: Real}
    bmin = 0.0
    bmax = exp(γ) * (maximum(agrid) - ω * maximum(sgrid) * maximum(egrid) * H) # NOTE egrid is not exactly in [elo, ehi] b/c
    return bmin .+ ((1:na) / na).^2 * (bmax - bmin)                           # slightly renormalized to ensure ∫e g(e) de = 1
end

function persistent_skill_process(sH_over_sL::AbstractFloat, pLH::AbstractFloat,
                                  pHL::AbstractFloat, ns::Int)
    f1 = [[1-pLH pLH];[pHL 1-pHL]] # f1[i,j] is prob of going from i to j
    ss_skill_distr = [pHL/(pLH+pHL); pLH/(pLH+pHL)]
    slo    = 1.0 / (ss_skill_distr'*[1;sH_over_sL])
    sgrid  = slo*[1;sH_over_sL]
    sscale = sgrid[2] - sgrid[1]
    swts   = (sscale/ns)*ones(ns) # Quadrature weights
    f      = f1 ./ repeat(swts',ns,1)
    return f, sgrid, swts, sscale
end

#=
"""
```
interp_one(xpts::T, ypts::T, x::T) where {T<:Vector{Float64}}
```
Mimics behavior of interp1 in MATLAB. Takes set {(x_i, y_i)}, linearly interpolates between each (x_i, y_i) and (x_{i+1}, y_{i+1}), and then returns vector y for interpolated points in x.
"""
=#
function interp_one(xpts::T1, ypts::T2,
                    x::T3) where {T1 <: AbstractVector{<: Real}, T2 <: AbstractVector{<: Real}, T3 <: AbstractVector{<: Real}}
    @assert length(xpts) == length(ypts)
    intf = extrapolate(interpolate((xpts,), ypts, Gridded(Linear())), Line())
    y = [intf(v) for v in x]
    return y
end

#=
"""
```
histc(points, grid)
```
Mimics behavior of histc function in MATLAB.
"""
=#
function histc(points, grid) # TODO: see if I can avoid declaring these variables global, e.g. found probably doesn't need to be
    N      = size(grid, 1)
    ib_pol = zeros(Int64, size(points))
    points_copy = deepcopy(points)
    for i in eachindex(points)
        p = points[i]
        global min_ind, max_ind = 1, N
        global found = false

        if p > grid[max_ind]
           # @warn "Point not in grid"
            found = true
            ib_pol[i] = max_ind - 1
            points_copy[i] = grid[ib_pol[i]]
        end
        if p < grid[min_ind]
            ib_pol[i] = min_ind
            points_copy[i] = grid[ib_pol[i]]
            found = true
        end

        while !found
            ind = Int(floor((min_ind + max_ind) / 2))
            if p >= grid[ind]
                if p < grid[ind+1]
                    found = true
                    ib_pol[i] = ind
                elseif p == grid[ind+1]
                    found = true
                    ib_pol[i] = ind+1
                else
                    global min_ind = ind
                    if min_ind == max_ind - 1
                        found = true
                        ib_pol[i] = ind
                    end
                end
            else
                global max_ind = ind
            end
        end
    end
 #   ib_pol[ib_pol .== length(grid)] .= length(grid)-1
    wei = 1 .- ((points_copy - grid[ib_pol]) ./
        (grid[ib_pol.+1] - grid[ib_pol]))

    return ib_pol, wei
end

#=
"""
```
generate_us_and_es(ni, ne, gfunc, gfunc_type)
```

There is no need to recall this function, unless one wants to undo the seeding
in all past saved output.

The us are draws from U[0, 1] used to construct nodes for the skill distribution.

The es are the grid nodes for the exogenous i.i.d productivity shock whose
pdf is given by `gfunc` and name by `gfunc_type`.
"""=#
function generate_us_and_es(ni, ne, gfunc::Function, gfunc_type::Symbol)
    us = rand(ni, 8)
    ue = rand(ni, 8)

    egrid  = collect(range(0., stop = 2., length = ne))
    #eprob  = [2*gfunc(egrid[i], 2., 0.) / ne for i=1:ne]
    eprob  = [2*gfunc(egrid[i], 2., 0.) / ne for i=1:ne]
    eprob /= sum(eprob)

    ecdf = cumsum(eprob)
    eave = 0.5 * egrid[1:ne-1] + 0.5 * egrid[2:ne]
    es   = esample(ue, egrid, ecdf, ni, ne)
#=
    JLD2.jldopen("$HETDSGEGOVDEBT/us_es_$(gfunc_type).jld2", true, true, true, IOStream) do file
        file["us"] = us
        file["es"] = es
    end
=#
    return us, es
end

#=
"""
```
generate_us_and_ue(ni, ne)
```

There is no need to recall this function, unless one wants to undo the seeding
in all past saved output.

The us are draws from U[0, 1] used to construct nodes for the skill distribution.

The ue are draws from U[0, 1] used to construct nodes for the exogenous i.i.d productivity shock
whose pdf is given by `gfunc` and name by `gfunc_type`.
"""=#
function generate_us_and_ue(ni, ne, gfunc_type::Symbol, filepath::String)
    us = rand(ni, 8)
    ue = rand(ni, 8)
#=
    JLD2.jldopen("$(filepath)/us_ue_$(gfunc_type).jld2", true, true, true, IOStream) do file
        file["us"] = us
        file["ue"] = ue
    end
=#
    return us, ue
end

function print_summary_stats(m::HetDSGEGovDebt)
    na = get_setting(m, :na)
    agrid = m.grids[:agrid].points
    los_save = sum(m[:Dstar].value[1:na] .* (agrid - m[:cstar].value[1:na]))
    his_save = sum(m[:Dstar].value[1+na:end] .* (agrid - m[:cstar].value[1+na:end]))
    los_C    = sum(m[:Dstar].value[1:na] .* m[:cstar].value[1:na])
    his_C    = sum(m[:Dstar].value[1+na:end] .* m[:cstar].value[1+na:end])
    println("Aggregate savings by low and high skill workers (resp.):            " *
            "($(round(los_save, digits = 3)), " * "$(round(his_save, digits = 3)))")
    println("Aggregate consumption by low and high skill workers (resp.):        " *
            "($(round(los_C, digits = 3)), " * "$(round(his_C, digits = 3)))")
end
#=
"""
```
CashOnHandError <: Exception
```
A `CashOnHandError` is thrown when:

1. The implied assets today `b` are all negative during the Euler iteration.
2. The consumption policy at the smallest `b` value is smaller than the constrained consumption policy.
3. Some assets today `b` for constrained agents are negative.
4. Consumption is not always less than cash on hand after mapping the consumption policy from `(b, s, e)` to `(a, s)`.
"""
=#
mutable struct CashOnHandError <: Exception
    msg::String
end
CashOnHandError() = CashOnHandError("Consumption policy is not consistent with cash on hand.")
Base.showerror(io::IO, ex::CashOnHandError) = print(io, ex.msg)

mutable struct MalformedTruncationError <: Exception
    msg::String
end
Base.showerror(io::IO, ex::MalformedTruncationError) = print(io, ex.msg)
