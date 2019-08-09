"""
```
solve(m::AbstractModel; apply_altpolicy = false)
```

Driver to compute the model solution and augment transition matrices.

### Inputs

- `m`: the model object

## Keyword Arguments

- `apply_altpolicy::Bool`: whether or not to solve the model under the
  alternative policy. This should be `true` when we solve the model to
  forecast, but `false` when computing smoothed historical states (since
  the past was estimated under the baseline rule).

### Outputs
 - TTT, RRR, and CCC matrices of the state transition equation:
    ```
    S_t = TTT*S_{t-1} + RRR*ϵ_t + CCC
    ```
"""
function solve(m::AbstractModel; apply_altpolicy = false, verbose::Symbol = :high)

    altpolicy_solve = alternative_policy(m).solve


    if altpolicy_solve == solve || !apply_altpolicy

        # Get equilibrium condition matrices
        Γ0, Γ1, C, Ψ, Π  = eqcond(m)

        # Solve model
        #TTT_gensys, CCC_gensys, RRR_gensys, fmat, fwt, ywt, gev, eu, loose =
        #    gensys(Γ0, Γ1, C, Ψ, Π, 1+1e-6, verbose = verbose)
        TTT_gensys, CCC_gensys, RRR_gensys, eu = gensys(Γ0, Γ1, C, Ψ, Π, 1+1e-6, verbose = verbose)

        # Check for LAPACK exception, existence and uniqueness
        if eu[1] != 1 || eu[2] != 1
            throw(GensysError())
        end

        TTT_gensys = real(TTT_gensys)
        RRR_gensys = real(RRR_gensys)
        CCC_gensys = real(CCC_gensys)

        # Augment states
        TTT, RRR, CCC = augment_states(m, TTT_gensys, RRR_gensys, CCC_gensys)

    else
        # Change the policy rule
        TTT, RRR, CCC = altpolicy_solve(m)
    end

    return TTT, RRR, CCC
end

"""
```
solve(m::GHLS; parallel = false)
```

#  Arguments:
- `m::GHLS`: The non-linear GHLS model object
- `parallel::Bool`: Whether or not to solve the model in parallel.

# Description:
Computes the model solution and corresponding coefficients matrix α⋆ (see a in Equation 2.21 in Technical Appendix of GHLS (2017))
"""
function solve(m::GHLS, parallel::Bool=false)

    # Create the shock grid for use in interpolation
    m.approx.exoggrid, m.approx.shockbounds, m.approx.shockdistance = gen_shockgrid(m.approx.nshockgrid, m.approx.nexogshocks, m.approx.ns, m.approx.nexogvars,m.parameters,m.keys)

    # Get equilibrium conditions of linearized solution and run gensys to put in state-space form
    Γ0, Γ1, C, Ψ, Π = eqcond(m)
    TTT_gensys, CCC_gensys, RRR_gensys, eu = gensys(Γ0, Γ1, C, Ψ, Π, 1+1e-6, verbose = :high)

    # Check for LAPACK exception, existence and uniqueness
    if eu[1] != 1 || eu[2] != 1
        throw(GensysError())
    end

    TTT = real(TTT_gensys)
    RRR = real(RRR_gensys)

    QQQ = zeros(8, 8)
    QQQ[1, 1] = m[:σ_η]^2.0
    QQQ[2, 2] = m[:σ_μ]^2.0
    QQQ[3, 3] = m[:σ_Z]^2.0
    QQQ[4, 4] = m[:σ_R]^2.0
    QQQ[5, 5] = m[:σ_g]^2.0
    QQQ[6, 6] = m[:σ_elast]^2.0
    QQQ[7, 7] = m[:σ_elastw]^2.0
    QQQ[8, 8] = 0.

    RRR = RRR*sqrt.(QQQ)

    # For the linear simulation we only need the portion of the state-space matrices corresponding to variables in the non-linear version of the model
    ntotalvars = m.approx.nendogvars + m.approx.nexogvars
    pp = TTT[1:ntotalvars, 1:ntotalvars]
    sigma = RRR[1:ntotalvars, 1:m.approx.nexogvars]

    # Get steady state values
    steady_states = [i.value for i in m.steady_state]

    # Compute ergodic means of the endogenous variables (endog_emean), frequency of encountering zero lower bound, bounds of high probability region of endogenous variable domain, and at which exogenous states we hit zlb frequently
    m.approx.endog_emean, m.approx.zlbfrequency, m.approx.msvbounds, m.approx.statezlbinfo, convergence = simulate_linear(m.approx.ns, m.approx.nendogvars, m.approx.nexogvars, m.approx.nmsv, m.approx.nexogshocks, steady_states, m.endogenous_states[:rm_t], m.approx.nshockgrid, m.approx.shockbounds, m.approx.shockdistance, pp, sigma)

    # Find the slopes coefficients and constants to go from msv space to [-1,1] domain and vice-versa.
    m.approx.slopeconmsv, m.approx.slopeconxx = create_slopes(m.approx.nmsv, m.approx.msvbounds)

    # Transform linear solution state-space into representation using shocks directly rather than innovations:
    #    y _t = A*y_{t-1} + B*s_t where A is aalin, B is bblin, y_t is endogenous states and s_t are the shocks.
    # Used to get initial guess for nonlinear solution when fed into initial_α
    aalin, bblin = lindecrule_markov(pp, sigma, m.approx.nendogvars, m.approx.nexogvars, m.approx.nexogshocks)

    # Construct starting guess for the α matrix at solution by using the values of α at the linearized solution
    α_initial = initial_α(m.approx.nendogvars, m.approx.nexogvars, m.approx.nexogshocks, m.approx.nmsv, m.approx.ns, m.approx.ngridpoints, m.approx.exoggrid, steady_states, m.endogenous_states, m.approx.slopeconxx, m.approx.xgrid, m.approx.nfunc, m.approx.bbtinv, aalin, bblin)

    # Runs the fixedpoint convergence algorithm to find true non-linear solution and associated α coefficients
    α_star, convergence = if parallel
        fixedpoint_parallel(m[:rkss].value, m.approx, m.parameters, m.keys, m[:labss].value, m.exogenous_shocks, m.endogenous_states, α_initial, get_setting(m, :zero_lower_bound))
    else
        fixedpoint(m[:rkss].value, m.approx, m.parameters, m.keys, m[:labss].value, m.exogenous_shocks, m.endogenous_states, α_initial, get_setting(m, :zero_lower_bound))
    end

    return α_star
end

"""
    create_slopes(nmsv::Int, msvbounds::Array{Float64, 1})

# Arguments:
- `nmsv::Int`: Number of minimum state endogenous variables
- `msvbounds::Array{Float64,1}`: Bounds on the high probability region of the minimum state variables (first nmsv are lower, rest are uper).

# Description:
Find the slopes coefficients and constants to go from msv space to [-1,1] domain and vice-versa. See page 65 in Gust et. al (2017).
"""
function create_slopes(nmsv::Int, msvbounds::Array{Float64, 1})


    # Conversion from state domain (msv) to [-1, 1] domain (xx)
    slopeconmsv = zeros(2*nmsv)
    slopeconmsv[1:nmsv] = 2.0 ./  (msvbounds[nmsv + 1 :  2*nmsv] - msvbounds[1:nmsv])
    slopeconmsv[nmsv+1:2*nmsv] = -2.0 * msvbounds[1:nmsv] ./ (msvbounds[nmsv + 1 : 2*nmsv] .- msvbounds[1:nmsv]) .- 1.

    # Conversion from xx to msv domains
    slopeconxx = zeros(2*nmsv)
    slopeconxx[1:nmsv] =  0.5 * (msvbounds[nmsv + 1 : 2*nmsv] - msvbounds[1:nmsv])
    slopeconxx[nmsv+1:2*nmsv] = msvbounds[1:nmsv] + 0.5 * (msvbounds[nmsv + 1 : 2*nmsv] - msvbounds[1:nmsv])

    return slopeconmsv, slopeconxx
end

"""
    lindecrule_markov(pp::Array{Float64, 2}, sigma::Array{Float64, 2}, nendogvars::Int, nexogvars::Int, nexogshocks::In

# Arguments
- `pp::Array{Float64,2}`: Feedback (endogenous variable lags) part of the linear transition equation.
- `sigma::Array{Float64,2}`: Innovation part of the linear transition equation.
- `nendogvars::Int`: Number of endogenous variables.
- `nexogvars::Int`: Number of exogenous variables (shocks and fixed values).
- `nexogshocks::Int`: Number of active exogenous shocks (with strictly greater than one possible value)

# Description:
Transform linear solution state-space into representation using shocks directly rather than innovations. Used to get initial guess for nonlinear solution.
The transformed system has the form:
    y_t = A y_{t-1} + B s_{t},
where y_t are the endogenous variables excluding the shocks and s_t are the shocks.
...
"""
function lindecrule_markov(pp::Array{Float64, 2}, sigma::Array{Float64, 2}, nendogvars::Int, nexogvars::Int, nexogshocks::Int)

    #Initilize Variables
    bblin=zeros(nendogvars,nexogvars)

    @fastmath @inbounds @simd for i in 1:nexogshocks
        bblin[:,i] = sigma[1:nendogvars,i]/sigma[nendogvars+i,i]
    end

    return pp[1:nendogvars,1:nendogvars],bblin

end


"""
    simulate_linear(ns::Int, nendogvars::Int, nexogvars::Int, nmsv::Int, nexogshocks::Int, steady_states::Array{Float64, 1}, rm_t::Int, nshockgrid:: Array{Int, 1}, shockbounds::Array{Float64,2}, shockdistance::Array{Float64,1}, pp::Array{Float64,2}, sigma::Array{Float64, 2})

# Arguments:
- `ns::Int`: Total number of grid points on exogenous shock grid (equal to the number of distinct possible combinations of shock values)
- `nendogvars::Int`: Number of endogenous variables
- `nexogvars:Int`:  Number of exogenous variables
- `nmsv::Int`: Number of minimum state endogenous variables
- `nexogshocks::Int`: Number of active exogenous shocks (those with strictly greater than one possible values)
- `steady_states::Vector{Float64}`: The model's steady state values
- `rm_t::Int`: Index of rm_t
- `nshockgrid::Vector{Int}`: Vector containing number of possible values for each shock.
- `shockbounds::Array{Float64,2}`: Lower and upper bounds for each exogenous shock on the exogenous shock grid.
- `shockdistance::Array{Float64,1}`: Distance between grid points for each shock.
- `pp::Array{Float64,2}`: Feedback (endogenous variable lags) part of the linear transition equation.
- `sigma::Array{Float64,2}`: Innovation part of the linear transition equation.

Description:
Simulates a linearized version of the model in order to compute high probability region of endogenous variable domain as well as frequency of hitting zero lower bound (and at which states)
"""
function simulate_linear(ns::Int, nendogvars::Int, nexogvars::Int, nmsv::Int, nexogshocks::Int, steady_states::Array{Float64, 1}, rm_t::Int, nshockgrid:: Array{Int, 1}, shockbounds::Array{Float64,2}, shockdistance::Array{Float64,1}, pp::Array{Float64,2}, sigma::Array{Float64, 2})

    # Initialize variables
    total_periods = 100000
    periods_per_iter = 400
    countzlb = 0
    count_exploded = 0
    counter = 1
    convergence = true

    statezlbinfo = zeros(Int64, ns)
    endog_emean = Array{Float64}(undef, nendogvars + nexogvars)
    msvbounds = Array{Float64}(undef, 2*nmsv)
    shockindex = Array{Int64}(undef, nexogvars)
    countzlbstates = zeros(Int64, ns)
    msvhigh = Array{Float64}(undef, nmsv)
    msvlow = Array{Float64}(undef, nmsv)
    innovations = Array{Float64}(undef, nexogvars)
    xrandn = Array{Float64}(undef, nexogvars, total_periods)
    msv_std = zeros(nmsv)
    endogvar = zeros(nendogvars+nexogvars, total_periods+1)

    # Set up random generator
    path = dirname(@__FILE__)
    iseed = MersenneTwister(101294)
    randn!(iseed, xrandn)

    #For testing
    xrandnFortran=readdlm("$path/xrandn.txt")
    xrandn=reshape(xrandnFortran,nexogvars,total_periods)

    # Set up for first period
    endogvar[1:nmsv,1]= steady_states[1:nmsv]
    msvhigh = log(2.0) + endogvar[1:nmsv, 1]
    msvlow = log(0.01) + endogvar[1:nmsv, 1]

    #Zero out shocks not included in nonlinear model
    if (nexogshocks < nexogvars)
        sigma[:,nexogshocks+1:nexogvars] .= 0.0
    end
    stateindex = 0

    while counter <= total_periods + 1
        explosiveerror = false
        llim = counter+1
        ulim = min(total_periods+1,counter+periods_per_iter)

        # Simulate each period
        for ttsim in llim:ulim

            # Draw innovations
            innovations[1:nexogshocks] = xrandn[1:nexogshocks,ttsim-1]

            #Update endogenous variables given current values and innovations
            endogvar[:,ttsim] = decrlin(endogvar[:,ttsim-1], innovations, nendogvars, nexogvars, sigma, pp, steady_states)

            # Determine if zlb was hit
            if (endogvar[rm_t,ttsim] < 0.0)
                countzlb = countzlb + 1

                fill!(shockindex,1)

                # Finds position of each shock in exogenous shock grid
                for i in 1:nexogshocks
                    # If lower than lowest grid value assign lowest possible index
                    if (endogvar[nendogvars+i,ttsim] < shockbounds[i,1])
                        shockindex[i] = 1

                    # Otherwise assign it the index whose corresponding value is closest (rounding down)
                    else
                        shockindex[i]  = min(nshockgrid[i], floor(1.0+(endogvar[nendogvars+i,ttsim]-shockbounds[i,1])/shockdistance[i]))
                    end
                end

                # Indexes state that hit zlb
                stateindex = exogposition(shockindex,nshockgrid,nexogvars)

                # If that state hits zlb enough times then consider it a zlb state
                countzlbstates[stateindex] = countzlbstates[stateindex] + 1
                if (countzlbstates[stateindex] > 5)
                    statezlbinfo[stateindex] = 1
                end
            end
        end

        # Loop to check for explosive path
        for ttsim in llim:ulim

            # Path was non-explosive so long as it was within bounds
            non_explosive = ( all(msvlow.<endogvar[1:nmsv,ttsim].<msvhigh) )
            explosiveerror = ( (non_explosive == false) | (isnan(endogvar[1,ttsim]) == true) )
            if (explosiveerror == true)
                counter = max(ttsim-200,0) #Will be 0 if exploded in first half
                println("solution exploded at ", ttsim, " vs ulim ", ulim)
                count_exploded = count_exploded + 1

                if (counter == 0)
                    convergence = false
                    println("degenerated back to the beginning")
                end
                break
            end
        end

        if (count_exploded == 50)
            println("linear solution exploded for the 50th time")
            convergence = false
        end

        if (convergence  == false)
            zlbfrequency = 100.0*countzlb/total_periods
            scalebd = 3.0
            endog_emean = vec(sum(endogvar[:,2:(total_periods+1)], dims=2))/total_periods
            return endog_emean,zlbfrequency,msvbounds,statezlbinfo,convergence
        end

        if (explosiveerror == false)
            counter = counter + periods_per_iter
        end
    end

    # Calculates observed means and standard deviations
    zlbfrequency = 100.0*countzlb/total_periods
    scalebd = 3.0
    endog_emean = vec(sum(endogvar[:,2:total_periods+1],dims=2))/total_periods

    # Calculate std for endogenous var
    for i in 1:nmsv
        msv_std[i] = sqrt(sum((endogvar[i,2:total_periods+1].-endog_emean[i]).^2 )/(total_periods-1))
    end

    # We consider the upper and lower bound of the high probability region to be +- three standard deviations
    msvbounds[1:nmsv] = endog_emean[1:nmsv]-scalebd*msv_std
    msvbounds[nmsv+1:nmsv+nmsv] = endog_emean[1:nmsv]+scalebd*msv_std

    return endog_emean,zlbfrequency,msvbounds,statezlbinfo,convergence

end

"""
# Arguments:
- `nendogvars::Int`: Number of endogenous variables.
- `nexogvars::Int`: Number of exogenous variables (shocks and fixed values).
- `nexogshocks::Int`: Number of active exogenous shocks (with strictly greater than one possible value)
- `nmsv::Int`: Number of minimum state endogenous variables
- `ns::Int`: Number of exogenous states (equal to number of possible distinct combinations of shock values)
- `ngridpoints::Int`: Number of grid points on the Smolyak grid
- `exoggrid::Array{Float64, 2}`: Grid containing the values of each shock at each exogenous state
- `steady_states::Array{Float64,1}`: Steady state values of model
- `endogenous_states::OrderedDict{Symbol,Int64}`: Maps human-readable endogenous states to indice
- `xgrid::Array{Float64, 2}`: Value of each minimum state variable (transformed to [-1, 1] domain) at each Smolyak grid point
- `nfunc::Int`: Number of functions to approximate
- `bbtinv::Array{Float64, 2}`: Inverse of the transposed matrix of Smolyak basis functions evaluated at each Smolyak grid point
- `pp::Array{Float64,2}`: Feedback (endogenous variable lags) part of the linear transition equation.
- `sigma::Array{Float64,2}`: Innovation part of the linear transition equation.

Description:
Computes an initial guess for the solution of the non-linear model using the solution to the linearized version and returns the α coefficients that approximate those functions.
"""
function initial_α(nendogvars::Int, nexogvars::Int, nexogshocks::Int, nmsv::Int, ns::Int, ngridpoints::Int, exoggrid::Array{Float64, 2}, steady_states::Array{Float64,1}, endogenous_states::OrderedCollections.OrderedDict{Symbol,Int64}, slopeconxx::Array{Float64, 1}, xgrid::Array{Float64, 2}, nfunc::Int, bbtinv::Array{Float64,2}, aalin::Array{Float64, 2}, bblin::Array{Float64, 2})

    #Initialize variables
    endogvar = Array{Float64}(undef,nendogvars)
    exogpart = Array{Float64}(undef,nendogvars)
    slopeconxxmsv = Array{Float64}(undef,2*nmsv)
    initialalphas = zeros(nfunc*ngridpoints,2*ns)
    alphass = zeros(nfunc,ngridpoints)
    endogvarm1 = zeros(nendogvars,ngridpoints)
    endogsteady = steady_states[1:nendogvars + nexogvars]

    # Get conversion from xx to msv
    slopeconxxmsv[1:nmsv] = slopeconxx[1:nmsv]
    slopeconxxmsv[nmsv+1:2*nmsv] = slopeconxx[nmsv+1:nmsv+nmsv]

    # Converts endog var back to msv domain and calculates deviation from steady state
    @inbounds @simd for i in 1:ngridpoints
        endogvarm1[1:nmsv,i] = msv2xx(xgrid[1:nmsv,i],nmsv,slopeconxxmsv)-endogsteady[1:nmsv]
    end

    yy = zeros(nfunc,ngridpoints)

    # For each shock grid point
    @fastmath @inbounds @simd for ss in 1:ns
        exogval = zeros(nexogvars)

        # Get shock values
        exogval[1:nexogshocks] = exoggrid[1:nexogshocks,ss]

        # Get linear solution
        @fastmath @inbounds @simd for i in 1:ngridpoints
            mul!(endogvar, aalin, endogvarm1[:,i])
            mul!(exogpart, bblin, exogval)
            endogvar = endogsteady[1:nendogvars] + endogvar + exogpart
            yy[:,i] = endogvar[[endogenous_states[:λc],endogenous_states[:qk_t],endogenous_states[:Vp_t],endogenous_states[:Vw_t],endogenous_states[:bc_t],endogenous_states[:bi_t],endogenous_states[:u_t]] ]
        end

        # Get alphas by multiplying by inverse of Smoylak basis functions (since we want alphas to solve yy = bbt*alphass)
        mul!(alphass,yy,bbtinv)
        @fastmath @inbounds @simd for i in 1:ngridpoints
            @fastmath @inbounds @simd for ifunc in 1:nfunc
                #If (alphass(ifunc,i) < 1.0e-8) alphass(ifunc,i) = 0.0d0
                initialalphas[(ifunc-1)*ngridpoints+i,ss] = alphass[ifunc,i]
                #Initial guess for ZLB polynomials is same as non-ZLB
                initialalphas[(ifunc-1)*ngridpoints+i,ns+ss] = alphass[ifunc,i]
            end
        end
    end

    return initialalphas

end


"""
    fixedpoint(rkss::Float64, approx::Approximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_initial::Array{Float64,2}, zlbswitch::Bool)

# Arguments:
- `rkss::Float64`: Steady State value for rental rate of capital
- `approx::Approximation`: Object containing various details for function and integral approximation
- `params::Array{AbstractParameter{Float64},1}`: Model parameters
- `keys::OrderedDict{Symbol,Int64}`: Maps human-readable names to indices for parameters
- `labss::Float64`: Steady state labor
- `exogenous_shocks::OrderedDict{Symbol,Int64}`: Maps human-readable exogenous shocks to indices
- `endogenous_states::OrderedDict{Symbol,Int64}`: Maps human-readable endogenous states to indices
- `α_initial::Array{Float64,2}`: Initial guess for function approximation coefficients, α
- `zlbswitch::Bool`: Determines whether the model treats the zero lower bounds as a binding constraint or not

# Description:
Uses a fixed point convergence algorithm to determine the functions that solve the model and their associated α coefficients. In particular, we put the equations defining the model solution into the form f = g(f), where f is a vector of functions and g is a vector-valued function. We then iterate on f until we reach such a fixed point. Note also that here we use an approximation for f rather than the true f, which is why the α coefficients are needed. See section 2 of technical appendix of Gust et. al (2017).
"""
function fixedpoint(rkss::Float64, approx::Approximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_initial::Array{Float64,2}, zlbswitch::Bool)

    # Initialize
    α_star = copy(α_initial)
    α_new = Array{Float64}(undef, approx.nfunc*approx.ngridpoints, 2*approx.ns)
    α_temp = Array{Float64}(undef, approx.ngridpoints, 2*approx.nfunc)
    updated_approx_functions = Array{Float64}(undef, 2*approx.nfunc, approx.ngridpoints)
    convergence = false
    avg_error = 0.0

    # Settings for fixed point algorithm
    niter = 150
    tolfun = 1.0e-04
    step  = 7.0e-01

    # Get fixed point using iterative convergence method
    # Loop until convergence (avg_error < tolfun) or niter reached
    for i in 1:niter
        avg_error = 0.0

        # Calculate g(f) to get new guess for f and then calculate new approximation
        # Note that we can do this separately for each exogenous state (which corresponds to a grid point on the exogenous shock grid)
        for j in 1:approx.ns
            err = 0.0
            for k in 1:approx.ngridpoints
                updated_approx_fucntions[:, k], err2 = decr_euler(rkss, approx, k, j, params, keys, α_star, labss, exogenous_shocks, endogenous_states, m[:zero_lower_bound])
                err += err2
            end

            # Solve for α by multiplying by inverse matrix of Smolyak basis polynomials
            mul!(α_temp, approx.bbtinv', updated_approx_polynomials')

            # Transform the matrix of α coefficients associated with this exogenous state to a vector and store in the matrix of new α coefficients
            α_new[:, j] = vec(α_temp[:, 1:approx.nfunc])
            α_new[:, j + approx.ns] = vec(α_temp[:, approx.nfunc+1:2*approx.nfunc])

            avg_error += err
        end

        # Normalize summed error to get average
        avg_error /= 2*approx.ngridpoints*approx.ns

        # Convergence fails if new α has non-numerical elements
        if (any([isnan(a) for a in α_new]) == true)
            convergence = false
            return α_star, convergence
        end

        # Converge when error gets small enough in a given iteration
        if (avg_error < tolfun)
            convergence = true
            return α_star, convergence
        end

        # Updated α are convex combination of old and new (dampening step to help fixed point algorithm converge)
        α_star = (1.0 - step)*α_star + step*α_new
    end

    return α_star, convergence
end

"""
    fixedpoint(rkss::Float64, approx::Approximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_initial::Array{Float64,2}, zlbswitch::Bool)

- `rkss::Float64`: Steady State value for rental rate of capital
- `approx::Approximation`: Object containing various details for function and integral approximation
- `params::Array{AbstractParameter{Float64},1}`: Model parameters
- `keys::OrderedDict{Symbol,Int64}`: Maps human-readable names to indices for parameters
- `labss::Float64`: Steady state labor
- `exogenous_shocks::OrderedDict{Symbol,Int64}`: Maps human-readable exogenous shocks to indices
- `endogenous_states::OrderedDict{Symbol,Int64}`: Maps human-readable endogenous states to indices
- `α_initial::Array{Float64,2}`: Initial guess for function approximation coefficients, α
- `zlbswitch::Bool`: Determines whether the model treats the zero lower bounds as a binding constraint or not

# Description:
Uses a parallel version of the fixed point convergence algorithm to determine the functions that solve the model and their associated α coefficients. In particular, we put the equations defining the model solution into the form f = g(f), where f is a vector of functions and g is a vector-valued function. We then iterate on f until we reach such a fixed point. Note also that here we use an approximation for f rather than the true f, which is why the α coefficients are needed. See section 2 of technical appendix of Gust et. al (2017).
"""
function fixedpoint_parallel(rkss::Float64, approx::Approximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_initial::Array{Float64,2}, zlbswitch::Bool)


    # Initialize
    α_star = copy(α_initial)
    α_new = Array{Float64}(undef, approx.nfunc*approx.ngridpoints, 2*approx.ns)
    α_temp = Array{Float64}(undef, approx.ngridpoints, 2*approx.nfunc)
    updated_approx_functions = Array{Float64}(undef, 2*approx.nfunc, approx.ngridpoints)
    convergence = false
    avg_error = 0.0

    # Settings for fixed point algorithm
    niter = 150
    tolfun = 1.0e-04
    step  = 7.0e-01

    # Get fixed point using iterative convergence method
    # Loop until convergence (avg_error < tolfun) or niter reached
    for i in 1:niter
        avg_error = 0.0

        # Calculates new α_new and avg_error
        # Note that are doing this in parallel for each exogenous state (which corresponds to a grid point on the exogenous shock grid)
        α_here = @sync @distributed (hcat) for j in 1:approx.ns
            parallel_help(rkss, approx, params, keys, labss, exogenous_shocks, endogenous_states,α_star,j, zlbswitch)
        end

        α_new[:,1:approx.ns] = α_here[:,1:3:end]
        α_new[:,approx.ns+1:end] = α_here[:,2:3:end]
        avg_error = sum(α_here[1,3:3:end])

        # Normalize summed error to get average
        avg_error /= 2*approx.ngridpoints*approx.ns

        # Convergence fails if new α has non-numerical elements
        if (any([isnan(a) for a in α_new]) == true)
            convergence = false
            return α_star, convergence
        end

        # Converge when error gets small enough in a given iteration
        if (avg_error < tolfun)
            convergence = true
            return α_star, convergence
        end

        # Updated α are convex combination of old and new (dampening step to help fixed point algorithm converge)
        α_star = (1.0 - step)*α_star + step*α_new
    end

    return α_star, convergence
end

"""
    parallel_help(rkss::Float64, approx::Approximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_star::Array{Float64,2}, shockpos::Int, zlbswitch::Bool)

- `rkss::Float64`: Steady State value for rental rate of capital
- `approx::Approximation`: Object containing various details for function and integral approximation
- `params::Array{AbstractParameter{Float64},1}`: Model parameters
- `keys::OrderedDict{Symbol,Int64}`: Maps human-readable names to indices for parameters
- `labss::Float64`: Steady state labor
- `exogenous_shocks::OrderedDict{Symbol,Int64}`: Maps human-readable exogenous shocks to indices
- `endogenous_states::OrderedDict{Symbol,Int64}`: Maps human-readable endogenous states to indices
- `α_star::Array{Float64,2}`: Current function approximation coefficients
- `shockpos::Int`: Exogenous state to calculate g(f) at
- `zlbswitch::Bool`: Determines whether the model treats the zero lower bounds as a binding constraint or not

# Description:
Helper function for parallel version of fixed point that calculates g(f) at a given exogenous state.
"""
function parallel_help(rkss::Float64, approx::Approximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_star::Array{Float64,2},shockpos::Int, zlbswitch::Bool)

    # Initialize variables
    parallel_info = zeros(approx.nfunc*approx.ngridpoints,3)
    updated_approx_functions = zeros(2*approx.nfunc, approx.ngridpoints)
    err = 0.0

    # Calculate g(f) to get new guess for f at given exogenous state and then calculate new approximation
    @inbounds @simd for k in 1:approx.ngridpoints
        updated_approx_polynomials[:, k], err2 = decr_euler(rkss, approx, k, shockpos, params, keys, α_star, labss, exogenous_shocks, endogenous_states, m[:zero_lower_bound])
        err += err2
    end

    # Solve for α by multiplying by inverse matrix of Smolyak basis polynomials
    mul!(α_temp, approx.bbtinv', updated_approx_polynomials')

    # Transform the matrix of α coefficients associated with this exogenous state to a vector and store in the matrix of new α coefficient
    parallel_info[:, 1] = vec(α_temp[:, 1:approx.nfunc])
    parallel_info[:, 2] = vec(α_temp[:, approx.nfunc+1:2*approx.nfunc])
    parallel_info[1,3] = err

    return parallel_info
end

"""
```
GensysError <: Exception
```
A `GensysError` is thrown when:

1. Gensys does not give existence and uniqueness, or
2. A LAPACK error was thrown while computing the Schur decomposition of Γ0 and Γ1

If a `GensysError`is thrown during Metropolis-Hastings, it is caught by
`posterior`.  `posterior` then returns a value of `-Inf`, which
Metropolis-Hastings always rejects.

### Fields

* `msg::String`: Info message. Default = \"Error in Gensys\"
"""
mutable struct GensysError <: Exception
    msg::String
end
GensysError() = GensysError("Error in Gensys")
Base.showerror(io::IO, ex::GensysError) = print(io, ex.msg)
