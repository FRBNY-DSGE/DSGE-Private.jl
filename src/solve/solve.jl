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
solve(m::AbstractModel; parallel = false)
```

Driver to compute the model solution and corresponding coefficients matrix

### Inputs

- `m`: the model object

## Keyword Arguments

- `parallel::Bool`: whether or not to solve the model in parallel.

### Output
    α⋆, the matrix of parameters (see a in Equation 2.21 in Technical Appendix of GHLS (2017))
"""
function solve(m::GHLS, parallel::Bool=false)

    # Find grid points of each combination of shock values, and lower and upper bounds of each shock, and the distance between shock values for any two grid points.
    m.approx.exoggrid, m.approx.shockbounds, m.approx.shockdistance = get_shockdetails(m.approx.nexogcont, m.approx.number_shock_values, m.approx.nshockgrid, m.approx.exogvarinfo, m.approx.nexogshock, m.approx.ns, m.approx.nexog,m.parameters,m.keys)

    # Get Equilibrium Conditions and run gensys
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
    pp = TTT[1:28, 1:28]
    sigma = RRR[1:28, 1:6]

    # Get steady state values
    steady_states = [i.value for i in m.steady_state]

    # Compute ergodic means of the endogenous variables (endog_emean)
    m.approx.endog_emean, m.approx.zlbfrequency, m.approx.msvbounds, m.approx.statezlbinfo, m.approx.convergence = simulate_linear(m.approx.ns, m.approx.nvars, m.approx.nexog, m.approx.nmsv, m.approx.nexogcont, m.approx.nexogshock, steady_states, m.approx.nshockgrid, m.approx.shockbounds, m.approx.shockdistance, pp, sigma)

    # Find the slopes coefficients and constants to go from msv space to [-1,1] domain and vice-versa.
    m.approx.slopeconmsv, m.approx.slopeconxx = create_slopes(m.approx.nmsv, m.approx.nexogcont, m.approx.msvbounds)

    # Transform linear solution into representation using shocks directly rather than innovations:
    #     e_t = A*e_{t-1} + B*ν_t where A is aalin, B is bblin, e_t is endogenous states and ν_t are the shocks.
    # Used to get initial guess for nonlinear solution when fed into initial_α
    aalin, bblin = lindecrule_markov(pp, sigma, m.approx.nvars, m.approx.nexog, m.approx.nexogcont, m.approx.nexogshock)

    # Construct starting guess for α matrix
    α_initial = initial_α(m.approx.nvars, m.approx.nexog, m.approx.nexogshock, m.approx.nmsv, m.approx.nexogcont, m.approx.ns, m.approx.ngrid, m.approx.exoggrid, steady_states, m.approx.slopeconxx, m.approx.xgrid, m.approx.nfunc, m.approx.bbtinv, aalin, bblin)

    # Runs the fixedpoint convergence algorithm to find α⋆
    α_star, convergence = if parallel
        fixedpoint_parallel(m[:rkss].value, m.approx, m.parameters, m.keys, m[:labss].value, m.exogenous_shocks, m.endogenous_states, α_initial)
    else
        fixedpoint(m[:rkss].value, m.approx, m.parameters, m.keys, m[:labss].value, m.exogenous_shocks, m.endogenous_states, α_initial)
    end

    return α_star
end

"""
    create_slopes(nmsv::Int, nexogcont::Int, msvbounds::Array{Float64, 1})

Find the slopes coefficients and constants to go from msv space to [-1,1] domain and vice-versa.

## Inputs
`nmsv`::Int - Number of minimum state endogenous variables
`nexogcont`::Int - Shocks on smooth part of approximated nonlinear decision rule.
`msvbounds`::Array{Float64,1} - Bounds on minimum state variables (first nmsv are lower, rest are uper).
"""
function create_slopes(nmsv::Int, nexogcont::Int, msvbounds::Array{Float64, 1})

    #Total number of state variables - these values should come from model
    nmsvplus = nmsv + nexogcont

    # Conversion from state domain (msv) to [-1, 1] domain (xx)
    slopeconmsv = zeros(2*nmsvplus)
    slopeconmsv[1:nmsvplus] = 2.0 ./  (msvbounds[nmsvplus + 1 :  2*nmsvplus] - msvbounds[1:nmsvplus])
    slopeconmsv[nmsvplus+1:2*nmsvplus] = -2.0 * msvbounds[1:nmsvplus] ./ (msvbounds[nmsvplus + 1 : 2*nmsvplus] .- msvbounds[1:nmsvplus]) .- 1.

    # Conversion from xx to msv domains
    slopeconxx = zeros(2*nmsvplus)
    slopeconxx[1:nmsvplus] =  0.5 * (msvbounds[nmsvplus + 1 : 2*nmsvplus] - msvbounds[1:nmsvplus])
    slopeconxx[nmsvplus+1:2*nmsvplus] = msvbounds[1:nmsvplus] + 0.5 * (msvbounds[nmsvplus + 1 : 2*nmsvplus] - msvbounds[1:nmsvplus])

    return slopeconmsv, slopeconxx
end

"""
    lindecrule_markov(pp::Array{Float64, 2}, sigma::Array{Float64, 2}, nvars::Int, nexog::Int, nexogcont::Int, nexogshock::In

Transform linear solution into representation that can be used to simulate shocks rather than innovations directly.
Used to get initial guess for nonlinear solution.
The transformed system has the form:
    $y_t = A y_{t-1} + B s_{t}$,
where y_t are the endogenous variables excluding the shocks and s_t are the shocks.
...
# Arguments
- `pp`::Array{Float64,2} - Part of the linear decision rule (feedback part).
- `sigma`::Array{Float64,2} - Part of the linear decision rule (innovation part).
- `nvars`::Int - Number of endogenous variables.
- `nexog`::Int - Number of exogenous variables (shocks and fixed values).
- `nexogshock`::Int - Number of exogenous shocks
...
"""
function lindecrule_markov(pp::Array{Float64, 2}, sigma::Array{Float64, 2}, nvars::Int, nexog::Int, nexogcont::Int, nexogshock::Int)

    #Initilize Variables
    bblin=zeros(nvars,nexog)

    @fastmath @inbounds @simd for i in 1:nexogshock
        bblin[:,i] = sigma[1:nvars,i]/sigma[nvars+i,i]
    end

    @fastmath @inbounds @simd for i in 1:nexogcont
        bblin[:,nexog-i+1] = sigma[1:nvars,nexog-i+1]/sigma[nvars+nexog-i+1,nexog-i+1]
    end

    return pp[1:nvars,1:nvars],bblin

end

"""
    fixedpoint(rkss::Float64, approx::SmolyakApproximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_initial::Array{Float64,2})

Runs fixedpoint convergence to solve for α⋆
...
# Arguments
- `rkss::Float64`: Steady State value for rental rate of capital
- `approx::SmolyakApproximation`: Object containing various details for Smolyak approximation
- `params::Array{AbstractParameter{Float64},1}`: Model parameters (obtianed with m.parameters if m is a model of type GHLS)
- `keys::OrderedDict{Symbol,Int64}`: Keys for the model parameters (obtained with m.keys)
- `labss::Float64`: Steady State Total Labor
- `exogenous_shocks::OrderedDict{Symbol,Int64}`: Exogenous shocks in model (m.exogenous_shocks)
- `endogenous_states::OrderedDict{Symbol,Int64}`: Endogenous states in model (m.endogenous_states)
- `α_initial::Array{Float64,2}`: Initial guess for polynomial coefficients, α (from initial_α)
...
"""
function fixedpoint(rkss::Float64, approx::SmolyakApproximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_initial::Array{Float64,2})

    # Initialize
    α_star = copy(α_initial)
    α_new = Array{Float64}(undef, approx.nfunc*approx.ngrid, 2*approx.ns)
    #α_temp = Array{Float64}(undef, 2*nfunc, ngrid)
    α_temp = Array{Float64}(undef, approx.ngrid, 2*approx.nfunc)
    updated_approx_polynomials = Array{Float64}(undef, 2*approx.nfunc, approx.ngrid)
    convergence = false
    avg_error = 0.0

    # Settings - should probably be stored in model
    niter = 150
    tolfun = 1.0e-04
    step  = 7.0e-01

    # Get fixed point using iterative convergence method
    for i in 1:niter
        @show i
        avg_error = 0.0

        for j in 1:approx.ns
            err = 0.0

            # Update polynomials using new guess for α
            for k in 1:approx.ngrid
                updated_approx_polynomials[:, k], err2 = decr_euler(rkss, approx, k, j, params, keys, α_star, labss, exogenous_shocks, endogenous_states)
                err += err2
            end

            # Solve for α by multiplying by inverse matrix (we take transpose since julia is column major so this will be faster to reindex)
            mul!(α_temp, approx.bbtinv', updated_approx_polynomials')

            #Reindex
            α_new[:, j] = vec(α_temp[:, 1:approx.nfunc])
            α_new[:, j + approx.ns] = vec(α_temp[:, approx.nfunc+1:2*approx.nfunc])

            avg_error += err
        end

        # Normalize summed error to get average
        avg_error /= 2*approx.ngrid*approx.ns

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
        # Updated α are convex combination of old and new
        α_star = (1.0 - step)*α_star + step*α_new
    end
    return α_star, convergence
end

"""
    simulate_linear(ns::Int, nvars::Int, nexog::Int, nmsv::Int, nexogcont::Int, nexogshock::Int, steady_states::Array{Float64, 1}, nshockgrid:: Array{Int, 1}, shockbounds::Array{Float64,2}, shockdistance::Array{Float64,1}, pp::Array{Float64,2}, sigma::Array{Float64, 2})

Simulate data to compute ergodic means of economy's variables.

## Inputs
- `ns`::Int - Total number of grid points (Possible combinations of shock values at grid points)
- `nvars`::Int - Number of endogenous variables.
- `nexog`::Int - Number of exogenous variables (shocks and fixed values).
- `nmsv`::Int - Number of minimum state endogenous variables
- `nexogcont`::Int - Shocks on smooth part of approximated nonlinear decision rule.
- `nexogshock`::Int - Number of exogenous shocks
- `steady_states`::Vector{Float64} - The model's steady state values
- `nshockgrid`::Vector{Int} - Vector containing grid size for each shock.
- `shockbounds`::Array{Float64,2} - Lower and upper bounds on each exogenous shock.
- `shockdistance`::Array{Float64,1} - Distance between grid points for each shock.
- `pp`::Array{Float64,2} - Part of the linear decision rule (feedback part).
- `sigma`::Array{Float64,2} - Part of the linear decision rule (innovation part).
"""
function simulate_linear(ns::Int, nvars::Int, nexog::Int, nmsv::Int, nexogcont::Int, nexogshock::Int, steady_states::Array{Float64, 1}, nshockgrid:: Array{Int, 1}, shockbounds::Array{Float64,2}, shockdistance::Array{Float64,1}, pp::Array{Float64,2}, sigma::Array{Float64, 2})
    # Initialize variables - some of these should be in model settings
    total_periods = 100000
    periods_per_iter = 400
    countzlb = 0
    count_exploded = 0
    counter = 1
    convergence = true

    statezlbinfo = zeros(Int64, ns)
    endog_emean = Array{Float64}(undef, nvars + nexog)
    msvbounds = Array{Float64}(undef, 2*(nmsv + nexogcont))
    shockindex = Array{Int64}(undef, nexog - nexogcont)
    countzlbstates = zeros(Int64, ns)
    msvhigh = Array{Float64}(undef, nmsv)
    msvlow = Array{Float64}(undef, nmsv)
    innovations = Array{Float64}(undef, nexog)
    xrandn = Array{Float64}(undef, nexog, total_periods)
    msv_std = zeros(nmsv + nexogcont)
    endogvar = zeros(nvars+nexog, total_periods+1)

    # Set up random generator
    path = dirname(@__FILE__)
    iseed = MersenneTwister(101294)
    randn!(iseed, xrandn)
    xrandnFortran=readdlm("$path/xrandn.txt") #For testing
    xrandn=reshape(xrandnFortran,nexog,total_periods)

    # Set up for first period
    endogvar[1:nmsv,1]= steady_states[1:nmsv]
    msvhigh = log(2.0) + endogvar[1:nmsv, 1]
    msvlow = log(0.01) + endogvar[1:nmsv, 1]

    #zero out shocks not included in nonlinear model
    if (nexogshock + nexogcont < nexog)
        sigma[:,nexogshock+1:nexog-nexogcont] .= 0.0
    end
    stateindex = 0

    while counter <= total_periods + 1
        explosiveerror = false
        llim = counter+1
        ulim = min(total_periods+1,counter+periods_per_iter)

        # Simulate each period
        for ttsim in llim:ulim

            # Draw innovations
            innovations[1:nexogshock] = xrandn[1:nexogshock,ttsim-1]

            if (nexogcont > 0)
                innovations[nexog-nexogcont+1:nexog] = xrandn[nexog-nexogcont+1:nexog,ttsim-1]
            end

            #THIS SHOULD BE DONE DIFFERENTLY HERE
            endogvar[:,ttsim] = decrlin(endogvar[:,ttsim-1], innovations, nvars, nexog, sigma, pp, steady_states)

            # Account for ZLB, why is this not 1 though?
            if (endogvar[5,ttsim] < 0.0)
                countzlb = countzlb + 1

                fill!(shockindex,1)

                # Assigns each shock to position in grid
                for i in 1:nexogshock
                    # If lower than lowest grid value assign lowest possible index
                    if (endogvar[nvars+i,ttsim] < shockbounds[i,1])
                        shockindex[i] = 1

                    # Otherwise assign it the index whose corresponding value is closest (rounding down)
                    else
                        shockindex[i]  = min(nshockgrid[i], floor(1.0+(endogvar[nvars+i,ttsim]-shockbounds[i,1])/shockdistance[i]))
                    end
                end

                # Indexes state that gave zlb
                stateindex = exogposition(shockindex,nshockgrid,nexog-nexogcont)

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

    #calculate mean of all variables; std. dev of minimum state variables and exogenous variables
    #that we include in polynomial part of approximated decision rule
    zlbfrequency = 100.0*countzlb/total_periods
    scalebd = 3.0
    endog_emean = vec(sum(endogvar[:,2:total_periods+1],dims=2))/total_periods

    # Calculate std for endogenous var
    for i in 1:nmsv
        msv_std[i] = sqrt(sum((endogvar[i,2:total_periods+1].-endog_emean[i]).^2 )/(total_periods-1))
    end

    msvbounds[1:nmsv] = endog_emean[1:nmsv]-scalebd*msv_std

    nmsvplus = nmsv + nexogcont # Added this line
    msvbounds[nmsvplus+1:nmsvplus+nmsv] = endog_emean[1:nmsv]+scalebd*msv_std

    # Calculate std for shocks
    for i in 1:nexogcont
        msv_std[nmsv+i] = sqrt(sum((endogvar[nvars+nexog-i+1,2:total_periods+1] -endog_emean[nvars+nexog-i+1]).^2 )/(total_periods-1))
        msvbounds[nmsv+i] = endog_emean[nvars+nexog-i+1]-scalebd*msv_std[nmsv+i]
        msvbounds[nmsvplus+nmsv+i] = endog_emean[nvars+nexog-i+1]+scalebd*msv_std[nmsv+i]
    end

    return endog_emean,zlbfrequency,msvbounds,statezlbinfo,convergence

end

"""

Takes steady states, Smolyak grid, and aalin and bblin from lindecrule_markov and calculates an initial guess for α.
...
# Arguments
- `approx::SmolyakApproximation`: Object containing various details for Smolyak approximation
- `steady_states::Array{Float64,1}`: Values for steady states from model object

A and B below refer to this equation: e_t = A*e_{t-1} + B*ν_t where A is aalin, B is bblin, e_t is endogenous states and ν_t are the shocks.
- `aalin::Array{Float64, 2}`: After lindecrule_markov, this is A
- `bblin::Array{Float64,2}`: After lindecrule_markov, this is B
...
"""
function initial_α(nvars::Int, nexog::Int, nexogshock::Int, nmsv::Int, nexogcont::Int, ns::Int, ngrid::Int, exoggrid::Array{Float64, 2}, steady_states::Array{Float64,1}, slopeconxx::Array{Float64, 1}, xgrid::Array{Float64, 2}, nfunc::Int, bbtinv::Array{Float64,2}, aalin::Array{Float64, 2}, bblin::Array{Float64, 2})

    #Initilize variables
    endogvar = Array{Float64}(undef,nvars)
    exogpart = Array{Float64}(undef,nvars)
    slopeconxxmsv = Array{Float64}(undef,2*nmsv)
    slopeconcont = Array{Float64}(undef,2*nexogcont)
    initialalphas = zeros(nfunc*ngrid,2*ns)
    alphass = zeros(nfunc,ngrid)
    endogvarm1 = zeros(nvars,ngrid) #holds lags
    nmsvplus = nmsv + nexogcont
    endogsteady = steady_states[1:nvars + nexog]

    # Get conversion from xx to msv
    slopeconxxmsv[1:nmsv] = slopeconxx[1:nmsv]
    slopeconxxmsv[nmsv+1:2*nmsv] = slopeconxx[nmsvplus+1:nmsvplus+nmsv]

    if (nexogcont > 0)
        slopeconcont[1:nexogcont] = slopeconxx[nmsv+1:nmsvplus]
        slopeconcont[nexogcont+1:2*nexogcont] = slopeconxx[nmsvplus+nmsv+1:2*nmsvplus]
    end

    # Converts endog var back to msv domain
    @inbounds @simd for i in 1:ngrid
        endogvarm1[1:nmsv,i] = msv2xx(xgrid[1:nmsv,i],nmsv,slopeconxxmsv)-endogsteady[1:nmsv] #CHECK THIS FUNCTION
    end

    yy = zeros(nfunc,ngrid)
    # For each shock grid point
    @fastmath @inbounds @simd for ss in 1:ns
        exogval = zeros(nexog)

        exogval[1:nexogshock] = exoggrid[1:nexogshock,ss]   #update shocks (in deviation from ss)

        @fastmath @inbounds @simd for i in 1:ngrid

            if (nexogcont > 0)
                exogval[nexog-nexogcont+1:nexog] = msv2xx(xgrid[nmsv+1:nmsv+nexogcont,i],nexogcont,slopeconcont)-endogsteady[nvars+nexog-nexogcont+1:nvars+nexog]
            end

            #get linear solution
            mul!(endogvar, aalin, endogvarm1[:,i]) #REMAKE THIS FUNCTION
            mul!(exogpart, bblin, exogval) # REMAKE THIS FUNCTION
            endogvar = endogsteady[1:nvars] + endogvar + exogpart
            yy[:,i] = endogvar[[10,11,18,19,21,22,13] ]
        end

        # Get alphas by inverting approximation function
        mul!(alphass,yy,bbtinv)
        @fastmath @inbounds @simd for i in 1:ngrid
            @fastmath @inbounds @simd for ifunc in 1:nfunc
                #if (alphass(ifunc,i) < 1.0e-8) alphass(ifunc,i) = 0.0d0
                initialalphas[(ifunc-1)*ngrid+i,ss] = alphass[ifunc,i]
                #initial guess for ZLB polynomials
                initialalphas[(ifunc-1)*ngrid+i,ns+ss] = alphass[ifunc,i]
            end
        end
    end

    return initialalphas

end

"""
    parallel_help(rkss::Float64, approx::SmolyakApproximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_star::Array{Float64,2},j::Int)

Helper function for the parallel implementation of fixedpoint.
...
# Arguments
- `rkss::Float64`: Steady State value for rental rate of capital
- `approx::SmolyakApproximation`: Object containing various details for Smolyak approximation
- `params::Array{AbstractParameter{Float64},1}`: Model parameters (obtianed with m.parameters if m is a model of type GHLS)
- `keys::OrderedDict{Symbol,Int64}`: Keys for the model parameters (obtained with m.keys)
- `labss::Float64`: Steady State Total Labor
- `exogenous_shocks::OrderedDict{Symbol,Int64}`: Exogenous shocks in model (m.exogenous_shocks)
- `endogenous_states::OrderedDict{Symbol,Int64}`: Endogenous states in model (m.endogenous_states)
- `α_initial::Array{Float64,2}`: Initial guess for polynomial coefficients, α (from initial_α)
...
"""
function parallel_help(rkss::Float64, approx::SmolyakApproximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_star::Array{Float64,2},j::Int)
    col1 = zeros(nfunc*ngrid,3)

    updated_approx_polynomials = zeros(2*approx.nfunc, approx.ngrid)
    err = 0.0

    # Update polynomials using new guess for α
    @fastmath @inbounds @simd for k in 1:approx.ngrid
        updated_approx_polynomials[:, k], err2 = decr_euler(rkss, approx, k, j, params, keys, α_star, labss, exogenous_shocks, endogenous_states)
        err += err2
    end

    # Solve for α by multiplying by inverse matrix
    α_temp = BLAS.gemm('T', 'T', approx.bbtinv, updated_approx_polynomials)

    #Reindex
    col1[:, 1] = vec(α_temp[:, 1:approx.nfunc])
    col1[:, 2] = vec(α_temp[:, approx.nfunc+1:2*approx.nfunc])
    col1[1,3] = err

    return col1
end


# Parallel Fixedpoint
"""
    fixedpoint_parallel(rkss::Float64, approx::SmolyakApproximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_initial::Array{Float64,2})

Runs fixedpoint convergence to solve for α⋆ in parallel
...
# Arguments
- `rkss::Float64`: Steady State value for rental rate of capital
- `approx::SmolyakApproximation`: Object containing various details for Smolyak approximation
- `params::Array{AbstractParameter{Float64},1}`: Model parameters (obtianed with m.parameters if m is a model of type GHLS)
- `keys::OrderedDict{Symbol,Int64}`: Keys for the model parameters (obtained with m.keys)
- `labss::Float64`: Steady State Total Labor
- `exogenous_shocks::OrderedDict{Symbol,Int64}`: Exogenous shocks in model (m.exogenous_shocks)
- `endogenous_states::OrderedDict{Symbol,Int64}`: Endogenous states in model (m.endogenous_states)
- `α_initial::Array{Float64,2}`: Initial guess for polynomial coefficients, α (from initial_α)
...
"""
function fixedpoint_parallel(rkss::Float64, approx::SmolyakApproximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, α_initial::Array{Float64,2})

    # Initialize
    α_star = copy(α_initial)
    α_new = zeros(Float64, approx.nfunc*approx.ngrid, 2*approx.ns)
    α_temp = zeros(Float64, 2*approx.nfunc, approx.ngrid)
    convergence = false

    # Settings - should probably be stored in model
    niter = 150
    tolfun = 1.0e-04
    step  = 7.0e-01

    # Get fixed point using iterative convergence method
    for i in 1:niter
        @show i
        avg_error = 0.0

        # Calculates new α_new and avg_error
        α_here = @sync @distributed (hcat) for j in 1:approx.ns #mystart:myend
            parallel_help(rkss, approx, params, keys, labss, exogenous_shocks, endogenous_states,α_star,j)
        end

        α_new[:,1:approx.ns] = α_here[:,1:3:end]
        α_new[:,approx.ns+1:end] = α_here[:,2:3:end]
        avg_error = sum(α_here[1,3:3:end])

        # Normalize summed error to get average
        avg_error /= 2*approx.ngrid*approx.ns

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

        # Updated α are convex combination of old and new
        α_star = (1.0 - step)*α_star + step*α_new
    end
    return α_star, convergence
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
