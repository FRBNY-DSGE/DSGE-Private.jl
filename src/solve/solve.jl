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

function solve(m::GHLS)

    # Get canonical matrices of linearized solution
    Γ0, Γ1, C, Ψ, Π  = eqcond(m)

    #SIMULATE LINEAR HERE
    endog_emean, zlbfrequency, msvbounds, statezlbinfo, convergence = simulate_linear(m, m.approx)
    m.approx <= Setting(:endog_emean, endog_emean)
    m.approx <= Setting(:zlbfrequency, zlbfrequency)
    m.approx <= Setting(:msvbounds, msvbounds)
    m.approx <= Setting(:statezlbinfo, statezlbinfo)
    m.approx <= Setting(:convergence, convergence)

    #Total number of state variables - these values should come from model
    nmsvplus = m.approx[:nmsv] + m.approx[:nexogcont]

    # Conversion from state domain (msv) to [-1, 1] domain (xx)
    slopeconmsv = zeros(2*nmsvplus)
    slopeconmsv[1:nmsvplus] = 2.0 ./  (msvbounds[nmsvplus + 1 :  2*nmsvpluss] - msvbounds[1:nmsvplus])
    slopeconmsv[nmsvplus+1:2*nmsvplus] = -2.0 * msvbounds[1:nmsvplus] ./ (msvbounds[nmsvplus + 1 : 2*nmsvplus] .- msvbounds[1:nmsvplus]) .- 1.
    m.approx <= Setting(:slopeconmsv, slopeconmsv)

    # Conversion from xx to msv domains
    slopeconxx = zeros(2*nmsvplus)
    slopeconxx[1:nmsvplus] =  0.5 * (msvbounds[nmsvplus + 1 : 2*nmsvplus] - msvbounds[1:nmsvplus])
    slopeconxx[nmsvplus+1:2*nmsvplus] = msvbounds[1:nmsvplus] + 0.5 * (msvbounds[nmsvplus + 1 : 2*nmsvplus] - msvbounds[1:nmsvplus])

    m.approx <= Setting(:slopeconxx, slopeconxx)

    # Construct starting guess
    aalin, bblin = lindecrule_markov(Γ0, Γ1, C, Ψ, Π)
    α_initial = initial_α(m, m.approx, aalin, bblin)

    α_star, convergence = fixedpoint(m, m.approx, α_initial)


end

# Matrix multiplication
function dgemm(α::FLoat64, A::Array{Float64}, B::Array{Float64})
    C = α*A*B
    return C
end

function fixedpoint(m::GHLS; approx::SmolyakApproximation; α_initial :: Array{Float64})

    # Initialize
    α_star = copy(α_initial)
    α_new = Array{Float64}(undef, approx[:nfunc]*approx[:ngrid], 2*approx[:ns])
    α_temp = Array{Float64}(undef, 2*approx[:nfunc], approx[:ngrid])
    convergence = false

    # Settings - should probably be stored in model
    niter = 150
    tolfun = 1.0e-04
    step  = 7.0e-01

    # Get fixed point using iterative convergence method
    for i in 1:niter
        avg_error = 0.0

        for j in 1:approx[:ns]

            updated_approx_polynomials = zeros(2*approx[:nfunc], approx[:ngrid])
            err = 0.0

            # Update polynomials using new guess for α
            for k in 1:approx[:ngrid]
                updated_approx_polynomials[: k], err = decr_euler(k, j, m, approx, α_star)
            end

            # Solve for α by multiplying by inverse matrix and then reindex
            for k in 1:approx[:ngrid]
                for l in 1:approx[:nfunc]
                    α_temp = dgemm(1.0 * updated_approx_polynomials * bbtinv)
                    α_new[(l - 1)*approx[:ngrid]+ k, j] = α_temp[l, k]
                    α_new[(l - 1)*approx[:ngrid]+ k, approx[:ns] + j] = α_temp[approx[:nfunc] + l, k]
                end
            end

            avg_error += err
        end

        # Normalize summed error to get average
        avg_error /= 2*approx[:ngrid]*approx[:ns]

        # Convergence fails if new α has non-numerical elements
        if (any([isnan(a) for a in α_new]) == true)
            convergence = false
            return α_star, convergence
        end

        # Converge when error gets small enough in a given iteration
        if (avgerror < tolfun)
            convergence = true
            return α_star, convergence

        # Updated α are convex combination of old and new
        α_star = (1.0 - step)*α_star + step*α_new
    end

    return α_star, convergence

end

function simulate_linear(m::GHLS, approx::SmolyakApproximation)
    # Initialize variables - some of these should be in model settings
    total_periods = 100000
    periods_per_iter = 400
    countzlb = 0
    count_exploded = 0
    counter = 1
    convergence = true

    statezlbinfo = zeros(Int64, approx[:ns])
    endog_ergodmean = Array{Float64}(undef, approx[:nvars])
    msvbounds = Array{Float64}(undef, 2*(approx[:nmsv] + approx[:nexogcont]))
    shockindex = Array{Int64}(undef, approx[:nexog] - approx[:nexogcont])
    countzlbstates = zeros(Int64, ns)
    msvhigh = Array{Float64}(undef, approx[:nmsv], 1)
    msvlow = Array{Float64}(undef, approx[:nmsv]], 1)
    innovations = Array{Float64}(undef, approx[:nexog], 1)
    xrandn = Array{Float64}(undef, approx[:nexog], total_periods)
    msv_std = zeros(approx[:nmsv] + approx[:nexogcont])
    endogvar = zeros(approx[:nvars], total_periods+1)

    # Set up random generator
    iseed = MersenneTwister(101294)
    randn!(iseed, xrandn)
    xrandnFortran=readdlm("xrandn.txt") #Don't have this file yet
    xrandn=reshape(xrandnFortran,approx[:nexog],total_periods)

    # Set up for first period
    endogvar[1:nmsv,1]= m.steady_state[1:nmsv]
    msvhigh = log.(exp.(m.steady_state[1:nmsv])*2.0)
    msvlow = log.(exp.(m.steady_state[1:nmsv])*0.01)


    while counter <= capt + 1

        explosiveerror = false
        llim = counter+1
        ulim = min(capt+1,counter+displacement)

        # Simulate each period
        for ttsim in llim:ulim

            # Draw innovations
            innovations[1:approx[:nexogshock]] = xrandn[1:approx[:nexogshock],ttsim-1]

            if (approx[:nexogshock] > 0)
                innovations[approx[:nexog]-approx[:nexogcont]+1:approx[:nexog]] = xrandn[approx[:nexog]-approx[:nexogcont]+1:approx[:nexog],ttsim-1]
            end

            #THIS SHOULD BE DONE DIFFERENTLY HERE
            endogvar[:,ttsim] = decrlin(endogvar[:,ttsim-1],innovations,m)

            # Account for ZLB, why is this not 1 though?
            if (endogvar[5,ttsim] < 0.0)
                countzlb = countzlb + 1

                fill!(shockindex,1)

                # Assigns each shock to position in grid
                for i in 1:approx[:nexog]shock
                    # If lower than lowest grid value assign lowest possible index
                    if (endogvar[approx[:nvars]-approx[:nexog]+i,ttsim] < shockbounds[i,1])
                        shockindex[i] = 1

                    # Otherwise assign it the index whose corresponding value is closest (rounding down)
                    else
                        shockindex[i]  = min( nshockgrid[i], floor(1.0+(endogvar[approx[:nvars]-approx[:nexog]+i,ttsim]-shockbounds[i,1])/shockdistance[i]))
                    end
                end

                # Indexes state that gave zlb
                stateindex = exogposition(shockindex,nshockgrid,approx[:nexog]-approx[:nexog]cont)

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
            zlbfrequency = 100.0*countzlb/capt
            scalebd = 3.0
            endog_emean = sum(endogvar[:,2:(capt+1)], dims=2)/capt
            return endog_emean,zlbfrequency,msvbounds,statezlbinfo,convergence
        end

        if (explosiveerror == false)
            counter = counter + displacement
        end
    end

    #calculate mean of all variables; std. dev of minimum state variables and exogenous variables
    #that we include in polynomial part of approximated decision rule
    zlbfrequency = 100.0*countzlb/capt
    scalebd = 3.0
    endog_emean = sum(endogvar[:,2:capt+1],dims=2)/capt

    # Calculate std for endogenous var
    for i in 1:approx[:nmsv]
        msv_std[i] = sqrt(sum((endogvar[i,2:capt+1].-endog_emean[i]).^2 )/(capt-1))
    end

    msvbounds[1:approx[:nmsv]] = endog_emean[1:approx[:nmsv]]-scalebd*msv_std
    msvbounds[approx[:nmsvplus]+1:approx[:nmsvplus]+approx[:nmsv]] = endog_emean[1:approx[:nmsv]]+scalebd*msv_std

    # Calculate std for shocks
    for i in 1:approx[:nexogcont]
        msv_std[approx[:nmsv]+i] = sqrt(sum((endogvar[approx[:nvars]-i+1,2:capt+1] -endog_emean[approx[:nvars]-i+1]).^2 )/(capt-1))
        msvbounds[approx[:nmsv]+i] = endog_emean[approx[:nvars]-i+1]-scalebd*msv_std[approx[:nmsv]+i]
        msvbounds[approx[:nmsvplus]+approx[:nmsv]+counter] = endog_emean[approx[:nvars]-i+1]+scalebd*msv_std[approx[:nmsv]+counter]
    end

    return endog_emean,zlbfrequency,msvbounds,statezlbinfo,convergence

end

function initialalphas(m :: GHLS,approx::SmolyakApproximation,aalin::Array{Float64},bblin::Array{Float64})

    #Initilize variables
    endogvar = Array{Float64}(undef,approx[:nvars])
    exogpart = Array{Float64}(undef,approx[:nvars])
    slopeconxxmsv = Array{Float64}(undef,2*approx[:nmsv])
    slopeconcont = Array{Float64}(undef,2*approx[:nexogcont])
    initialalphas = zeros(approx[:nfunc]*approx[:ngrid],2*ns)
    alphass = zeros(approx[:nfunc],approx[:ngrid])
    endogvarm1 = zeros(approx[:nvars],approx[:ngrid]) #holds lags
    nmsvplus = approx[:nmsv] + approx[:nexogcont]
    endog

    # Get conversion from xx to msv
    slopeconxxmsv[1:approx[:nmsv]] = approx.slopeconxx[1:approx[:nmsv]]
    slopeconxxmsv[approx[:nmsv]+1:2*approx[:nmsv]] = approx.slopeconxx[nmsvplus+1:nmsvplus+approx[:nmsv]]

    if (approx[:nexogcont] > 0)
        slopeconcont[1:approx[:nexogcont]] = approx.slopeconxx[approx[:nmsv]+1:nmsvplus]
        slopeconcont[approx[:nexogcont]+1:2*approx[:nexogcont]] = approx.slopeconxx[nmsvplus+approx[:nmsv]+1:2*nmsvplus]
    end

    # Converts endog var back to msv domain
    for i in 1:approx[:ngrid]
        endogvarm1[1:approx[:nmsv],i] = msv2xx(xgrid[1:approx[:nmsv],i],approx[:nmsv],slopeconxxmsv)-m.steady_state[1:approx[:nmsv]] #CHECK THIS FUNCTION
    end

    # For each shock grid point
    for ss in 1:ns
        yy = zeros(approx[:nfunc],approx[:ngrid])
        exogval = zeros(approx[:nexog])

        for i in 1:approx[:ngrid]
            exogval[1:approx[:nexogshock]] = exoggrid[1:approx[:nexogshock],ss]   #update shocks (in deviation from ss)

            if (approx[:nexogcont] > 0)
                exogval[approx[:nexog]-approx[:nexogcont]+1:approx[:nexog]] = msv2xx(xgrid[approx[:nmsv]+1:approx[:nmsv]+approx[:nexogcont],i],approx[:nexogcont],slopeconcont)-endogsteady[approx[:nvars]+approx[:nexog]-approx[:nexogcont]+1:approx[:nvars]+approx[:nexog]]
            end

            #get linear solution
            endogvar=dgemv(1.0, aalin, endogvarm1[:,i]) #REMAKE THIS FUNCTION
            exogpart=dgemv(1.0, bblin, exogval) # REMAKE THIS FUNCTION
            endogvar = endogsteady[1:approx[:nvars]] + endogvar + exogpart
            yy[:,i] = endogvar[[10,11,18,19,21,22,13] ]
        end

        # Get alphas by inverting approximation function
        alphass=dgemm(1.0,yy,approx[:bbtinv])
        for i in 1:approx[:ngrid]
            for ifunc in 1:approx[:nfunc]
                #if (alphass(ifunc,i) < 1.0e-8) alphass(ifunc,i) = 0.0d0
                initialalphas[(ifunc-1)*approx[:ngrid]+i,ss] = alphass[ifunc,i]
                #initial guess for ZLB polynomials
                initialalphas[(ifunc-1)*approx[:ngrid]+i,ns+ss] = alphass[ifunc,i]
            end
        end
    end

    return initialalphas

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
