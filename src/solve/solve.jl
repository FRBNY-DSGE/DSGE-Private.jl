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

function solve(m::GHLS, parallel::Bool=true)

    m.approx.exoggrid, m.approx.shockbounds, m.approx.shockdistance = get_shockdetails(m)

    m.approx.endog_emean, m.approx.zlbfrequency, m.approx.msvbounds, m.approx.statezlbinfo, m.approx.convergence = simulate_linear(m)

    m.approx.slopeconmsv, m.approx.slopeconxx = create_slopes(m)

    # Construct starting guess
    α_initial = initial_α(m)

    α_star, convergence = if parallel
        fixedpoint_parallel(m, α_initial)
    else
        fixedpoint(m, α_initial)
    end

    return α_star
end

function create_slopes(m::GHLS)

    #Total number of state variables - these values should come from model
    nmsvplus = m.approx.nmsv + m.approx.nexogcont

    # Conversion from state domain (msv) to [-1, 1] domain (xx)
    slopeconmsv = zeros(2*nmsvplus)
    slopeconmsv[1:nmsvplus] = 2.0 ./  (m.approx.msvbounds[nmsvplus + 1 :  2*nmsvplus] - m.approx.msvbounds[1:nmsvplus])
    slopeconmsv[nmsvplus+1:2*nmsvplus] = -2.0 * m.approx.msvbounds[1:nmsvplus] ./ (m.approx.msvbounds[nmsvplus + 1 : 2*nmsvplus] .- m.approx.msvbounds[1:nmsvplus]) .- 1.

    # Conversion from xx to msv domains
    slopeconxx = zeros(2*nmsvplus)
    slopeconxx[1:nmsvplus] =  0.5 * (m.approx.msvbounds[nmsvplus + 1 : 2*nmsvplus] - m.approx.msvbounds[1:nmsvplus])
    slopeconxx[nmsvplus+1:2*nmsvplus] = m.approx.msvbounds[1:nmsvplus] + 0.5 * (m.approx.msvbounds[nmsvplus + 1 : 2*nmsvplus] - m.approx.msvbounds[1:nmsvplus])

    return slopeconmsv, slopeconxx
end

function lindecrule_markov(m::GHLS)

 #Is this just dividing out the QQQ?

    #Initilize Variables
    aalin=zeros(m.approx.nvars,m.approx.nvars)
    bblin=zeros(m.approx.nvars,m.approx.nexog)

    Γ0, Γ1, C, Ψ, Π = eqcond(m)

    TTT_gensys, CCC_gensys, RRR_gensys, eu = gensys(Γ0, Γ1, C, Ψ, Π, 1+1e-6, verbose = :high)

    # Check for LAPACK exception, existence and uniqueness
    if eu[1] != 1 || eu[2] != 1
        throw(GensysError())
    end
    TTT = real(TTT_gensys)
    RRR = real(RRR_gensys)
    CCC = real(CCC_gensys)

    # Augment states
    #TTT, RRR, CCC = augment_states(m, TTT_gensys, RRR_gensys, CCC_gensys)

    QQQ = zeros(8, 8)
    QQQ[1, 1] = m[:σ_η]^2.0
    QQQ[2, 2] = m[:σ_μ]^2.0
    QQQ[3, 3] = m[:σ_Z]^2.0
    QQQ[4, 4] = m[:σ_R]^2.0
    QQQ[5, 5] = m[:σ_g]^2.0
    QQQ[6, 6] = m[:σ_elast]^2.0
    QQQ[7, 7] = m[:σ_elastw]^2.0
    QQQ[8, 8] = m[:σ_a]^2.0

    RRR *= sqrt.(QQQ)
    pp = TTT[1:28, 1:28]
    sigma = RRR[1:28, 1:6]

    aalin = pp[1:m.approx.nvars,1:m.approx.nvars]
    for i in 1:m.approx.nexogshock
        bblin[:,i] = sigma[1:m.approx.nvars,i]/sigma[m.approx.nvars+i,i]
    end

    for i in 1:m.approx.nexogcont

        bblin[:,m.approx.nexog-i+1] = sigma[1:m.approx.nvars,m.approx.nexog-i+1]/sigma[m.approx.nvars+m.approx.nexog-i+1,m.approx.nexog-i+1]
    end

    return aalin,bblin

end


# Matrix multiplication
function dgemm(α::Float64, A::Array{Float64}, B::Array{Float64})
    C = α*A*B
    return C
end

function fixedpoint(m::GHLS, α_initial::Array{Float64,2})

    # Initialize
    α_star = copy(α_initial)
    α_new = zeros(Float64, m.approx.nfunc*m.approx.ngrid, 2*m.approx.ns)
    α_temp = zeros(Float64, 2*m.approx.nfunc, m.approx.ngrid)
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

        for j in 1:m.approx.ns
            updated_approx_polynomials = zeros(2*m.approx.nfunc, m.approx.ngrid)
            err = 0.0

            # Update polynomials using new guess for α
            for k in 1:m.approx.ngrid
                updated_approx_polynomials[:, k], err2 = decr_euler(m, k, j, α_star)
                err += err2
            end

            # Solve for α by multiplying by inverse matrix and then reindex
            α_temp = dgemm(1.0, updated_approx_polynomials,m.approx.bbtinv)
            for k in 1:m.approx.ngrid
                for l in 1:m.approx.nfunc
                    α_new[(l - 1)*m.approx.ngrid+ k, j] = α_temp[l, k]
                    α_new[(l - 1)*m.approx.ngrid+ k, m.approx.ns + j] = α_temp[m.approx.nfunc + l, k]
                end
            end

            avg_error += err
        end

        # Normalize summed error to get average
        avg_error /= 2*m.approx.ngrid*m.approx.ns

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

function simulate_linear(m::GHLS)
    # Initialize variables - some of these should be in model settings
    approx = m.approx
    total_periods = 100000
    periods_per_iter = 400
    countzlb = 0
    count_exploded = 0
    counter = 1
    convergence = true

    statezlbinfo = zeros(Int64, approx.ns)
    endog_emean = Array{Float64}(undef, approx.nvars + approx.nexog)
    msvbounds = Array{Float64}(undef, 2*(approx.nmsv + approx.nexogcont))
    shockindex = Array{Int64}(undef, approx.nexog - approx.nexogcont)
    countzlbstates = zeros(Int64, approx.ns)
    msvhigh = Array{Float64}(undef, approx.nmsv)
    msvlow = Array{Float64}(undef, approx.nmsv)
    innovations = Array{Float64}(undef, approx.nexog)
    xrandn = Array{Float64}(undef, approx.nexog, total_periods)
    msv_std = zeros(approx.nmsv + approx.nexogcont)
    endogvar = zeros(approx.nvars+approx.nexog, total_periods+1)

    # Set up random generator
    path = dirname(@__FILE__)
    iseed = MersenneTwister(101294)
    randn!(iseed, xrandn)
    xrandnFortran=readdlm("$path/xrandn.txt") #For testing
    xrandn=reshape(xrandnFortran,approx.nexog,total_periods)

    # Set up for first period
    endogvar[1:approx.nmsv,1]= [i.value for i in m.steady_state[1:approx.nmsv]]
    msvhigh = log(2.0) + endogvar[1:approx.nmsv, 1]
    msvlow = log(0.01) + endogvar[1:approx.nmsv, 1]

    Γ0, Γ1, C, Ψ, Π = eqcond(m)
    TTT_gensys, CCC_gensys, RRR_gensys, eu= gensys(Γ0, Γ1, C, Ψ, Π, 1+1e-6, verbose = :high)

    # Check for LAPACK exception, existence and uniqueness
    if eu[1] != 1 || eu[2] != 1
        throw(GensysError())
    end

    TTT_gensys = real(TTT_gensys)
    RRR_gensys = real(RRR_gensys)
    CCC_gensys = real(CCC_gensys)

    # Augment states
    #TTT, RRR, CCC = augment_states(m, TTT_gensys, RRR_gensys, CCC_gensys)
    TTT, RRR, CCC = TTT_gensys, RRR_gensys, CCC_gensys

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

    #zero out shocks not included in nonlinear model
    if (m.approx.nexogshock + m.approx.nexogcont < m.approx.nexog)
        sigma[:,m.approx.nexogshock+1:m.approx.nexog-m.approx.nexogcont] .= 0.0
    end

    while counter <= total_periods + 1
        explosiveerror = false
        llim = counter+1
        ulim = min(total_periods+1,counter+periods_per_iter)

        # Simulate each period
        for ttsim in llim:ulim

            # Draw innovations
            innovations[1:approx.nexogshock] = xrandn[1:approx.nexogshock,ttsim-1]

            if (approx.nexogcont > 0)
                innovations[approx.nexog-approx.nexogcont+1:approx.nexog] = xrandn[approx.nexog-approx.nexogcont+1:approx.nexog,ttsim-1]
            end

            #THIS SHOULD BE DONE DIFFERENTLY HERE
            endogvar[:,ttsim] = decrlin(endogvar[:,ttsim-1],innovations,m,sigma,pp)

            # Account for ZLB, why is this not 1 though?
            if (endogvar[5,ttsim] < 0.0)
                countzlb = countzlb + 1

                fill!(shockindex,1)

                # Assigns each shock to position in grid
                for i in 1:approx.nexogshock
                    # If lower than lowest grid value assign lowest possible index
                    if (endogvar[approx.nvars+i,ttsim] < approx.shockbounds[i,1])
                        shockindex[i] = 1

                    # Otherwise assign it the index whose corresponding value is closest (rounding down)
                    else
                        shockindex[i]  = min(approx.nshockgrid[i], floor(1.0+(endogvar[approx.nvars+i,ttsim]-approx.shockbounds[i,1])/approx.shockdistance[i]))
                    end
                end

                # Indexes state that gave zlb
                stateindex = exogposition(shockindex,approx.nshockgrid,approx.nexog-approx.nexogcont)

                countzlbstates[stateindex] = countzlbstates[stateindex] + 1
                if (countzlbstates[stateindex] > 5)
                    statezlbinfo[stateindex] = 1
                end
            end
        end

        # Loop to check for explosive path
        for ttsim in llim:ulim
            # Path was non-explosive so long as it was within bounds
            non_explosive = ( all(msvlow.<endogvar[1:approx.nmsv,ttsim].<msvhigh) )
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
    for i in 1:approx.nmsv
        msv_std[i] = sqrt(sum((endogvar[i,2:total_periods+1].-endog_emean[i]).^2 )/(total_periods-1))
    end

    msvbounds[1:approx.nmsv] = endog_emean[1:approx.nmsv]-scalebd*msv_std

    nmsvplus = approx.nmsv + approx.nexogcont # Added this line
    msvbounds[nmsvplus+1:nmsvplus+approx.nmsv] = endog_emean[1:approx.nmsv]+scalebd*msv_std

    # Calculate std for shocks
    for i in 1:approx.nexogcont
        msv_std[approx.nmsv+i] = sqrt(sum((endogvar[approx.nvars+approx.nexog-i+1,2:total_periods+1] -endog_emean[approx.nvars+approx.nexog-i+1]).^2 )/(total_periods-1))
        msvbounds[approx.nmsv+i] = endog_emean[approx.nvars+approx.nexog-i+1]-scalebd*msv_std[approx.nmsv+i]
        msvbounds[nmsvplus+approx.nmsv+i] = endog_emean[approx.nvars+approx.nexog-i+1]+scalebd*msv_std[approx.nmsv+i]
    end

    return endog_emean,zlbfrequency,msvbounds,statezlbinfo,convergence

end

function initial_α(m :: GHLS)

    #Initilize variables
    endogvar = Array{Float64}(undef,m.approx.nvars)
    exogpart = Array{Float64}(undef,m.approx.nvars)
    slopeconxxmsv = Array{Float64}(undef,2*m.approx.nmsv)
    slopeconcont = Array{Float64}(undef,2*m.approx.nexogcont)
    initialalphas = zeros(m.approx.nfunc*m.approx.ngrid,2*m.approx.ns)
    alphass = zeros(m.approx.nfunc,m.approx.ngrid)
    endogvarm1 = zeros(m.approx.nvars,m.approx.ngrid) #holds lags
    nmsvplus = m.approx.nmsv + m.approx.nexogcont
    endogsteady = [i.value for i in m.steady_state[1:m.approx.nvars + m.approx.nexog]]

    aalin, bblin = lindecrule_markov(m)

    # Get conversion from xx to msv
    slopeconxxmsv[1:m.approx.nmsv] = m.approx.slopeconxx[1:m.approx.nmsv]
    slopeconxxmsv[m.approx.nmsv+1:2*m.approx.nmsv] = m.approx.slopeconxx[nmsvplus+1:nmsvplus+m.approx.nmsv]

    if (m.approx.nexogcont > 0)
        slopeconcont[1:m.approx.nexogcont] = m.approx.slopeconxx[m.approx.nmsv+1:nmsvplus]
        slopeconcont[m.approx.nexogcont+1:2*m.approx.nexogcont] = m.approx.slopeconxx[nmsvplus+m.approx.nmsv+1:2*nmsvplus]
    end

    # Converts endog var back to msv domain
    for i in 1:m.approx.ngrid
        endogvarm1[1:m.approx.nmsv,i] = msv2xx(m.approx.xgrid[1:m.approx.nmsv,i],m.approx.nmsv,slopeconxxmsv)-endogsteady[1:m.approx.nmsv] #CHECK THIS FUNCTION
    end

    # For each shock grid point
    for ss in 1:m.approx.ns
        yy = zeros(m.approx.nfunc,m.approx.ngrid)
        exogval = zeros(m.approx.nexog)

        for i in 1:m.approx.ngrid
            exogval[1:m.approx.nexogshock] = m.approx.exoggrid[1:m.approx.nexogshock,ss]   #update shocks (in deviation from ss)

            if (m.approx.nexogcont > 0)
                exogval[m.approx.nexog-m.approx.nexogcont+1:m.approx.nexog] = msv2xx(m.approx.xgrid[m.approx.nmsv+1:m.approx.nmsv+m.approx.nexogcont,i],m.approx.nexogcont,slopeconcont)-endogsteady[m.approx.nvars+m.approx.nexog-m.approx.nexogcont+1:m.approx.nvars+m.approx.nexog]
            end

            #get linear solution
            endogvar=dgemv(1.0, aalin, endogvarm1[:,i]) #REMAKE THIS FUNCTION
            exogpart=dgemv(1.0, bblin, exogval) # REMAKE THIS FUNCTION
            endogvar = endogsteady[1:m.approx.nvars] + endogvar + exogpart
            yy[:,i] = endogvar[[10,11,18,19,21,22,13] ]
        end

        # Get alphas by inverting approximation function
        alphass=dgemm(1.0,yy,m.approx.bbtinv)
        for i in 1:m.approx.ngrid
            for ifunc in 1:m.approx.nfunc
                #if (alphass(ifunc,i) < 1.0e-8) alphass(ifunc,i) = 0.0d0
                initialalphas[(ifunc-1)*m.approx.ngrid+i,ss] = alphass[ifunc,i]
                #initial guess for ZLB polynomials
                initialalphas[(ifunc-1)*m.approx.ngrid+i,m.approx.ns+ss] = alphass[ifunc,i]
            end
        end
    end

    return initialalphas

end

function dgemv(alpha::Real,A::Array,x::Array)

    z = alpha*A*x

    return z

end

function parallel_help(m::GHLS,α_star::Array{Float64,2},j::Int)
    col1 = zeros(m.approx.nfunc*m.approx.ngrid,3)

    updated_approx_polynomials = zeros(2*m.approx.nfunc, m.approx.ngrid)
    err = 0.0

    # Update polynomials using new guess for α
    for k in 1:m.approx.ngrid
        updated_approx_polynomials[:, k], err2 = decr_euler(m, k, j, α_star)
        err += err2
    end

        # Solve for α by multiplying by inverse matrix and then reindex
        α_temp = dgemm(1.0, updated_approx_polynomials,m.approx.bbtinv)
        for k in 1:m.approx.ngrid
            for l in 1:m.approx.nfunc
                col1[(l - 1)*m.approx.ngrid+ k,1] = α_temp[l,k]
                col1[(l - 1)*m.approx.ngrid+ k,2] = α_temp[m.approx.nfunc + l, k]
            end
        end

    col1[1,3] = err

    return col1
end


# Parallel Fixedpoint
function fixedpoint_parallel(m::GHLS, α_initial::Array{Float64,2})

    # Initialize
    α_star = copy(α_initial)
    α_new = zeros(Float64, m.approx.nfunc*m.approx.ngrid, 2*m.approx.ns)
    α_temp = zeros(Float64, 2*m.approx.nfunc, m.approx.ngrid)
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
        α_here = @sync @distributed (hcat) for j in 1:m.approx.ns #mystart:myend
            parallel_help(m,α_star,j)
        end

        α_new[:,1:m.approx.ns] = α_here[:,1:3:end]
        α_new[:,m.approx.ns+1:end] = α_here[:,2:3:end]
        avg_error = sum(α_here[1,3:3:end])

        # Normalize summed error to get average
        avg_error /= 2*m.approx.ngrid*m.approx.ns

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
