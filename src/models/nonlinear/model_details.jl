"""
    msv2xx(msv::Vector{Float64}, nmsv::Int, slopeconmsv::Vector{Float64})
Returns endogenous state variables in [-1,1] domain or the inverse function depending on slopeconmsv.
...
# Arguments
- `msv::Vector{Float64}`: Endogenous state variables.
- `nmsv::Int`: Minimum number of endogenous state variables.
- `slopeconmsv::Vector{Float64}`: Slope and constants that depend on the upper and lower bounds of minimum state variables.
...
"""
function msv2xx(msv::Vector{Float64}, nmsv::Int, slopeconmsv::Vector{Float64})
    return slopeconmsv[1:nmsv] .* msv + slopeconmsv[nmsv+1:2*nmsv]
end

"""
    exogposition(exogvec::Vector{Int64}, nrvec::Vector{Int64}, nlength::Int64)

Computes the position of the regime and position on exogenous grid.  Hard-wired for 6 or 7 variable for speed. Returns the index for the exogenous grid.
...
- `exogvec::Vector{Int64}`: Array with index for each shock value.
- `nrvec::Vector{Int64}`: Number of realizations for each shock.
- `nlength::Int64`: Length of exogenous processes.
...
"""
function exogposition(exogvec::Vector{Int64}, nrvec::Array{Int64, 1}, nlength::Int64)

    if (nlength == 6)
        return nrvec[6]*nrvec[5]*nrvec[4]*nrvec[3]*nrvec[2]*(exogvec[1]-1) + nrvec[6]*nrvec[5]*nrvec[4]*nrvec[3]*(exogvec[2]-1) + nrvec[6]*nrvec[5]*nrvec[4]*(exogvec[3]-1) + nrvec[6]*nrvec[5]*(exogvec[4]-1) + nrvec[6]*(exogvec[5]-1) + exogvec[6]
    elseif (nlength == 5)
        return nrvec[5]*nrvec[4]*nrvec[3]*nrvec[2]*(exogvec[1]-1) + nrvec[5]*nrvec[4]*nrvec[3]*(exogvec[2]-1) + nrvec[5]*nrvec[4]*(exogvec[3]-1) + nrvec[5]*(exogvec[4]-1) + exogvec[5]
    elseif (nlength == 4)
        return nrvec[4]*nrvec[3]*nrvec[2]*(exogvec[1]-1) + nrvec[4]*nrvec[3]*(exogvec[2]-1) + nrvec[4]*(exogvec[3]-1) + exogvec[4]
    elseif (nlength == 3)
        return nrvec[3]*nrvec[2]*(exogvec[1]-1) + nrvec[3]*(exogvec[2]-1) + exogvec[3]
    elseif (nlength == 2)
        return nrvec[2]*(exogvec[1]-1) + exogvec[2]
    elseif (nlength == 1)
        return exogvec[1]
    else
        println("There can only be six shocks. You need to modify exogposition. (exogposition - module polydef).")
    end

    return 0

end

"""
    intermediatedec!(endogvar::Vector{Float64}, nvars::Int,nexog::Int,params,labss::Float64,endogvarm1::Vector{Float64},currentshockvalues::Vector{Float64},polyvar::Vector{Float64},omegapoly::Float64,polyvarplus::Vector{Float64},zlbintermediate::Bool)

Updates endogenous variables given their lagged values and polynomial approximation.
...
# Arguments
- `m::GHLS`: GHLS model object. Relevantly contains:
      `nparams::Int`: Number of structural model parameters.
      `nvars::Int`: Number of endogenous variables.
      `nexog::Int`: Number of exogenous shocks.
      `nfunc::Int`: Number of functions approximated.
      Model parameters, including m[:labss], the steady state of labor
      Endogenous variables and shock values
- `endogvarm1::Vector{Float64}`: Lagged endogenous variables and shock values.
- `currentshockvalues::Vector{Float64}`: Current shock values.
- `polyvar::Vector{Float64}`: Values of polynomial functions in normal times.
- `omegapoly::Float64`: Weight on ZLB polynomials.
- `polyvarplus::Vector{Float64}`: Values of polynomial functions in zlb times.
- `zlbintermediate::Bool`: Indicator for whether there are two regimes for the polynomial or not.
...
"""
function intermediatedec!(endogvar::Vector{Float64},nvars::Int,nexog::Int,labss::Float64,endogvarm1::Vector{Float64},currentshockvalues::Vector{Float64},polyvar::Vector{Float64},omegapoly::Float64,polyvarplus::Vector{Float64},zlbintermediate::Bool,exogenous_shocks::OrderedCollections.OrderedDict{Symbol,Int64}, params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64})

    # Transform shocks from log-level deviations from steady state to levels
    invshk::Float64 = exp(currentshockvalues[exogenous_shocks[:μ_sh]])
    techshk::Float64 = exp(currentshockvalues[exogenous_shocks[:ztil_sh]])
    rrshk::Float64 = currentshockvalues[exogenous_shocks[:rm_sh]]
    gss::Float64 = 1.0/(1.0-params[keys[:shrgy]])
    gshk::Float64 = exp( log(gss) + currentshockvalues[exogenous_shocks[:g_sh]] )
    ashk::Float64 = exp(0.)

    # Update levels of endogenous variables
    if (zlbintermediate == true)
        endogvar[10] = omegapoly*exp(polyvar[1]) + (1.0-omegapoly)*exp(polyvarplus[1]) #lam
        endogvar[11] = omegapoly*exp(polyvar[2]) + (1.0-omegapoly)*exp(polyvarplus[2]) #qq
        bp = omegapoly*polyvar[3] + (1.0-omegapoly)*polyvarplus[3]
        bww = omegapoly*polyvar[4] + (1.0-omegapoly)*polyvarplus[4]
        endogvar[21] = omegapoly*exp(polyvar[5]) + (1.0-omegapoly)*exp(polyvarplus[5]) #bc
        endogvar[22] = omegapoly*polyvar[6] + (1.0-omegapoly)*polyvarplus[6] #bi
        endogvar[13] = omegapoly*exp(polyvar[7]) + (1.0-omegapoly)*exp(polyvarplus[7]) #util
    else
        endogvar[10] = exp(polyvar[1]) #lam
        endogvar[11] = exp(polyvar[2]) #qq
        bp::Float64 = polyvar[3]
        bww::Float64 = polyvar[4]
        endogvar[21] = exp(polyvar[5]) #bc
        endogvar[22] = polyvar[6] #bi
        endogvar[13] = exp(polyvar[7]) #util
    end

    endogvar[18] = (sqrt(1.0+4.0*bp)+1.0)/2.0 #vp
    endogvar[19] = (sqrt(1.0+4.0*bww)+1.0)/2.0 #vw
    dptildem1::Float64 = (params[keys[:π_bar]]^params[keys[:ap]])*(endogvarm1[6]^(1.0-params[keys[:ap]]))
    dwtildem1::Float64 = (params[keys[:π_bar]]^params[keys[:aw]])*(endogvarm1[6]^(1.0-params[keys[:aw]]))
    gzwage::Float64 = params[keys[:gz]]*techshk^(1.0-params[keys[:aw]]) #Note that aw should be bw but in steady state they are equal, might be better style to pass bw though

    endogvar[6] = endogvar[18]*dptildem1 #dp
    endogvar[20] = endogvar[19]*dwtildem1*gzwage #dw
    endogvar[16] = endogvar[10] + (params[keys[:γ]]/params[keys[:gz]])*params[keys[:β]]*endogvar[21] #muc
    endogvar[2] = params[keys[:γ]]*endogvarm1[2]/(params[keys[:gz]]*techshk)+1.0/endogvar[16] #cc
    cquad_vi::Float64 = endogvar[22]/(endogvar[11]*invshk)-(1.0-endogvar[11]*invshk)/(params[keys[:ϕ_I]].value*endogvar[11]*invshk)

    endogvar[17] = 0.5*(1.0+sqrt(1.0+4.0*cquad_vi)) #vi

    endogvar[3] = endogvar[17]*endogvarm1[3]/techshk #inv
    aayy::Float64 = 1.0/gshk-(params[keys[:ϕ_p]]/2.0)*(endogvar[18]-1.0)*(endogvar[18]-1.0)
    rkss::Float64 = params[keys[:gz]]/params[keys[:β]]-1.0+params[keys[:δ]]
    utilcost::Float64 = (rkss/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(endogvar[13]-1.0))-1.0)
    endogvar[7] = (1.0/aayy)*( endogvar[2]+endogvar[3] + utilcost*(endogvarm1[1]/(params[keys[:gz]]*techshk)) ) #gdp
    endogvar[4] = endogvarm1[4]*endogvar[20]/(params[keys[:gz]]*techshk*endogvar[6]) #rw
    endogvar[12] = endogvar[7]^(1.0/(1.0-params[keys[:α]]))*(endogvar[13]*endogvarm1[1]/(params[keys[:gz]]*techshk))^(params[keys[:α]]/(params[keys[:α]]-1.0))/ashk #lab
    endogvar[8] = params[keys[:α]]*log(endogvar[13]) + (1-params[keys[:α]])*(log(endogvar[12])-log(labss)) # Changed llabss to log(m[:labss]), this is xhp

    lrss::Float64 = log(params[keys[:gz]]*params[keys[:π_bar]]/params[keys[:β]])
    endogvar[5] = exp(lrss+params[keys[:ρ_R]]*(log(endogvarm1[5])-lrss) + (1.0-params[keys[:ρ_R]])*(params[keys[:γ_π]]*log(endogvar[6]/params[keys[:π_bar]]) + params[keys[:γ_g]]*log(endogvar[7]*techshk/endogvarm1[7]) + params[keys[:γ_x]]*endogvar[8] ) + rrshk) #notr

    endogvar[14] = endogvar[4]*endogvar[12]/((1.0-params[keys[:α]])*endogvar[7]) #mc
    endogvar[15] = (params[keys[:α]]/(1.0-params[keys[:α]]))*(endogvar[4]*endogvar[12]*params[keys[:gz]]*techshk/(endogvar[13]*endogvarm1[1])) #rentalk
    endogvar[1] = (1.0-params[keys[:δ]])*(endogvarm1[1]/(params[keys[:gz]]*techshk)) + invshk*endogvar[3]*(1.0- (params[keys[:ϕ_I]]/2.0)*(endogvar[17]-1.0)*(endogvar[17]-1.0) ) #cap
    endogvar[9] = copy(endogvar[5]) # copy as to no make it a pointer, this is nomr

    endogvar[nvars+1:nvars+nexog] = currentshockvalues

    return nothing
end

"""
    decr!(approx::SmolyakApproximation,endogvarm1::Vector{Float64},innovations::Vector{Float64},params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64}, endog_st, alphacoeff::Array{Float64,2})

The decision rule -- updates endogenous variables and shocks given lagged endogenous values and innovations.
...
# Arguments
- `m::GHLS`: GHLS model object, including the parameters, SmolyakApproximation object, and endogenou variables and shock values
- `endogvarm1::Vector{Float64}`: Lagged endogenous variables and shock values.
- `innovations::Vector{Float64}`: Innovations to the shocks.
- `alphacoeff::Arary{Float64, 2}`: Polynomial coefficients.
...
"""
function decr!(endogvar::Vector{Float64}, nvars::Int, nexog::Int, nexogcont::Int, nexogshock::Int, nfunc::Int, ninter::Int, nmsv::Int, ngrid::Int, nshockgrid::Array{Int, 1}, shockbounds::Array{Float64, 2}, shockdistance::Array{Float64, 1}, interpolatemat::Array{Int, 2}, slopeconmsv::Array{Float64, 1}, nindplus::Int, indplus::Array{Int, 1}, exoggrid::Array{Float64, 2}, ns::Int, zlbswitch::Bool, endogvarm1::Vector{Float64},innovations::Vector{Float64},params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64}, labss::Float64, alphacoeff::Array{Float64,2},exogenous_shocks::OrderedCollections.OrderedDict{Symbol,Int64},endogenous_states::OrderedCollections.OrderedDict{Symbol,Int64})

    #Initilize Variables
    shockindexall=ones(Int64,nexog-nexogcont)
    shockindex=Array{Int64}(undef,nexogshock)
    shockindex_inter=Array{Int64}(undef,nexogshock)
    lmsv=Array{Float64}(undef,nmsv+nexogcont)
    currentshockvalues=Array{Float64}(undef,nexog)
    funcmatplus=Array{Float64}(undef,nfunc,ninter)
    weighttemp=Array{Float64}(undef,nexogshock)
    funcapp=Array{Float64}(undef,nfunc)
    funcapp_plus=Array{Float64}(undef,nfunc)
    xx=Array{Float64}(undef,nmsv)
    polyvec=Array{Float64}(undef,ngrid)

    omegaweight = 100000.0

    nmsvplus = nmsv+nexogcont

    #update shocks (all shocks in deviation from SS)
    currentshockvalues[1]::Float64 = params[keys[:ρ_η]]*endogvarm1[endogenous_states[:b_t]] + params[keys[:σ_η]]*innovations[1]
    currentshockvalues[2]::Float64 = params[keys[:ρ_μ]]*endogvarm1[endogenous_states[:μ_t]] + params[keys[:σ_μ]]*innovations[2]
    currentshockvalues[3]::Float64 = params[keys[:σ_Z]]*innovations[3]
    currentshockvalues[4]::Float64 = params[keys[:ρ_int]]*endogvarm1[endogenous_states[:mon_t]] + params[keys[:σ_R]]*innovations[4]
    currentshockvalues[5]::Float64 = params[keys[:ρ_g]]*endogvarm1[endogenous_states[:g_t]] + params[keys[:σ_g]]*innovations[5]
    currentshockvalues[6] = 0.0 #a shock

    # Finds shock index associated with current shock value (rounded down)
    @simd for i in 1:nexogshock
        if (currentshockvalues[i] < shockbounds[i,1])
            shockindex[i] = 1
        elseif (currentshockvalues[i] > shockbounds[i,2])
            shockindex[i] = floor(Int64,(shockbounds[i,2]-shockbounds[i,1])/shockdistance[i] )
        else
            shockindex[i] = floor(Int64,1.0+(currentshockvalues[i]-shockbounds[i,1])/shockdistance[i])
        end
    end

    # Lowest possible state (all indices rounded down)
    shockindexall[1:nexogshock] = shockindex
    stateindex0 = exogposition(shockindexall,nshockgrid,nexog-nexogcont)

    # Highest possible state (all indices rounded up)
    shockindexall[1:nexogshock] = shockindex + interpolatemat[:,ninter]
    stateindex1 = exogposition(shockindexall,nshockgrid,nexog-nexogcont)

    # One possible function for each possible interpolation (i.e. every combination of rounding up/down)
    funcmat = Array{Float64}(undef,nfunc,ninter)
    weightvec = Array{Float64}(undef,ninter)

    #Log the minimum state variables
    lmsv[1:nmsv] = log.(endogvarm1[1:nmsv])
    if (nexogcont > 0)
        lmsv[nmsv+1:nmsv+nexogcont] = currentshockvalues[nexog-nexogcont+1:nexog]
    end

    # Convert endogenous variables to [-1, 1] domain
    xx = msv2xx(lmsv,nmsvplus,slopeconmsv)

    # Get polynomials evaluated at these values  (this is T(ϕ(X_{t-1})))
    polyvec = smolyakpoly(nmsvplus,ngrid,nindplus,indplus,xx)

    prod_sd = prod(shockdistance)

    @inbounds @simd for i in 1:ninter
        @inbounds @simd for j in 1:nexogshock
            shockindexall[j] = shockindex[j] + interpolatemat[j,i]

            # Shock grid points that are closer to actual shock values result in lower weights
            weighttemp[j]=(1-interpolatemat[j,i])*(currentshockvalues[j]-exoggrid[j,stateindex0]) + (interpolatemat[j,i])*(exoggrid[j,stateindex1]-currentshockvalues[j])
        end
        stateindex = exogposition(shockindexall,nshockgrid,nexog-nexogcont)
        stateindexplus = stateindex+ns
        @inbounds @simd for ifunc in 1:nfunc
            @views funcmat[ifunc,i] = BLAS.dot(ngrid, alphacoeff[(ifunc-1)*ngrid+1:ifunc*ngrid, 1, stateindex],polyvec, 1)
            @views funcmatplus[ifunc,i] = BLAS.dot(ngrid, alphacoeff[(ifunc-1)*ngrid+1:ifunc*ngrid, 1, stateindexplus],polyvec, 1)
        end

        # Give weight to inverse interpolation, so that in the end interpolations that are closer to actual shocks are given high weights
        # Note prod_sd is the most prod(weighttemp) can be
        weightvec[ninter+1-i] = prod(weighttemp)/prod_sd
    end

    #In-place version of funcapp = funcmat*weightvec
    mul!(funcapp, funcmat, weightvec)

    zlbintermediate = false  #start with evaluation of 1 poly case (omegapoly and 2nd funcapp irrelevant)
    omegapoly = 1.0 #SINCE ZLBINTERMEDIATE IS FALSE

    #endogvar = Array{Float64}(undef, nvars+nexog)
    intermediatedec!(endogvar,nvars,nexog,labss, endogvarm1,currentshockvalues, funcapp,omegapoly,funcapp,zlbintermediate, exogenous_shocks, params, keys)
    if ( (endogvar[endogenous_states[:rm_t]] < 1.0) & (zlbswitch == true) ) #zlb case
        zlbintermediate = true
        omegapoly = exp(omegaweight*log(endogvar[endogenous_states[:rm_t]])) #now omegapoly and funcapp_plus relevant
        mul!(funcapp_plus, funcmatplus,  weightvec)
        intermediatedec!(endogvar,nvars,nexog, labss, endogvarm1,currentshockvalues, funcapp,omegapoly,funcapp_plus,zlbintermediate, exogenous_shocks, params, keys)
        endogvar[9] = 1.0
    end

    return nothing
end

"""
    decrlin(endogvarm1::Vector{Float64},innovations::Vector{Float64},nvars::Int,nexog::Int,sigma::Array{Float64,2},pp::Array{Float64,2})

Linear decision rule -- returns endogenous variables and shocks given lagged endogenous values and innovations.
...
# Arguments
- `endogvarm1::Vector{Float64}`: Lagged endogenous variables and shock values.
- `innovations::Vector{Float64}`: Innovations to the shocks.
- `nvars::Int`: Number of endogenous variables.
- `nexog::Int`: Number of exogenous shocks.
- `sigma::Array{Float64, 2}`: R matrix to which ϵ_t is multiplied in the transition equation. Only for the first 28 minimum state variables and first 6 shocks
- `pp::Array{Float64, 2}`: T matrix to which s_{t-1} is multiplied in the transition equation. Only for the first 28 minimum state variables.
...
"""
function decrlin(endogvarm1::Vector{Float64},innovations::Vector{Float64},nvars::Int,nexog::Int,sigma::Array{Float64,2},pp::Array{Float64,2},steady_states::Array{Float64,1})


    # Initilize variables
    xx = Vector{Float64}(undef,nvars+nexog)
    exogpart = Array{Float64}(undef,nvars+nexog)
    endogsteady = steady_states[1:nvars + nexog]

    endogvarm1[1:nvars] -= endogsteady[1:nvars]

    mul!(xx, pp, endogvarm1[1:nvars+nexog])
    mul!(exogpart,sigma,innovations)
    xx = xx + exogpart

    xx[1:nvars] += endogsteady[1:nvars]

    return xx
end

"""
    decr_euler(approx::SmolyakApproximation,gridindex::Int64,shockpos::Int64,params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64},alphacoeff::Array{Float64,2})

For a given collocation point, return associated errors.
...
# Arguments
- `m::GHLS`: GHLS model object, which relevantly includes model parameters and SmolyakApprox object. SmolyakApprox relevantly includes:
        `slopeconxx::Vector{Float64}`: Slope coefficients and constants for converting xx to msv space.
        `bbt::Array{Float64, 2}`: Matrix with Smolyak basis functions along its columns evaluated at each grid point (rows).
        `xgrid::Array{Float64, 2}`: Matrix of grid points along its columns. Each grid point is NmsvX1
- `gridindex::Int64`: Index of one of the collocation points.
- `shockpos::Int64`: Index of current exogenous state.
- `alphacoeff::Array{Float64, 2}`: Polynomial coefficients.
...
"""
function decr_euler(rkss::Float64, ninter::Int, nexogshock::Int, nfunc::Int, nexog::Int, nvars::Int, nexogcont::Int, nmsv::Int, xgrid::Array{Float64, 2}, slopeconxx::Array{Float64, 1}, exoggrid::Array{Float64, 2}, gridindex::Int64,shockpos::Int64, ngrid::Int, nshockgrid::Array{Int, 1}, bbt::Array{Float64, 2}, statezlbinfo::Array{Int64, 1}, zlbswitch::Bool, nquad::Int, ghweights::Array{Float64, 1}, ghnodes::Array{Float64, 2}, shockbounds::Array{Float64, 2}, shockdistance::Array{Float64, 1}, interpolatemat::Array{Int, 2}, slopeconmsv::Array{Float64, 1}, nindplus::Int, indplus::Array{Int ,1}, ns::Int, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64},alphacoeff::Array{Float64,2}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64})

    #Initilize Variables
    zlbinfo  = statezlbinfo[shockpos]
    polyappnew = Array{Float64}(undef,2*nfunc)
    endogvar = Array{Float64}(undef,nvars+nexog)
    endogvarzlb = Array{Float64}(undef,nvars+nexog)
    endogvarp = Array{Float64}(undef,nvars+nexog)
    endogvarzlbp = Array{Float64}(undef,nvars+nexog)
    slopeconcont = Array{Float64}(undef,2*nexogcont)
    xgridshock = Array{Float64}(undef,nexogcont)
    slopeconxxmsv = Array{Float64}(undef,2*nmsv)
    xgridmsv = Array{Float64}(undef,nmsv)
    abserror = Array{Float64}(undef,2*nfunc)
    ev = Array{Float64}(undef,12)
    exp_eul = Array{Float64}(undef,12)

    polyapp = Array{Float64}(undef,2*nfunc)
    endogvarm1 = Array{Float64}(undef,nvars+nexog)
    exp_var = zeros(12)
    innovations = Array{Float64}(undef,nexog)

    nmsvplus = nmsv+nexogcont
    if (nexogcont > 0)
        slopeconcont[1:nexogcont] = slopeconxx[nmsv+1:nmsvplus]
        slopeconcont[nexogcont+1:2*nexogcont] = slopeconxx[nmsvplus+nmsv+1:2*nmsvplus]
        xgridshock = xgrid[nmsv+1:nmsvplus,gridindex]
        currentshockvalues[nexog-nexogcont+1:nexog] = msv2xx(xgridshock,nexogcont,slopeconcont)
    end

    # Gets shock values for given position on shock grid
    currentshockvalues[1:nexogshock] = exoggrid[1:nexogshock,shockpos]

    # Calculates T(ϕ(X_{t-1}))α_{l,j,k} for each function
    shockpospoly = shockpos + ns
    for ifunc in 1:nfunc
        polyapp[ifunc] = BLAS.dot(ngrid, alphacoeff[(ifunc-1)*ngrid+1:ifunc*ngrid,shockpos],1,bbt[:,gridindex], 1)
        polyapp[nfunc+ifunc] = BLAS.dot(ngrid, alphacoeff[(ifunc-1)*ngrid+1:ifunc*ngrid,shockpospoly],1,bbt[:,gridindex], 1)
    end

    # Get endogenous variable values for that grid point and convert back from [-1, 1] domain
    xgridmsv = xgrid[1:nmsv,gridindex]
    slopeconxxmsv[1:nmsv] = slopeconxx[1:nmsv]
    slopeconxxmsv[nmsv+1:2*nmsv] = slopeconxx[nmsvplus+1:nmsvplus+nmsv]
    endogvarm1[1:nmsv] = exp.( msv2xx(xgridmsv,nmsv,slopeconxxmsv) )

    zlbintermediate = false  #force evaluation of 1 poly case at date t
    omegapoly = 1.0 #SET OMEGAPOLY TO 1 SINCE ZLBINTERMEDIATE IS FALSE

    # Updates given polynomial approximations
    intermediatedec!(endogvar,nvars,nexog, labss, endogvarm1,currentshockvalues, polyapp[1:nfunc],omegapoly,polyapp[1:nfunc],zlbintermediate, exogenous_shocks, params, keys)

    if ((zlbinfo != 0) & (zlbswitch == true))
        intermediatedec!(endogvar, nvars,nexog, labss, endogvarm1,currentshockvalues, polyapp[nfunc+1:2*nfunc], omegapoly,polyapp[nfunc+1:2*nfunc],zlbintermediate, exogenous_shocks, params, keys)
        endogvarzlb[9] = 1.0
    end

    # Calculates conditional expectations
    for ss in 1:nquad
        innovations[1:nexogshock] = ghnodes[:,ss]
        if (nexogcont > 0)
            innovations[nexog-nexogcont+1:nexog] =  ghnodes[nexogshock+1:nexogshock+nexogcont,ss]
        end

        decr!(endogvarp,nvars, nexog, nexogcont, nexogshock, nfunc, ninter, nmsv, ngrid, nshockgrid, shockbounds, shockdistance, interpolatemat, slopeconmsv, nindplus, indplus, exoggrid, ns, zlbswitch,endogvar,innovations,params, keys, labss, alphacoeff, exogenous_shocks, endogenous_states)

        techshkp = exp(endogvarp[25])
        invshkp = exp(endogvarp[24])
        ev[1] = endogvarp[10]/(endogvarp[6]*techshkp)

        utilcostp = (rkss/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(endogvarp[13]-1.0))-1.0) # a(u_t)
        ev[2] = (endogvarp[10]/techshkp)*( (endogvarp[15]*endogvarp[13])-utilcostp+(1.0-params[keys[:δ]])*endogvarp[11])
        ev[3] = endogvarp[10]*(endogvarp[18]-1.0)*endogvarp[18]*endogvarp[7]
        ev[4] = (endogvarp[19]-1.0)*endogvarp[19]
        ev[5] = endogvarp[16]/techshkp
        ev[6] = endogvarp[10]*endogvarp[11]*invshkp*(endogvarp[17]-1.0)*endogvarp[17]*endogvarp[17]

        if ((zlbinfo != 0) & (zlbswitch == true))
            decr!(endogvarzlbp,nvars, nexog, nexogcont, nexogshock, nfunc, ninter, nmsv, ngrid, nshockgrid, shockbounds, shockdistance, interpolatemat, slopeconmsv, nindplus, indplus, exoggrid, ns, zlbswitch,endogvarzlb,innovations, params, keys, labss, alphacoeff, exogenous_shocks, endogenous_states, test)
            ev[7] = endogvarzlbp[10]/(endogvarzlbp[6]*techshkp)
            utilcostp = (rkss/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(endogvarzlbp[13]-1.0))-1.0)
            ev[8] = (endogvarzlbp[10]/techshkp)*( (endogvarzlbp[15]*endogvarzlbp[13])-utilcostp+(1-params[16])*endogvarzlbp[11] )
            ev[9] = endogvarzlbp[10]*(endogvarzlbp[18]-1.0)*endogvarzlbp[18]*endogvarzlbp[7]
            ev[10] = (endogvarzlbp[19]-1.0)*endogvarzlbp[19]
            ev[11] = endogvarzlbp[16]/techshkp
            ev[12] = endogvarzlbp[10]*endogvarzlbp[11]*invshkp*(endogvarzlbp[17]-1.0)*endogvarzlbp[17]*endogvarzlbp[17]
        else
            ev[7:12] = ev[1:6]
        end

        # Approximate expectation by taking weighted sum over integration nodes
        exp_var += ghweights[ss]*ev
    end

    liqshk = exp(endogvar[23]) # Change to m.endogenous_states[:b_t]?
    invshk = exp(endogvar[24]) # Change to m.endogenous_states[:μ_t]?
    ep = params[keys[:ϵ_p]].scaledvalue # Is this constant elasticity of demand?

    exp_eul[1] =  (params[keys[:β]]/params[keys[:gz]])*liqshk*endogvar[9]*exp_var[1]
    exp_eul[2] =  (params[keys[:β]]/params[keys[:gz]])*exp_var[2]/endogvar[10]
    exp_eul[3] =  params[keys[:β]]*exp_var[3]/(endogvar[10]*endogvar[7])+(params[keys[:ϵ_p]]/params[keys[:ϕ_p]])*(endogvar[14]-(params[keys[:ϵ_p]]-1.0)/params[keys[:ϵ_p]])
    exp_eul[4] = params[keys[:β]]*exp_var[4]+(params[keys[:ϵ_w]]/params[keys[:ϕ_w]])*endogvar[10]*endogvar[12]*
           ( (params[keys[:ψ_L]]*endogvar[12]^params[keys[:σ_L]]/endogvar[10])-((params[keys[:ϵ_w]]-1.0)/params[keys[:ϵ_w]])*endogvar[4] )
    exp_eul[5] = exp_var[5]
    exp_eul[6] = (params[keys[:β]]/params[keys[:gz]])*exp_var[6]/endogvar[10] - 0.5*endogvar[11]*invshk*(endogvar[17]-1.0)*(endogvar[17]-1.0)

    polyappnew[1] = log(exp_eul[1])
    polyappnew[2] = log(exp_eul[2])
    polyappnew[3] = exp_eul[3]
    polyappnew[4] = exp_eul[4]
    polyappnew[5] = log(exp_eul[5])
    polyappnew[6] = exp_eul[6]
    polyappnew[7] = log( 1.0 + (1/params[keys[:σ_a]])*log(endogvar[15]/rkss) )

    if ((zlbinfo != 0) & (zlbswitch == true))
        exp_eul[7] =  (params[keys[:β]]/params[keys[:gz]])*liqshk*endogvarzlb[9]*exp_var[7]
        exp_eul[8] = (params[keys[:β]]/params[keys[:gz]])*exp_var[8]/endogvarzlb[10]
        exp_eul[9] =  params[keys[:β]]*exp_var[9]/(endogvarzlb[10]*endogvarzlb[7])+(params[keys[:ϵ_p]]/params[keys[:ϕ_p]])*( endogvarzlb[14]-(params[keys[:ϵ_p]]-1.0)/params[keys[:ϵ_p]])
        exp_eul[10] = params[keys[:β]]*exp_var[10]+(params[keys[:ϵ_w]]/params[keys[:ϕ_w]])*endogvarzlb[10]*endogvarzlb[12]*
           ( (params[keys[:ψ_L]]*endogvarzlb[12]^params[keys[:σ_L]]/endogvarzlb[10])-((params[keys[:ϵ_w]]-1.0)/params[keys[:ϵ_w]])*endogvarzlb[4])
        exp_eul[11] = exp_var[11]
        exp_eul[12] = (params[keys[:β]]/params[keys[:gz]])*exp_var[12]/endogvarzlb[10] - 0.5*endogvarzlb[11]*invshk*(endogvarzlb[17]-1.0)*(endogvarzlb[17]-1.0)

        polyappnew[8] = log(exp_eul[7])
        polyappnew[9] = log(exp_eul[8])
        polyappnew[10] = exp_eul[9]
        polyappnew[11] = exp_eul[10]
        polyappnew[12] = log(exp_eul[11])
        polyappnew[13] = exp_eul[12]
        polyappnew[14] = log( 1.0 + (1/params[keys[:σ_a]])*log(endogvarzlb[15]/rkss) )
    else
        exp_eul[7:12] = exp_eul[1:6]
        polyappnew[8:14] = polyappnew[1:7]
    end

    # Gets average error across all polynomials
    erravg = 0.0
    for ifunc in 1:2*nfunc
        abserror[ifunc] = abs(polyappnew[ifunc]-polyapp[ifunc])
        erravg += abserror[ifunc]
    end

    erravg /= (2*nfunc)

    return polyappnew, erravg

end

"""
    finite_grid(n::Int64,rho::Float64,sigmaep::Float64)

For a given shock, this populates the shockgrid array with the possible values of the shock, spanning -nu to nu and evenly spaced. It also records the shockdistance (distance between shock values) and shockbounds (min and max shock values).
...
# Arguments
- `shockgrid::Array{Float64, 1}: Array containing possible values of shock
- `shockdistance::Float64: Distance between values of shocks on grid
- `shockbounds::Array{Float64, 2}: Minimum and maximum values of shock
- `n::Int`: Number of shock realizations.
- `rho::Float64`: AR(1) coefficient.
- `sigmaep::Float64`: Standard deviation of innovation.
...
"""
function finite_grid!(shockgrid::Array{Float64, 1}, shockdistance::Float64, shockbounds::Array{Float64, 2}, n::Int,rho::Float64,sigmaep::Float64)

    #Initialize variables
    maxgridstd = 3.0

    #Maximum value shock will take on
    @fastmath nu = sqrt(1.0/(1.0-rho^2))*maxgridstd*sigmaep

    #The last point on the grid is the max value
    shockgrid[n] = nu
    shockbounds[2]  = nu #Note that shockbounds is actually a row vector and therefore 2D, this syntax gets the second element in column-major order

    #The first point is the minimum (which is negative of the maximum)
    shockgrid[1]  = -1*shockgrid[n]
    shockbounds[1] = -1*shockgrid[n]

    #Every point in between is evenly spaced
    shockdistance = @fastmath (shockgrid[n] - shockgrid[1]) / (n - 1)

    @inbounds @simd for i in 2:n-1
        shockgrid[i] = @fastmath shockgrid[1] + shockdistance * (i - 1)
    end

    return nothing
end

"""
    get_shockdetails(approx::SmolyakApproximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64}

For each exogenous shock, this associates a given grid point with the value that that shock takes on at that grid point. In total there are  'ns' such points, with each point representing a distinct combination of shock values. Also returns the upper and lower bounds of each shock, as well as the distance between shock values for any two grid points.
...
- `m::GHLS`: GHLS model object, which relevantly includes (within SmolyakApprox object):
        - `nparams::Int`: Number of parameter values.
        - `nexog::Int`: Number of exogenous shock processes.
        - `nexogshock::Int`: Number of exogenous shock processes in finite-element part of approximation.
        - `nexogcont::Int`: number of exogenous shocks used in poly approximation.
        - `ns::Int`: Number of exogenous states.
        - `number_shock_values::Int`: Number of shock values.
        - `nshockgrid::Vector{Float64}`: Array indexing number of realizations for each shock (length nexog).
        - `exogvarinfo`
...
"""
function get_shockdetails(nexogcont::Int, number_shock_values::Int, nshockgrid::Array{Int,1}, exogvarinfo::Array{Int64, 2}, nexogshock::Int, ns::Int, nexog::Int, params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64})

    #Initilize Variables
    shockbounds = Array{Float64}(undef,nexogshock,2)
    shockdistance = Array{Float64}(undef,nexogshock)
    currentshockindex = Array{Float64}(undef,nexogshock)
    nshocksum_vec = Array{Float64}(undef,nexogshock)
    shockvalues = Array{Float64}(undef,number_shock_values)
    exoggrid = zeros(nexog-nexogcont,ns) #Not undef since sometimes not every row is initialized (when nexogshock < nexog-nexogcont)

    rhovec = [params[keys[:ρ_η]].value,params[keys[:ρ_μ]].value,0.0,params[keys[:ρ_int]].value,params[keys[:ρ_g]].value, 0.0]
    sigmavec = [params[keys[:σ_η]].scaledvalue,params[keys[:σ_μ]].scaledvalue,params[keys[:σ_Z]].scaledvalue,params[keys[:σ_R]].scaledvalue,params[keys[:σ_g]].scaledvalue,0.0] #we should remove ashk it is not used

    # Tracks the total number of shock values that have been inputted into shockvalues array
    numshockvalues = 0

    # For each shock populate the shockvalues array with the values it can take on
    @simd for i in 1:nexogshock

        # Use views so that array slices can be mutated
        @views finite_grid!(shockvalues[numshockvalues+1:numshockvalues+nshockgrid[i]], shockdistance[i], shockbounds[i, :], nshockgrid[i],rhovec[i],sigmavec[i])
        numshockvalues += nshockgrid[i]
    end

    # For each distinct combination of shocks
    @simd for ss in 1:ns
        numshockvalues = 0

        # Put the shock value (rather than index as is stored in exogvarinfo) in to exoggrid
        for i in 1:nexogshock
            # Shock values is 1D (which makes it easier to accomodate different shocks having a different number of values) so need to offset index from exogvarinfo by numshockvalues to get correct value
            exoggrid[i, ss] = shockvalues[numshockvalues + exogvarinfo[i, ss]]
            numshockvalues += nshockgrid[i]
        end
    end

    return exoggrid,shockbounds,shockdistance

end
