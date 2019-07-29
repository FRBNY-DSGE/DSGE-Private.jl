"""
    float_dot(x:: SubArray{Float64,1,Array{Float64,2},Tuple{UnitRange{Int64},Int64},true},y::Array{Float64, 1})

Computes the dot product between two vectors with the same length, the first given by aplpying the view function below.
# Examples
```julia-repl
julia> float_dot(view([1,2,4,6], 1, 3, 1), [1, 2, 4])
1
```
"""
function float_dot(x:: SubArray{Float64,1,Array{Float64,2},Tuple{UnitRange{Int64},Int64},true},y::Array{Float64, 1})

    dot=0.0
    @simd for i in 1:length(y)
        dot = dot+x[i]*y[i]
    end
    return dot
end

"""
    view(x, l::Int, u::Int, i::Int)
Returns the ith column of x  at the rows between l and u. u must be at least as large as l and l >= 1 while u <= nrow(x). Also, 1 <= i <= ncol(x).
"""
@views view(x,l::Int,u::Int,i::Int) = x[l:u,i]  #More quickly takes slice of an array

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
function exogposition(exogvec::Vector{Int64}, nrvec::Array{Int64, 2}, nlength::Int64)

    #Initilize variables
    exogposition =0

    if (nlength == 6)
        exogposition = nrvec[6]*nrvec[5]*nrvec[4]*nrvec[3]*nrvec[2]*(exogvec[1]-1) + nrvec[6]*nrvec[5]*nrvec[4]*nrvec[3]*(exogvec[2]-1) + nrvec[6]*nrvec[5]*nrvec[4]*(exogvec[3]-1) + nrvec[6]*nrvec[5]*(exogvec[4]-1) + nrvec[6]*(exogvec[5]-1) + exogvec[6]
    elseif (nlength == 5)
        exogposition = nrvec[5]*nrvec[4]*nrvec[3]*nrvec[2]*(exogvec[1]-1) + nrvec[5]*nrvec[4]*nrvec[3]*(exogvec[2]-1) + nrvec[5]*nrvec[4]*(exogvec[3]-1) + nrvec[5]*(exogvec[4]-1) + exogvec[5]
    elseif (nlength == 4)
        exogposition = nrvec[4]*nrvec[3]*nrvec[2]*(exogvec[1]-1) + nrvec[4]*nrvec[3]*(exogvec[2]-1) + nrvec[4]*(exogvec[3]-1) + exogvec[4]
    elseif (nlength == 3)
        exogposition = nrvec[3]*nrvec[2]*(exogvec[1]-1) + nrvec[3]*(exogvec[2]-1) + exogvec[3]
    elseif (nlength == 2)
        exogposition = nrvec[2]*(exogvec[1]-1) + exogvec[2]
    elseif (nlength == 1)
        exogposition = exogvec[1]
    else
        println("There can only be six shocks. You need to modify exogposition. (exogposition - module polydef).")
    end

    return exogposition

end

"""
    intermediatedec(nvars::Int,nexog::Int,params,labss::Float64,endogvarm1::Vector{Float64},currentshockvalues::Vector{Float64},polyvar::Vector{Float64},omegapoly::Float64,polyvarplus::Vector{Float64},zlbintermediate::Bool)

Returns endogenous variables and shock values given their lagged values and polynomial approximation.
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
function intermediatedec(nvars::Int,nexog::Int,labss::Float64,endogvarm1::Vector{Float64},currentshockvalues::Vector{Float64},polyvar::Vector{Float64},omegapoly::Float64,polyvarplus::Vector{Float64},zlbintermediate::Bool,exogenous_shocks::OrderedDict{Symbol,Int64}, params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64})

    #Input
    keys = [i.key for i in params]
    values = [i.value for i in params]
    shrgy = values[findfirst(keys .== ":shrgy")]

    #Initilize Variables
    endogvar=Array{Float64}(undef,nvars+nexog)

    invshk = exp(currentshockvalues[exogenous_shocks[:μ_sh]])
    techshk = exp(currentshockvalues[exogenous_shocks[:ztil_sh]])
    rrshk = currentshockvalues[exogenous_shocks[:rm_sh]]
    gss = 1.0/(1.0-shrgy)
    gshk = exp( log(gss) + currentshockvalues[exogenous_shocks[:g_sh]] )
    ashk = exp(0) # Which of the last 3 shocks is this?

    capm1 = endogvarm1[1] # There is no k_t1 in m.endogenous_states
    ccm1 = endogvarm1[2]
    invm1 = endogvarm1[3]
    rwm1 = endogvarm1[4]
    notrm1 = endogvarm1[5] # Add in endogenous_states
    dpm1 = endogvarm1[6] # Add in endogenous_states
    gdpm1 = endogvarm1[7]

    # Not changing the values in model object
    if (zlbintermediate == true)
        lam = omegapoly*exp(polyvar[1]) + (1.0-omegapoly)*exp(polyvarplus[1])
        qq = omegapoly*exp(polyvar[2]) + (1.0-omegapoly)*exp(polyvarplus[2])
        bp = omegapoly*polyvar[3] + (1.0-omegapoly)*polyvarplus[3]
        bww = omegapoly*polyvar[4] + (1.0-omegapoly)*polyvarplus[4]
        bc = omegapoly*exp(polyvar[5]) + (1.0-omegapoly)*exp(polyvarplus[5])
        bi = omegapoly*polyvar[6] + (1.0-omegapoly)*polyvarplus[6]
        util = omegapoly*exp(polyvar[7]) + (1.0-omegapoly)*exp(polyvarplus[7])
    else
        lam = exp(polyvar[1])
        qq = exp(polyvar[2])
        bp = polyvar[3]
        bww = polyvar[4]
        bc = exp(polyvar[5])
        bi = polyvar[6]
        util = exp(polyvar[7])
    end

    vp = (sqrt(1.0+4.0*bp)+1.0)/2.0
    vw = (sqrt(1.0+4.0*bww)+1.0)/2.0
    dptildem1 = (params[keys[:π_bar]]^params[keys[:ap]])*(dpm1^(1.0-params[keys[:ap]]))
    dwtildem1 = (params[keys[:π_bar]]^params[keys[:aw]])*(dpm1^(1.0-params[keys[:aw]]))
    gzwage = params[keys[:gz]]*techshk^(1.0-params[keys[:bw]])
    dp = vp*dptildem1
    dw = vw*dwtildem1*gzwage
    muc = lam + (params[keys[:γ]]/params[keys[:gz]])*params[keys[:β]]*bc
    cc = params[keys[:γ]]*ccm1/(params[keys[:gz]]*techshk)+1.0/muc
    cquad_vi = bi/(qq*invshk)-(1.0-qq*invshk)/(params[keys[:ϕ_I]].value*qq*invshk)

    vi = 0.5*(1.0+sqrt(1.0+4.0*cquad_vi))

    inv = vi*invm1/techshk
    aayy = 1.0/gshk-(params[keys[:ϕ_p]]/2.0)*(vp-1.0)*(vp-1.0)
    rkss = params[keys[:gz]]/params[keys[:β]]-1.0+params[keys[:δ]]
    utilcost = (rkss/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(util-1.0))-1.0)
    gdp = (1.0/aayy)*( cc+inv + utilcost*(capm1/(params[keys[:gz]]*techshk)) )
    rw = rwm1*dw/(params[keys[:gz]]*techshk*dp)
    lab = gdp^(1.0/(1.0-params[keys[:α]]))*(util*capm1/(params[keys[:gz]]*techshk))^(params[keys[:α]]/(params[keys[:α]]-1.0))/ashk
    xhp = params[keys[:α]]*log(util) + (1-params[keys[:α]])*(log(lab)-log(labss)) # Changed llabss to log(m[:labss])

    lrss = log(params[keys[:gz]]*params[keys[:π_bar]]/params[keys[:β]])
    notr = exp(lrss+params[keys[:ρ_R]]*(log(notrm1)-lrss) + (1.0-params[keys[:ρ_R]])*(params[keys[:γ_π]]*log(dp/params[keys[:π_bar]]) + params[keys[:γ_g]]*log(gdp*techshk/gdpm1) + params[keys[:γ_x]]*xhp ) + rrshk)

    mc = rw*lab/((1.0-params[keys[:α]])*gdp)
    rentalk = (params[keys[:α]]/(1.0-params[keys[:α]]))*(rw*lab*params[keys[:gz]]*techshk/(util*capm1))
    cap = (1.0-params[keys[:δ]])*(capm1/(params[keys[:gz]]*techshk)) + invshk*inv*(1.0- (params[keys[:ϕ_I]]/2.0)*(vi-1.0)*(vi-1.0) )
    nomr = copy(notr) # copy as to no make it a pointer

    #all variables are in levels except for shocks
    #shocks are in log-level deviations from SS
    #(1) cap, (2) cc, (3) inv, (4) rw, (5) notr, (6) dp, (7) gdp, (8) xhp, (9) nomr, (10) lam, (11) qq,
    #(12) lab, (13) util, (14) mc, (15) rentalk, (16) muc, (17) vi, (18) vp, (19) vw, (20) dw, (21) bc, (22) bi,
    #(23) liqshk, (24) invshk, (25) techshk, (26) rrshk, (27) gshk, (28) ashk

    endogvar = vcat([cap, cc, inv, rw, notr, dp, gdp, xhp, nomr, lam, qq, lab, util, mc, rentalk, muc, vi, vp, vw, dw, bc, bi], currentshockvalues) #Concatenate two arrays

    return endogvar
end

"""
    decr(approx::SmolyakApproximation,endogvarm1::Vector{Float64},innovations::Vector{Float64},params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64}, endog_st, alphacoeff::Array{Float64,2})

The decision rule -- returns endogenous variables and shocks given lagged endogenous values and innovations.
...
# Arguments
- `m::GHLS`: GHLS model object, including the parameters, SmolyakApproximation object, and endogenou variables and shock values
- `endogvarm1::Vector{Float64}`: Lagged endogenous variables and shock values.
- `innovations::Vector{Float64}`: Innovations to the shocks.
- `alphacoeff::Arary{Float64, 2}`: Polynomial coefficients.
...
"""
function decr(nvars::Int, nexog::Int, nexogcont::Int, nexogshock::Int, nfunc::Int, ninter::Int, nmsv::Int, ngrid::Int, nshockgrid::Array{Int, 2}, shockbounds::Array{Float64, 2}, shockdistance::Array{Float64, 1}, interpolatemat::Array{Float64, 2}, slopeconmsv::Array{Float64, 1}, nindplus::Int, indplus::Array{Int, 1}, exoggrid::Array{Float64, 2},ns::Int, zlbswitch::Bool, endogvarm1::Vector{Float64},innovations::Vector{Float64},params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64}, labss::Float64, alphacoeff::Array{Float64,2},exogenous_shocks::OrderedCollections.OrderedDict{Symbol,Int64},endogenous_states::OrderedCollections.OrderedDict{Symbol,Int64})

    #Initilize Variables
    endogvar=Array{Float64}(undef,nvars+nexog)
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

    #shock parameters
    # sdevtech = params[24]
    # rhog = params[25]
    # sdevg = params[26]
    # rhoinv = params[27]
    # sdevinv = params[28]
    # rholiq = params[29]
    # sdevliq = params[30]
    # rhoint = params[31]
    # sdevint = params[32]
    # rhoa = params[33]
    # sdeva = params[34]

    #update shocks (all shocks in deviation from SS)
    ## Check the parameters replacing endogvarm1
    currentshockvalues[1] = params[keys[:ρ_η]]*endogvarm1[endogenous_states[:b_t]] + params[keys[:σ_η]]*innovations[1]
    currentshockvalues[2] = params[keys[:ρ_μ]]*endogvarm1[endogenous_states[:μ_t]] + params[keys[:σ_μ]]*innovations[2]
    currentshockvalues[3] = params[keys[:σ_Z]]*innovations[3]
    currentshockvalues[4] = params[keys[:ρ_int]]*endogvarm1[endogenous_states[:mon_t]] + params[keys[:σ_R]]*innovations[4]
    currentshockvalues[5] = params[keys[:ρ_g]]*endogvarm1[endogenous_states[:g_t]] + params[keys[:σ_g]]*innovations[5]
    currentshockvalues[6] = 0.0 #a shock?

    #find position of shocks for interpolation
    @simd for i in 1:nexogshock
        if (currentshockvalues[i] < shockbounds[i,1])   #extrapolation below
            shockindex[i] = 1
        elseif (currentshockvalues[i] > shockbounds[i,2])  #extrapolation above
            shockindex[i] = floor(Int64,(shockbounds[i,2]-shockbounds[i,1])/shockdistance[i] )
        else
            shockindex[i] = floor(Int64,1.0+(currentshockvalues[i]-shockbounds[i,1])/shockdistance[i])  #interpolation case
        end
    end

    #loop for interpolating between shocks
    shockindexall[1:nexogshock] = shockindex
    stateindex0 = exogposition(shockindexall,nshockgrid,nexog-nexogcont)
    shockindexall[1:nexogshock] = shockindex + interpolatemat[:,ninter]
    stateindex1 = exogposition(shockindexall,nshockgrid,nexog-nexogcont)
    funcmat = zeros(nfunc,ninter)
    weightvec = zeros(ninter)

    #Log the minimum state variables
    lmsv[1:nmsv] = log.(endogvarm1[1:nmsv])
    if (nexogcont > 0)
        lmsv[nmsv+1:nmsv+nexogcont] = currentshockvalues[nexog-nexogcont+1:nexog]
    end

    xx = msv2xx(lmsv,nmsvplus,slopeconmsv)

    polyvec = smolyakpoly(nmsvplus,ngrid,nindplus,indplus,xx)

    prod_sd = prod(shockdistance)

    for i in 1:ninter # @inbounds, stops the bounds check
        @fastmath @inbounds @simd for j in 1:nexogshock
            shockindexall[j] = shockindex[j] + interpolatemat[j,i]
            weighttemp[j]=(1-interpolatemat[j,i])*(currentshockvalues[j]-exoggrid[j,stateindex0]) + (interpolatemat[j,i])*(exoggrid[j,stateindex1]-currentshockvalues[j])
        end
        stateindex = exogposition(shockindexall,nshockgrid,nexog-nexogcont)
        stateindexplus = stateindex+ns
        @fastmath @inbounds @simd for ifunc in 1:nfunc
            funcmat[ifunc,i] = float_dot(view(alphacoeff,(ifunc-1)*ngrid+1,ifunc*ngrid,stateindex),polyvec)
            funcmatplus[ifunc,i] = float_dot(view(alphacoeff,(ifunc-1)*ngrid+1,ifunc*ngrid,stateindexplus),polyvec)
        end
        weightvec[ninter-i+1] = prod(weighttemp)/prod_sd
    end

    funcapp = dgemv(1.0,funcmat, weightvec)

    zlbintermediate = false  #start with evaluation of 1 poly case (omegapoly and 2nd funcapp irrelevant)
    omegapoly = 1.0 #SINCE ZLBINTERMEDIATE IS FALSE

    endogvar=intermediatedec(nvars,nexog,labss, endogvarm1,currentshockvalues, funcapp,omegapoly,funcapp,zlbintermediate, exogenous_shocks, params, keys)
    if ( (endogvar[endogenous_states[:rm_t]] < 1.0) & (zlbswitch == true) ) #zlb case
        zlbintermediate = true
        omegapoly = exp(omegaweight*log(endogvar[endogenous_states[:rm_t]])) #now omegapoly and funcapp_plus relevant
        funcapp_plus = dgemv(1.0, funcmatplus,  weightvec)
        endogvar = intermediatedec(nvars,nexog, labss, endogvarm1,currentshockvalues, funcapp,omegapoly,funcapp_plus,zlbintermediate, exogenous_shocks, params, keys)
        endogvar[9] = 1.0
    end

    return endogvar

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
function decrlin(endogvarm1::Vector{Float64},innovations::Vector{Float64},nvars::Int,nexog::Int,sigma::Array{Float64,2},pp::Array{Float64,2},steady_states::Array{AbstractParameter{Float64},1})


    # Initilize variables
    endogvar = Array{Float64}(undef,nvars+nexog)
    xxm1 = zeros(nvars+nexog)
    xx = zeros(nvars+nexog)
    exogpart = Array{Float64}(undef,nvars+nexog)
    endogsteady = [i.value for i in steady_states[1:nvars + nexog]]

    xxm1[1:nvars] = endogvarm1[1:nvars]-endogsteady[1:nvars]
    xxm1[nvars+1:nvars+nexog] = endogvarm1[nvars+1:nvars+nexog]

    xx = dgemv(1.0,pp, xxm1)
    exogpart = dgemv(1.0,sigma, innovations)
    xx = xx + exogpart

    endogvar[1:nvars] = xx[1:nvars]+endogsteady[1:nvars]
    endogvar[nvars+1:nvars+nexog] = xx[nvars+1:nvars+nexog]

    return endogvar
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
function decr_euler(nexogshock::Int, nfunc::Int, nexog::Int, nvars::Int, nexogcont::Int, nmsv::Int, xgrid::Array{Float64, 2}, slopeconxx::Array{Float64, 1}, exoggrid::Array{Float64, 2}, gridindex::Int64,shockpos::Int64, ngrid::Array{Int, 1}, nshockgrid::Array{Int, 2}, bbt::Array{Float64, 2}, statezlbinfo::Array{Int64, 1}, zlbswitch::Bool, nquad::Int, ghweights::Array{Float64, 1}, ghnodes::Array{Float64, 2}, shockbounds::Array{Float64, 2}, shockdistance::Array{Float64, 1}, interpolatemat::Array{Float64, 2}, slopeconmsv::Array{Float64, 1}, nindplus::Int, indplus::Array{Int ,1}, ns::Int, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64},alphacoeff::Array{Float64,2}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64})

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

    #keys = [i.key for i in params]
    #values = [i.value for i in params]

    currentshockvalues = zeros(nexog)
    polyapp = zeros(2*nfunc)
    endogvarm1 = zeros(nvars+nexog)
    exp_var = zeros(12)
    innovations = zeros(nexog)

    nmsvplus = nmsv+nexogcont
    if (nexogcont > 0)
        slopeconcont[1:nexogcont] = slopeconxx[nmsv+1:nmsvplus]
        slopeconcont[nexogcont+1:2*nexogcont] = slopeconxx[nmsvplus+nmsv+1:2*nmsvplus]
        xgridshock = xgrid[nmsv+1:nmsvplus,gridindex]
        currentshockvalues[nexog-nexogcont+1:nexog] = msv2xx(xgridshock,nexogcont,slopeconcont)
    end

    currentshockvalues[1:nexogshock] = exoggrid[1:nexogshock,shockpos]

    shockpospoly = shockpos + ns
    for ifunc in 1:nfunc
        polyapp[ifunc] = dot(alphacoeff[(ifunc-1)*ngrid+1:ifunc*ngrid,shockpos],bbt[:,gridindex])
        polyapp[nfunc+ifunc] = dot(alphacoeff[(ifunc-1)*ngrid+1:ifunc*ngrid,shockpospoly],bbt[:,gridindex])
    end

    xgridmsv = xgrid[1:nmsv,gridindex]
    slopeconxxmsv[1:nmsv] = slopeconxx[1:nmsv]
    slopeconxxmsv[nmsv+1:2*nmsv] = slopeconxx[nmsvplus+1:nmsvplus+nmsv]
    endogvarm1[1:nmsv] = exp.( msv2xx(xgridmsv,nmsv,slopeconxxmsv) )
    zlbintermediate = false  #force evaluation of 1 poly case at date t
    omegapoly = 1.0 #SET OMEGAPOLY TO 1 SINCE ZLBINTERMEDIATE IS FALSE

    endogvar=intermediatedec(nvars,nexog, labss, endogvarm1,currentshockvalues, polyapp[1:nfunc],omegapoly,polyapp[1:nfunc],zlbintermediate, params, keys)

    if ((zlbinfo != 0) & (zlbswitch == true))
        endogvarzlb= intermediatedec(nvars,nexog, labss, endogvarm1,currentshockvalues, polyapp[nfunc+1:2*nfunc], omegapoly,polyapp[nfunc+1:2*nfunc],zlbintermediate, params, keys)
        endogvarzlb[9] = 1.0
    end

    for ss in 1:nquad
        innovations[1:nexogshock] = ghnodes[:,ss]
        if (nexogcont > 0)
            innovations[nexog-nexogcont+1:nexog] =  ghnodes[nexogshock+1:nexogshock+nexogcont,ss]
        end

        endogvarp=decr(nvars, nexog, nexogcont, nexogshock, nfunc, nmsv, ngrid, nshockgrid, shockbounds, shockdistance, interpolatemat, slopeconmsv, nindplus, indplus, exoggrid, ns, zlbswitch,endogvar,innovations,params, keys, labss, alphacoeff, exogenous_shocks, endogenous_states) # NOT SURE WHAT TO DO WITH THIS

        techshkp = exp(endogvarp[25])
        invshkp = exp(endogvarp[24])
        ev[1] = endogvarp[10]/(endogvarp[6]*techshkp)

        utilcostp = (params[keys[:rkss]]/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(endogvarp[13]-1.0))-1.0) # a(u_t)
        ev[2] = (endogvarp[10]/techshkp)*( (endogvarp[15]*endogvarp[13])-utilcostp+(1.0-params[keys[:δ]])*endogvarp[11])
        ev[3] = endogvarp[10]*(endogvarp[18]-1.0)*endogvarp[18]*endogvarp[7]
        ev[4] = (endogvarp[19]-1.0)*endogvarp[19]
        ev[5] = endogvarp[16]/techshkp
        ev[6] = endogvarp[10]*endogvarp[11]*invshkp*(endogvarp[17]-1.0)*endogvarp[17]*endogvarp[17]

        if ((zlbinfo != 0) & (zlbswitch == true))
            endogvarzlbp=decr(nvars, nexog, nexogcont, nexogshock, nfunc, nmsv, ngrid, nshockgrid, shockbounds, shockdistance, interpolatemat, slopeconmsv, nindplus, indplus, exoggrid, ns, zlbswitch,endogvarzlb,innovations, params, keys, labss, alphacoeff, exogenous_shocks, endogenous_states, test)
            ev[7] = endogvarzlbp[10]/(endogvarzlbp[6]*techshkp)
            utilcostp = (params[keys[:rkss]]/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(endogvarzlbp[13]-1.0))-1.0)
            ev[8] = (endogvarzlbp[10]/techshkp)*( (endogvarzlbp[15]*endogvarzlbp[13])-utilcostp+(1-params[16])*endogvarzlbp[11] )
            ev[9] = endogvarzlbp[10]*(endogvarzlbp[18]-1.0)*endogvarzlbp[18]*endogvarzlbp[7]
            ev[10] = (endogvarzlbp[19]-1.0)*endogvarzlbp[19]
            ev[11] = endogvarzlbp[16]/techshkp
            ev[12] = endogvarzlbp[10]*endogvarzlbp[11]*invshkp*(endogvarzlbp[17]-1.0)*endogvarzlbp[17]*endogvarzlbp[17]
        else
            ev[7:12] = ev[1:6]
        end
        exp_var = exp_var + ghweights[ss]*ev
    end

    liqshk = exp(endogvar[23]) # Change to m.endogenous_states[:b_t]?
    invshk = exp(endogvar[24]) # Change to m.endogenous_states[:μ_t]?
    ep = params[keys[:ϵ_p]].scaledvalue # Is this constant elasticity of demand?

    exp_eul[1] =  (params[keys[:β]]/params[keys[:gz]])*liqshk*endogvar[9]*exp_var[1] # Change endogvar?
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
    polyappnew[7] = log( 1.0 + (1/params[keys[:σ_a]])*log(endogvar[15]/params[keys[:rkss]]) )

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
        polyappnew[14] = log( 1.0 + (1/params[keys[:σ_a]])*log(endogvarzlb[15]/params[keys[:rkss]]) )
    else
        exp_eul[7:12] = exp_eul[1:6]
        polyappnew[8:14] = polyappnew[1:7]
    end

    errsum = 0.0
    errmax = 0.0
    imax = 0

    for ifunc in 1:2*nfunc
        abserror[ifunc] = abs(polyappnew[ifunc]-polyapp[ifunc])
        errsum = errsum + abserror[ifunc]
        if (abserror[ifunc] > errmax)
            errmax = abserror[ifunc]
            imax = copy(ifunc)
        end
    end

    errsum = errsum/(2*nfunc)

    return polyappnew, errsum

end

"""
    finite_grid(n::Int64,rho::Float64,sigmaep::Float64)

Returns an array containing the values of shock at which model is solved.
...
# Arguments
- `n::Int64`: Number of shock realizations.
- `rho::Float64`: AR(1) coefficient.
- `sigmaep::Float64`: Standard deviation of innovation.
...
"""
function finite_grid(n::Int64,rho::Float64,sigmaep::Float64)

    #Initilize Variables
    shockgrid=zeros(n)
    maxgridstd = 3.0

    nu = sqrt(1.0/(1.0-rho^2))*maxgridstd*sigmaep
    shockgrid[n]  = nu
    shockgrid[1]  = -1*shockgrid[n]
    zstep = (shockgrid[n] - shockgrid[1]) / (n - 1)

    for i in 2:n-1
        shockgrid[i] = shockgrid[1] + zstep * (i - 1)
    end

    return zstep,shockgrid

end

"""
    get_shockdetails(approx::SmolyakApproximation, params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64}

Get the markov processes for the shocks.
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
function get_shockdetails(number_shock_values::Int, nshockgrid::Array{Int,2}, exogvarinfo::Array{Int64, 2}, nexogshock::Int, ns::Int, nexog::Int, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64})

    #Initilize Variables
    shockbounds = Array{Float64}(undef,nexogshock,2)
    shockdistance = Array{Float64}(undef,nexogshock)
    currentshockindex = Array{Float64}(undef,nexogshock)
    nshocksum_vec = Array{Float64}(undef,nexogshock)
    exoggrid = zeros(nexog-nexogcont,ns)

    rhovec = [params[keys[:ρ_η]].value,params[keys[:ρ_μ]].value,0.0,params[keys[:ρ_int]].value,params[keys[:ρ_g]].value, 0.0] # Unsure about rhoa
    sigmavec = [params[keys[:σ_η]].scaledvalue,params[keys[:σ_μ]].scaledvalue,params[keys[:σ_Z]].scaledvalue,params[keys[:σ_R]].scaledvalue,params[keys[:σ_g]].scaledvalue,0.0] # Unsure about sdeva

    nshocksum = 0
    shockvalues = zeros(number_shock_values)

    for i in 1:nexogshock
        xshock = Array{Float64}(undef,nshockgrid[i])
        shockdistance[i],xshock=finite_grid(nshockgrid[i],rhovec[i],sigmavec[i])
        shockbounds[i,1] = xshock[1]
        shockbounds[i,2] = xshock[nshockgrid[i]]
        shockvalues[nshocksum+1:nshocksum+nshockgrid[i]] = xshock
        nshocksum = nshocksum + nshockgrid[i]
    end

    for ss in 1:ns
        currentshockindex = exogvarinfo[:,ss]
        nshocksum_vec = zeros(Int64,nexogshock)
        nall = nshockgrid[1]
        shockpos = currentshockindex[1]
        exoggrid[1,ss] = shockvalues[currentshockindex[1]]
        for i in 2:nexogshock
            nshocksum_vec[i] = nshocksum_vec[i-1] + nshockgrid[i-1]
            shockpos = shockpos + nall*(currentshockindex[i]-1)
            nall = nall*nshockgrid[i]
            exoggrid[i,ss] = shockvalues[nshocksum_vec[i] + currentshockindex[i]]
        end
    end

    return exoggrid,shockbounds,shockdistance

end
