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
function intermediatedec!(endogvar::Vector{Float64},nvars::Int,nexog::Int,labss::Float64,endogvarm1::Vector{Float64},currentshockvalues::Vector{Float64},polyvar::Vector{Float64},omegapoly::Float64,polyvarplus::Vector{Float64},zlbintermediate::Bool,endogenous_states::OrderedCollections.OrderedDict{Symbol,Int64}, exogenous_shocks::OrderedCollections.OrderedDict{Symbol,Int64}, params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64})

    # Transform shocks from log-level deviations from steady state to levels
    invshk::Float64 = exp(currentshockvalues[exogenous_shocks[:μ_sh]])
    techshk::Float64 = exp(currentshockvalues[exogenous_shocks[:ztil_sh]])
    rrshk::Float64 = currentshockvalues[exogenous_shocks[:rm_sh]]
    gss::Float64 = 1.0/(1.0-params[keys[:shrgy]])
    gshk::Float64 = exp( log(gss) + currentshockvalues[exogenous_shocks[:g_sh]] )
    ashk::Float64 = exp(0.)

    # Update levels of endogenous variables
    ## The variables below are initially set to the polynomial approximations.
    ## Thus, they refer to the V_{l,j}(X_{t-1},τ_t)functions given in equations (2.14) - (2.20) in Technical Appendix of GHLS (2017)
    if (zlbintermediate == true)
        endogvar[endogenous_states[:λc]] = omegapoly*exp(polyvar[1]) + (1.0-omegapoly)*exp(polyvarplus[1]) #lam
        endogvar[endogenous_states[:qk_t]] = omegapoly*exp(polyvar[2]) + (1.0-omegapoly)*exp(polyvarplus[2]) #qq
        bp = omegapoly*polyvar[3] + (1.0-omegapoly)*polyvarplus[3]
        bww = omegapoly*polyvar[4] + (1.0-omegapoly)*polyvarplus[4]
        endogvar[endogenous_states[:bc_t]] = omegapoly*exp(polyvar[5]) + (1.0-omegapoly)*exp(polyvarplus[5]) #bc
        endogvar[endogenous_states[:bi_t]] = omegapoly*polyvar[6] + (1.0-omegapoly)*polyvarplus[6] #bi
        endogvar[endogenous_states[:u_t]] = omegapoly*exp(polyvar[7]) + (1.0-omegapoly)*exp(polyvarplus[7]) #util
    else
        endogvar[endogenous_states[:λc]] = exp(polyvar[1]) #lam
        endogvar[endogenous_states[:qk_t]] = exp(polyvar[2]) #qq
        bp::Float64 = polyvar[3]
        bww::Float64 = polyvar[4]
        endogvar[endogenous_states[:bc_t]] = exp(polyvar[5]) #bc
        endogvar[endogenous_states[:bi_t]] = polyvar[6] #bi
        endogvar[endogenous_states[:u_t]] = exp(polyvar[7]) #util
    end

    endogvar[endogenous_states[:Vp_t]] = (sqrt(1.0+4.0*bp)+1.0)/2.0 #vp - see (2.5) in GHLS (2017) Technical Appendix (TA). vp = \frac{π(X_{t-1},τ_t)}{\tilde{π}_{t-1}/2} where (X_{t-1},τ_t) is the minimum state vector. (see (2.1) and (2.2) in TA)
    endogvar[endogenous_states[:Vw_t]] = (sqrt(1.0+4.0*bww)+1.0)/2.0 #vw = π_w(X_{t-1},τ_t)/\tilde{π}_{w,t} (see (2.6) in TA)
    dptildem1::Float64 = (params[keys[:π_bar]]^params[keys[:ap]])*(endogvarm1[endogenous_states[:π_t]]^(1.0-params[keys[:ap]])) # Indexation term for price changes - (1.4) of TA --> Expectation of future inflation
    dwtildem1::Float64 = (params[keys[:π_bar]]^params[keys[:aw]])*(endogvarm1[endogenous_states[:π_t]]^(1.0-params[keys[:aw]])) # Component of indexation term for wage changes - (1.10) in TA - see dw below
    gzwage::Float64 = params[keys[:gz]]*techshk^(1.0-params[keys[:aw]]) # Component of indexation term for wage changes - (1.10) in TA. Note the indexation is dwtildem1*gzwage
    ##Note that aw should be bw but in steady state they are equal, might be better style to pass bw though

    endogvar[endogenous_states[:π_t]] = endogvar[endogenous_states[:Vp_t]]*dptildem1 #dp = π(X_{t-1},τ_t) from (2.5) of TA
    endogvar[endogenous_states[:π_w]] = endogvar[endogenous_states[:Vw_t]]*dwtildem1*gzwage #dw = π_w (X_{t-1},τ_t) from (2.6) of TA
    endogvar[endogenous_states[:muc_t]] = endogvar[endogenous_states[:λc]] + (params[keys[:γ]]/params[keys[:gz]])*params[keys[:β]]*endogvar[endogenous_states[:bc_t]] #muc (marginal utility of consumption), see last term in (2.8) of TA or (1.27) of TA
    endogvar[endogenous_states[:c_t]] = params[keys[:γ]]*endogvarm1[endogenous_states[:c_t]]/(params[keys[:gz]]*techshk)+1.0/endogvar[endogenous_states[:muc_t]] #cc = c (X_{t-1},τ_t) (see (2.8) in TA)
    cquad_vi::Float64 = endogvar[endogenous_states[:bi_t]]/(endogvar[endogenous_states[:qk_t]]*invshk)-(1.0-endogvar[endogenous_states[:qk_t]]*invshk)/(params[keys[:ϕ_I]].value*endogvar[endogenous_states[:qk_t]]*invshk)# This is is term in the square root minus 1 divided by 4 of (2.9) of TA

    endogvar[endogenous_states[:Vi_t]] = 0.5*(1.0+sqrt(1.0+4.0*cquad_vi)) #vi = i(X_{t-1},τ_t)/i_{t-1} from (2.9) in TA

    endogvar[endogenous_states[:i_t]] = endogvar[endogenous_states[:Vi_t]]*endogvarm1[endogenous_states[:i_t]]/techshk #inv = i(X_{t-1},τ_t) from (2.9) in TA
    aayy::Float64 = 1.0/gshk-(params[keys[:ϕ_p]]/2.0)*(endogvar[endogenous_states[:Vp_t]]-1.0)*(endogvar[endogenous_states[:Vp_t]]-1.0) #A_{y,t} from (1.33) in TA
    rkss::Float64 = params[keys[:gz]]/params[keys[:β]]-1.0+params[keys[:δ]] #r^k from (1.46) in TA
    utilcost::Float64 = (rkss/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(endogvar[endogenous_states[:u_t]]-1.0))-1.0) #Utilization Cost = a(u_t) from (1.14) in TA
    endogvar[endogenous_states[:y_t]] = (1.0/aayy)*( endogvar[endogenous_states[:c_t]]+endogvar[endogenous_states[:i_t]] + utilcost*(endogvarm1[endogenous_states[:k_t]]/(params[keys[:gz]]*techshk)) ) #gdp = y_t - see (1.32) in TA
    endogvar[endogenous_states[:w_t]] = endogvarm1[endogenous_states[:w_t]]*endogvar[endogenous_states[:π_w]]/(params[keys[:gz]]*techshk*endogvar[endogenous_states[:π_t]]) #rw = w_t (1.34) in TA
    endogvar[endogenous_states[:L_t]] = endogvar[endogenous_states[:y_t]]^(1.0/(1.0-params[keys[:α]]))*(endogvar[endogenous_states[:u_t]]*endogvarm1[endogenous_states[:k_t]]/(params[keys[:gz]]*techshk))^(params[keys[:α]]/(params[keys[:α]]-1.0))/ashk #lab = N_t (1.35) in TA
    endogvar[endogenous_states[:x_t]] = params[keys[:α]]*log(endogvar[endogenous_states[:u_t]]) + (1-params[keys[:α]])*(log(endogvar[endogenous_states[:L_t]])-log(labss)) # Changed llabss to log(m[:labss]), this is xhp = x_t^g from (1.17) in TA. This is same as loged version of the formula at the bottom of pg. 1978 in the main paper for GHLS (2017).

    lrss::Float64 = log(params[keys[:gz]]*params[keys[:π_bar]]/params[keys[:β]]) # Log of steady state nominal interest rate - see (1.43) in TA
    endogvar[endogenous_states[:rm_t]] = exp(lrss+params[keys[:ρ_R]]*(log(endogvarm1[endogenous_states[:rm_t]])-lrss) + (1.0-params[keys[:ρ_R]])*(params[keys[:γ_π]]*log(endogvar[endogenous_states[:π_t]]/params[keys[:π_bar]]) + params[keys[:γ_g]]*log(endogvar[endogenous_states[:y_t]]*techshk/endogvarm1[endogenous_states[:y_t]]) + params[keys[:γ_x]]*endogvar[endogenous_states[:x_t]] ) + rrshk) #Notional interest rate (Interest rate without a zero lower bound) - see (1.39) in TA

    endogvar[endogenous_states[:mc_t]] = endogvar[endogenous_states[:w_t]]*endogvar[endogenous_states[:L_t]]/((1.0-params[keys[:α]])*endogvar[endogenous_states[:y-t]]) #Marginal Cost: mc_t from (1.36) in TA
    endogvar[endogenous_states[:rk_t]] = (params[keys[:α]]/(1.0-params[keys[:α]]))*(endogvar[endogenous_states[:w_t]]*endogvar[endogenous_states[:L_t]]*params[keys[:gz]]*techshk/(endogvar[endogenous_states[:u_t]]*endogvarm1[endogenous_states[:k_t]])) #rentalk: r_t^k from (1.37) in TA
    endogvar[endogenous_states[:k_t]] = (1.0-params[keys[:δ]])*(endogvarm1[endogenous_states[:k_t]]/(params[keys[:gz]]*techshk)) + invshk*endogvar[endogenous_states[:i_t]]*(1.0- (params[keys[:ϕ_I]]/2.0)*(endogvar[endogenous_states[:Vi_t]]-1.0)*(endogvar[endogenous_states[:Vi_t]]-1.0) ) #cap: \bar{k}_{t+1} from (1.38) in TA
    endogvar[endogenous_states[:R_t]] = copy(endogvar[endogenous_states[:rm_t]]) # copy as to no make it a pointer, this is nomr (Nominal Interest Rate) - see (1.18) in TA

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
function decr!(endogvar::Vector{Float64}, approx::SmolyakApproximation, endogvarm1::Vector{Float64},innovations::Vector{Float64},params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64}, labss::Float64, alphacoeff::Array{Float64,2},exogenous_shocks::OrderedCollections.OrderedDict{Symbol,Int64},endogenous_states::OrderedCollections.OrderedDict{Symbol,Int64})

    #Initialize Variables
    shockindexall=ones(Int64,approx.nexog)
    shockindex=Array{Int64}(undef,approx.nexogshock)
    shockindex_inter=Array{Int64}(undef,approx.nexogshock)
    lmsv=Array{Float64}(undef,approx.nmsv)
    currentshockvalues=Array{Float64}(undef,approx.nexog)
    funcmatplus=Array{Float64}(undef,approx.nfunc,approx.ninter)
    weighttemp=Array{Float64}(undef,approx.nexogshock)
    funcapp=Array{Float64}(undef,approx.nfunc)
    funcapp_plus=Array{Float64}(undef,approx.nfunc)
    xx=Array{Float64}(undef,approx.nmsv)
    polyvec=Array{Float64}(undef,approx.ngrid)

    omegaweight = 100000.0

    #update shocks (all shocks in deviation from SS)
    currentshockvalues[1]::Float64 = params[keys[:ρ_η]]*endogvarm1[endogenous_states[:b_t]] + params[keys[:σ_η]]*innovations[1]
    currentshockvalues[2]::Float64 = params[keys[:ρ_μ]]*endogvarm1[endogenous_states[:μ_t]] + params[keys[:σ_μ]]*innovations[2]
    currentshockvalues[3]::Float64 = params[keys[:σ_Z]]*innovations[3]
    currentshockvalues[4]::Float64 = params[keys[:ρ_int]]*endogvarm1[endogenous_states[:mon_t]] + params[keys[:σ_R]]*innovations[4]
    currentshockvalues[5]::Float64 = params[keys[:ρ_g]]*endogvarm1[endogenous_states[:g_t]] + params[keys[:σ_g]]*innovations[5]
    currentshockvalues[6] = 0.0 #a shock

    # Finds shock index associated with current shock value (rounded down)
    @simd for i in 1:approx.nexogshock
        if (currentshockvalues[i] < approx.shockbounds[i,1])
            shockindex[i] = 1
        elseif (currentshockvalues[i] > approx.shockbounds[i,2])
            shockindex[i] = floor(Int64,(approx.shockbounds[i,2]-approx.shockbounds[i,1])/approx.shockdistance[i] )
        else
            shockindex[i] = floor(Int64,1.0+(currentshockvalues[i]-approx.shockbounds[i,1])/approx.shockdistance[i])
        end
    end

    # Lowest possible state (all indices rounded down)
    shockindexall[1:approx.nexogshock] = shockindex
    stateindex0 = exogposition(shockindexall,approx.nshockgrid,approx.nexog)

    # Highest possible state (all indices rounded up)
    shockindexall[1:approx.nexogshock] = shockindex + approx.interpolatemat[:,approx.ninter]
    stateindex1 = exogposition(shockindexall,approx.nshockgrid,approx.nexog)

    # One possible function for each possible interpolation (i.e. every combination of rounding up/down)
    funcmat = Array{Float64}(undef,approx.nfunc,approx.ninter)
    weightvec = Array{Float64}(undef,approx.ninter)

    #Log the minimum state variables
    lmsv[1:approx.nmsv] = log.(endogvarm1[1:approx.nmsv])

    # Convert endogenous variables to [-1, 1] domain
    xx = msv2xx(lmsv,approx.nmsv,approx.slopeconmsv)

    # Get polynomials evaluated at these values  (this is T(ϕ(X_{t-1})))
    polyvec = smolyakpoly(approx.nmsv,approx.ngrid,approx.nindplus,approx.indplus,xx)

    prod_sd = prod(approx.shockdistance)

    @inbounds @simd for i in 1:approx.ninter
        @inbounds @simd for j in 1:approx.nexogshock
            shockindexall[j] = shockindex[j] + approx.interpolatemat[j,i]

            # Shock grid points that are closer to actual shock values result in lower weights
            # Can we not precalculate?
            weighttemp[j]=(1-approx.interpolatemat[j,i])*(currentshockvalues[j]-approx.exoggrid[j,stateindex0]) + (approx.interpolatemat[j,i])*(approx.exoggrid[j,stateindex1]-currentshockvalues[j])
        end
        stateindex = exogposition(shockindexall,approx.nshockgrid,approx.nexog)
        stateindexplus = stateindex+approx.ns
        @inbounds @simd for ifunc in 1:approx.nfunc
            @views funcmat[ifunc,i] = BLAS.dot(approx.ngrid, alphacoeff[(ifunc-1)*approx.ngrid+1:ifunc*approx.ngrid, stateindex], 1, polyvec, 1)
            @views funcmatplus[ifunc,i] = BLAS.dot(approx.ngrid, alphacoeff[(ifunc-1)*approx.ngrid+1:ifunc*approx.ngrid, stateindexplus],1, polyvec, 1)
        end

        # Give weight to inverse interpolation, so that in the end interpolations that are closer to actual shocks are given high weights
        # Note prod_sd is the most prod(weighttemp) can be
        weightvec[approx.ninter+1-i] = prod(weighttemp)/prod_sd
    end

    #In-place version of funcapp = funcmat*weightvec
    mul!(funcapp, funcmat, weightvec)

    zlbintermediate = false  #start with evaluation of 1 poly case (omegapoly and 2nd funcapp irrelevant)
    omegapoly = 1.0 #SINCE ZLBINTERMEDIATE IS FALSE

    #endogvar = Array{Float64}(undef, nvars+nexog)
    intermediatedec!(endogvar,approx.nvars,approx.nexog,labss, endogvarm1,currentshockvalues, funcapp,omegapoly,funcapp,zlbintermediate, endogenous_states, exogenous_shocks, params, keys)
    if ( (endogvar[endogenous_states[:rm_t]] < 1.0) & (approx.zlbswitch == true) ) #zlb case
        zlbintermediate = true
        omegapoly = exp(omegaweight*log(endogvar[endogenous_states[:rm_t]])) #now omegapoly and funcapp_plus relevant
        mul!(funcapp_plus, funcmatplus,  weightvec)
        intermediatedec!(endogvar,approx.nvars,approx.nexog, labss, endogvarm1,currentshockvalues, funcapp,omegapoly,funcapp_plus,zlbintermediate, endogenous_states, exogenous_shocks, params, keys)
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
function decr_euler(rkss::Float64, approx::SmolyakApproximation, gridindex::Int64,shockpos::Int64, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64},alphacoeff::Array{Float64,2}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64})

    #Initilize Variables
    zlbinfo  = approx.statezlbinfo[shockpos]
    polyappnew = Array{Float64}(undef,2*approx.nfunc)
    endogvar = Array{Float64}(undef,approx.nvars+approx.nexog)
    endogvarzlb = Array{Float64}(undef,approx.nvars+approx.nexog)
    endogvarp = Array{Float64}(undef,approx.nvars+approx.nexog)
    endogvarzlbp = Array{Float64}(undef,approx.nvars+approx.nexog)
    slopeconxxmsv = Array{Float64}(undef,2*approx.nmsv)
    xgridmsv = Array{Float64}(undef,approx.nmsv)
    abserror = Array{Float64}(undef,2*approx.nfunc)
    ev = Array{Float64}(undef,12)
    exp_eul = Array{Float64}(undef,12)

    currentshockvalues = Array{Float64}(undef,approx.nexog)
    polyapp = Array{Float64}(undef,2*approx.nfunc)
    endogvarm1 = Array{Float64}(undef,approx.nvars+approx.nexog)
    exp_var = zeros(12)
    innovations = Array{Float64}(undef,approx.nexog)

    # Gets shock values for given position on shock grid
    currentshockvalues[1:approx.nexogshock] = approx.exoggrid[1:approx.nexogshock,shockpos]

    # Calculates T(ϕ(X_{t-1}))α_{l,j,k} for each function
    shockpospoly = shockpos + approx.ns
    for ifunc in 1:approx.nfunc
        @views polyapp[ifunc] = BLAS.dot(approx.ngrid, alphacoeff[(ifunc-1)*approx.ngrid+1:ifunc*approx.ngrid,shockpos],1,approx.bbt[:,gridindex], 1)
        @views polyapp[approx.nfunc+ifunc] = BLAS.dot(approx.ngrid, alphacoeff[(ifunc-1)*approx.ngrid+1:ifunc*approx.ngrid,shockpospoly],1,approx.bbt[:,gridindex], 1)
    end

    # Get endogenous variable values for that grid point and convert back from [-1, 1] domain
    xgridmsv = approx.xgrid[1:approx.nmsv,gridindex]
    slopeconxxmsv[1:approx.nmsv] = approx.slopeconxx[1:approx.nmsv]
    slopeconxxmsv[approx.nmsv+1:2*approx.nmsv] = approx.slopeconxx[approx.nmsv+1:approx.nmsv+approx.nmsv]
    endogvarm1[1:approx.nmsv] = exp.( msv2xx(xgridmsv,approx.nmsv,slopeconxxmsv) )

    zlbintermediate = false  #force evaluation of 1 poly case at date t
    omegapoly = 1.0 #SET OMEGAPOLY TO 1 SINCE ZLBINTERMEDIATE IS FALSE

    # Updates given polynomial approximations
    intermediatedec!(endogvar,approx.nvars,approx.nexog, labss, endogvarm1,currentshockvalues, polyapp[1:approx.nfunc],omegapoly,polyapp[1:approx.nfunc],zlbintermediate, endogenous_states, exogenous_shocks, params, keys)

    if ((zlbinfo != 0) & (approx.zlbswitch == true))
        intermediatedec!(endogvar, approx.nvars,approx.nexog, labss, endogvarm1,currentshockvalues, polyapp[approx.nfunc+1:2*approx.nfunc], omegapoly,polyapp[approx.nfunc+1:2*approx.nfunc],zlbintermediate, endogenous_states, exogenous_shocks, params, keys)
        endogvarzlb[9] = 1.0
    end

    # Calculates conditional expectations
    for ss in 1:approx.nquad
        innovations[1:approx.nexogshock] = approx.ghnodes[:,ss]

        decr!(endogvarp,approx,endogvar,innovations,params, keys, labss, alphacoeff, exogenous_shocks, endogenous_states)

        techshkp = exp(endogvarp[endogenous_states[:ztil_t]])
        invshkp = exp(endogvarp[endogenous_states[:μ_t]])
        ## Note below that when multiple equations are cited, they are both the same expression, just in different places
        ev[1] = endogvarp[endogenous_states[:λc]]/(endogvarp[endogenous_states[:π_t]]*techshkp) # This is what's inside the expectation in from (1.26) in Technical Appendix of GHLS (2017), for V_{λ,t}, same as expectation in (2.14)

        utilcostp = (rkss/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(endogvarp[13]-1.0))-1.0) # a(u_t) - see equation (5) on pg. 1977 of GHLS (2017), same as (1.14) in Technical Appendix (TA)
        ev[2] = (endogvarp[endogenous_states[:λc]]/techshkp)*( (endogvarp[endogenous_states[:rk_t]]*endogvarp[endogenous_states[:u_t]])-utilcostp+(1.0-params[keys[:δ]])*endogvarp[endogenous_states[:qk_t]]) # Expectation in (1.29) of TA, for V_{q,t} or (2.16) - same expectation
        ev[3] = endogvarp[endogenous_states[:λc]]*(endogvarp[endogenous_states[:Vp_t]]-1.0)*endogvarp[endogenous_states[:Vp_t]]*endogvarp[endogenous_states[:y_t]] # Expectation in (2.18) and (2.3) of TA, for V_{π,j} and V_{π,t}
        ev[4] = (endogvarp[endogenous_states[:Vw_t]]-1.0)*endogvarp[endogenous_states[:Vw_t]] # Expectation in (2.19) and(2.4) of TA, for V_{w,j} and V_{w,t}
        ev[5] = endogvarp[endogenous_states[:muc_t]]/techshkp # Expectation in (2.15) of TA, for V_{c,j} or (1.28) of TA
        ev[6] = endogvarp[endogenous_states[:λc]]*endogvarp[endogenous_states[:qk_t]]*invshkp*(endogvarp[endogenous_states[:Vi_t]]-1.0)*endogvarp[endogenous_states[:Vi_t]]*endogvarp[endogenous_states[:Vi_t]] # Expectation in (2.17) of TA, for V_{i,j} or (1.31) of TA

        # Equation sources are same as above, just endogvarzlbp is used instead of endogvarp
        if ((zlbinfo != 0) & (approx.zlbswitch == true))
            decr!(endogvarzlbp,approx,endogvarzlb,innovations, params, keys, labss, alphacoeff, exogenous_shocks, endogenous_states)
            ev[7] = endogvarzlbp[endogenous_states[:λc]]/(endogvarzlbp[endogenous_states[:π_t]]*techshkp)
            utilcostp = (rkss/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(endogvarzlbp[endogenous_states[:u_t]]-1.0))-1.0)
            ev[8] = (endogvarzlbp[endogenous_states[:λc]]/techshkp)*( (endogvarzlbp[endogenous_states[:rk_t]]*endogvarzlbp[endogenous_states[:u_t]])-utilcostp+(1-params[keys[:δ]])*endogvarzlbp[endogenous_states[:qk_t]] )
            ev[9] = endogvarzlbp[endogenous_states[:λc]]*(endogvarzlbp[endogenous_states[:Vp_t]]-1.0)*endogvarzlbp[endogenous_states[:Vp_t]]*endogvarzlbp[endogenous_states[:y_t]]
            ev[10] = (endogvarzlbp[endogenous_states[:Vw_t]]-1.0)*endogvarzlbp[endogenous_states[:Vw_t]]
            ev[11] = endogvarzlbp[endogenous_states[:muc_t]]/techshkp
            ev[12] = endogvarzlbp[endogenous_states[:λc]]*endogvarzlbp[endogenous_states[:qk_t]]*invshkp*(endogvarzlbp[endogenous_states[:Vi_t]]-1.0)*endogvarzlbp[endogenous_states[:Vi_t]]*endogvarzlbp[endogenous_states[:Vi_t]]
        else
            ev[7:12] = ev[1:6]
        end

        # Approximate expectation by taking weighted sum over integration nodes
        exp_var += approx.ghweights[ss]*ev
    end

    liqshk = exp(endogvar[endogenous_states[:b_t]])
    invshk = exp(endogvar[endogenous_states[:μ_t]])
    ep = params[keys[:ϵ_p]].scaledvalue # Is this constant elasticity of demand?

    exp_eul[1] =  (params[keys[:β]]/params[keys[:gz]])*liqshk*endogvar[endogenous_states[:R_t]]*exp_var[1] # λ_t from equation (1.26) in TA
    exp_eul[2] =  (params[keys[:β]]/params[keys[:gz]])*exp_var[2]/endogvar[endogenous_states[:λc]] # q_t from equation (1.29) in TA
    exp_eul[3] =  params[keys[:β]]*exp_var[3]/(endogvar[endogenous_states[:λc]]*endogvar[endogenous_states[:y_t]])+(params[keys[:ϵ_p]]/params[keys[:ϕ_p]])*(endogvar[endogenous_states[:mc_t]]-(params[keys[:ϵ_p]]-1.0)/params[keys[:ϵ_p]]) #V_{π_t} from (2.3) in TA
    exp_eul[4] = params[keys[:β]]*exp_var[4]+(params[keys[:ϵ_w]]/params[keys[:ϕ_w]])*endogvar[endogenous_states[:λc]]*endogvar[endogenous_states[:L_t]]*
           ( (params[keys[:ψ_L]]*endogvar[endogenous_states[:L_t]]^params[keys[:σ_L]]/endogvar[endogenous_states[:λc]])-((params[keys[:ϵ_w]]-1.0)/params[keys[:ϵ_w]])*endogvar[endogenous_states[:w_t]] ) # V_{w,t} from (2.4) in TA
    exp_eul[5] = exp_var[5] # V_{c,t} from (1.28) in TA
    exp_eul[6] = (params[keys[:β]]/params[keys[:gz]])*exp_var[6]/endogvar[endogenous_states[:λc]] - 0.5*endogvar[endogenous_states[:qk_t]]*invshk*(endogvar[endogenous_states[:Vi_t]]-1.0)*(endogvar[endogenous_states[:Vi_t]]-1.0) # V_{i,t} from (1.31) in TA (This not being multiplied by m[:ϕ_I]  since in cquad_vi of intermediatedec! we don't divide V_{i,t} by m[:ϕ_I])

    polyappnew[1] = log(exp_eul[1])
    polyappnew[2] = log(exp_eul[2])
    polyappnew[3] = exp_eul[3]
    polyappnew[4] = exp_eul[4]
    polyappnew[5] = log(exp_eul[5])
    polyappnew[6] = exp_eul[6]
    polyappnew[7] = log( 1.0 + (1/params[keys[:σ_a]])*log(endogvar[endogenous_states[:rk_t]]/rkss) ) # V_{u,t} from (2.20)

    # Below equations are the same as above, just for the zero lower bound case.
    if ((zlbinfo != 0) & (approx.zlbswitch == true))
        exp_eul[7] =  (params[keys[:β]]/params[keys[:gz]])*liqshk*endogvarzlb[endogenous_states[:R_t]]*exp_var[7]
        exp_eul[8] = (params[keys[:β]]/params[keys[:gz]])*exp_var[8]/endogvarzlb[endogenous_states[:λc]]
        exp_eul[9] =  params[keys[:β]]*exp_var[9]/(endogvarzlb[endogenous_states[:λc]]*endogvarzlb[endogenous_states[:y_t]])+(params[keys[:ϵ_p]]/params[keys[:ϕ_p]])*( endogvarzlb[endogenous_states[:mc_t]]-(params[keys[:ϵ_p]]-1.0)/params[keys[:ϵ_p]])
        exp_eul[10] = params[keys[:β]]*exp_var[10]+(params[keys[:ϵ_w]]/params[keys[:ϕ_w]])*endogvarzlb[endogenous_states[:λc]]*endogvarzlb[endogenous_states[:L_t]]*
           ( (params[keys[:ψ_L]]*endogvarzlb[endogenous_states[:L_t]]^params[keys[:σ_L]]/endogvarzlb[endogenous_states[:λc]])-((params[keys[:ϵ_w]]-1.0)/params[keys[:ϵ_w]])*endogvarzlb[endogenous_states[:w_t]])
        exp_eul[11] = exp_var[11]
        exp_eul[12] = (params[keys[:β]]/params[keys[:gz]])*exp_var[12]/endogvarzlb[endogenous_states[:λc]] - 0.5*endogvarzlb[endogenous_states[:qk_t]]*invshk*(endogvarzlb[endogenous_states[:Vi_t]]-1.0)*(endogvarzlb[endogenous_states[:Vi_t]]-1.0)

        polyappnew[8] = log(exp_eul[7])
        polyappnew[9] = log(exp_eul[8])
        polyappnew[10] = exp_eul[9]
        polyappnew[11] = exp_eul[10]
        polyappnew[12] = log(exp_eul[11])
        polyappnew[13] = exp_eul[12]
        polyappnew[14] = log( 1.0 + (1/params[keys[:σ_a]])*log(endogvarzlb[endogenous_states[:rk_t]]/rkss) )
    else
        exp_eul[7:12] = exp_eul[1:6]
        polyappnew[8:14] = polyappnew[1:7]
    end

    # Gets average error across all polynomials
    erravg = 0.0
    for ifunc in 1:2*approx.nfunc
        abserror[ifunc] = abs(polyappnew[ifunc]-polyapp[ifunc])
        erravg += abserror[ifunc]
    end

    erravg /= (2*approx.nfunc)

    return polyappnew, erravg

end

"""
    finite_grid!(shockgrid::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}, shockdistance::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}, shockbounds::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}, n::Int64,rho::Float64,sigmaep::Float64)

For a given shock, this populates the shockgrid array with the possible values of the shock, spanning -nu to nu and evenly spaced. It also records the shockdistance (distance between shock values) and shockbounds (min and max shock values).
...
# Arguments
- `shockgrid::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}: Array containing possible values of shock
- `shockdistance::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}: Distance between values of shocks on grid
- `shockbounds::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}: Minimum and maximum values of shock
- `n::Int`: Number of shock realizations.
- `rho::Float64`: AR(1) coefficient.
- `sigmaep::Float64`: Standard deviation of innovation.
...
"""
function finite_grid!(shockgrid::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}, shockdistance::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}, shockbounds::SubArray{Float64, 1, Array{Float64,2}, Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true}, n::Int,rho::Float64,sigmaep::Float64)

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
    shockdistance[1] = @fastmath (shockgrid[n] - shockgrid[1]) / (n - 1)

    @inbounds @simd for i in 2:n-1
        shockgrid[i] = @fastmath shockgrid[1] + shockdistance[1] * (i - 1)
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
        - `ns::Int`: Number of exogenous states.
        - `number_shock_values::Int`: Number of shock values.
        - `nshockgrid::Vector{Float64}`: Array indexing number of realizations for each shock (length nexog).
        - `exogvarinfo`
...
"""
function get_shockdetails(number_shock_values::Int, nshockgrid::Array{Int,1}, exogvarinfo::Array{Int64, 2}, nexogshock::Int, ns::Int, nexog::Int, params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64})

    #Initilize Variables
    shockbounds = Array{Float64}(undef,nexogshock,2)
    shockdistance = Array{Float64}(undef,nexogshock)
    currentshockindex = Array{Float64}(undef,nexogshock)
    nshocksum_vec = Array{Float64}(undef,nexogshock)
    shockvalues = Array{Float64}(undef,number_shock_values)
    exoggrid = zeros(nexog,ns) #Not undef since sometimes not every row is initialized (when nexogshock < nexog)

    rhovec = [params[keys[:ρ_η]].value,params[keys[:ρ_μ]].value,0.0,params[keys[:ρ_int]].value,params[keys[:ρ_g]].value, 0.0]
    sigmavec = [params[keys[:σ_η]].scaledvalue,params[keys[:σ_μ]].scaledvalue,params[keys[:σ_Z]].scaledvalue,params[keys[:σ_R]].scaledvalue,params[keys[:σ_g]].scaledvalue,0.0] #we should remove ashk it is not used

    # Tracks the total number of shock values that have been inputted into shockvalues array
    numshockvalues = 0

    # For each shock populate the shockvalues array with the values it can take on
    @simd for i in 1:nexogshock

        # Use views so that array slices can be mutated
        @views finite_grid!(shockvalues[numshockvalues+1:numshockvalues+nshockgrid[i]], shockdistance[i:i], shockbounds[i, :], nshockgrid[i],rhovec[i],sigmavec[i])
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
