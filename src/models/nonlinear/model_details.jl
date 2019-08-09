"""
    msv2xx(msv::Vector{Float64}, nmsv::Int, slopeconmsv::Vector{Float64})

# Arguments:
- `msv::Vector{Float64}`: Minimum state endogenous variables.
- `nmsv::Int`: Number of minimum state endogenous state variables.
- `slopeconmsv::Vector{Float64}`: Slope and constants that depend on the upper and lower bounds of minimum state variables.

# Description:
Returns endogenous state variables in [-1,1] domain or the inverse function depending on what is input as slopeconmsv.
"""
function msv2xx(msv::Vector{Float64}, nmsv::Int, slopeconmsv::Vector{Float64})
    return slopeconmsv[1:nmsv] .* msv + slopeconmsv[nmsv+1:2*nmsv]
end

"""
    exogstate(exogvec::Vector{Int64}, nrvec::Vector{Int64}, nexogvars::Int64)

# Arguments:
- `exogvec::Vector{Int64}`: Array with index for each shock value.
- `nshockgrid::Vector{Int64}`: Number of realizations for each shock.
- `nexogvars::Int64`: Number of exogenous variables.

# Description:
Computes exogenous state given values of each shock by inverting the calculation performed in gen_exoggrid_indices. Hard-wired for six or less exogenous variables for speed.
"""
function exogstate(exogvec::Vector{Int64}, nshockgrid::Array{Int64, 1}, nexogvars::Int64)
    if (nexogvars == 6)
        return nshockgrid[6]*nshockgrid[5]*nshockgrid[4]*nshockgrid[3]*nshockgrid[2]*(exogvec[1]-1) + nshockgrid[6]*nshockgrid[5]*nshockgrid[4]*nshockgrid[3]*(exogvec[2]-1) + nshockgrid[6]*nshockgrid[5]*nshockgrid[4]*(exogvec[3]-1) + nshockgrid[6]*nshockgrid[5]*(exogvec[4]-1) + nshockgrid[6]*(exogvec[5]-1) + exogvec[6]

    elseif (nexogvars == 5)
        return nshockgrid[5]*nshockgrid[4]*nshockgrid[3]*nshockgrid[2]*(exogvec[1]-1) + nshockgrid[5]*nshockgrid[4]*nshockgrid[3]*(exogvec[2]-1) + nshockgrid[5]*nshockgrid[4]*(exogvec[3]-1) + nshockgrid[5]*(exogvec[4]-1) + exogvec[5]

    elseif (nexogvars == 4)
        return nshockgrid[4]*nshockgrid[3]*nshockgrid[2]*(exogvec[1]-1) + nshockgrid[4]*nshockgrid[3]*(exogvec[2]-1) + nshockgrid[4]*(exogvec[3]-1) + exogvec[4]

    elseif (nexogvars == 3)
        return nshockgrid[3]*nshockgrid[2]*(exogvec[1]-1) + nshockgrid[3]*(exogvec[2]-1) + exogvec[3]

    elseif (nexogvars == 2)
        return nshockgrid[2]*(exogvec[1]-1) + exogvec[2]

    elseif (nexogvars == 1)
        return exogvec[1]
    else
        println("There can at most six exogenous variables.")
    end

    return 0

end

"""
    intermediatedec!(endogvar::Vector{Float64},nendogvars::Int,nexogvars::Int,labss::Float64,endogvarm1::Vector{Float64},currentshockvalues::Vector{Float64},func_vals::Vector{Float64},omega::Float64,funcplus_vals::Vector{Float64},zlbintermediate::Bool,endogenous_states::OrderedCollections.OrderedDict{Symbol,Int64}, exogenous_shocks::OrderedCollections.OrderedDict{Symbol,Int64}, params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64})

# Arguments:
- `endogvar::Vector{Float64}`: Container for updated values of endogenous variables
- `nendogvars::Int`: Number of endogenous variables.
- `nexogvars::Int`: Number of exogenous variables.
- `labss::Float64`: Steady state of labor
- `endogvarm1::Vector{Float64}`: Values of endogenous variables and shocks from previous period.
- `currentshockvalues::Vector{Float64}`: Shock values updated for new period
- `func_vals::Vector{Float64}`: Values of first piece approximated functions (equal to approximated function value when not at zlb)
- `omega::Float64`: Determines weight on zlb and non-zlb function values
- `funcplus_vals::Vector{Float64}`: Values of second piece of approximated functions (when need to split up functions due to zlb)
- `zlbintermediate::Bool`: Indicator for whether we are at zlb or not
- `endogenous_states::OrderedCollections.OrderedDict{Symbol,Int64}`: Maps human-readable endogenous states to indices
- `exogenous_shocks::OrderedCollections.OrderedDict{Symbol,Int64}`: Maps human-readable exogenous states to indices
- `params::Array{AbstractParameter{Float64},1}`: Model parameters
- `keys::OrderedCollections.OrderedDict{Symbol,Int64}`: Maps human-readable parameters to indices

# Description:
Updates endogenous variables and shocks given values of the endogenous variables in the previous period, the current shocks, and the necessary approximated function values.
"""
function intermediatedec!(endogvar::Vector{Float64},nendogvars::Int,nexogvars::Int,labss::Float64,endogvarm1::Vector{Float64},currentshockvalues::Vector{Float64},func_vals::Vector{Float64},omega::Float64,funcplus_vals::Vector{Float64},zlbintermediate::Bool,endogenous_states::OrderedCollections.OrderedDict{Symbol,Int64}, exogenous_shocks::OrderedCollections.OrderedDict{Symbol,Int64}, params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64})

    # Transform shocks from log-level deviations from steady state to levels
    invshk::Float64 = exp(currentshockvalues[exogenous_shocks[:μ_sh]])
    techshk::Float64 = exp(currentshockvalues[exogenous_shocks[:z_sh]])
    rrshk::Float64 = currentshockvalues[exogenous_shocks[:Rnotional_sh]]
    gss::Float64 = 1.0/(1.0-params[keys[:shrgy]])
    gshk::Float64 = exp( log(gss) + currentshockvalues[exogenous_shocks[:g_sh]] )
    ashk::Float64 = exp(0.) #Not used in this model so fixed at 1

    # Update levels of endogenous variables
    # The variables below refer to the V_{l,j}(X_{t-1},τ_t) functions given in equations (2.14) - (2.20) in Technical Appendix of GHLS (2017), which are the functions we approximate
    if (zlbintermediate == true)
        endogvar[endogenous_states[:λ]] = omega*exp(func_vals[1]) + (1.0-omega)*exp(funcplus_vals[1])
        endogvar[endogenous_states[:qk_t]] = omega*exp(func_vals[2]) + (1.0-omega)*exp(funcplus_vals[2])
        bp = omega*func_vals[3] + (1.0-omega)*funcplus_vals[3]
        bww = omega*func_vals[4] + (1.0-omega)*funcplus_vals[4]
        endogvar[endogenous_states[:bc_t]] = omega*exp(func_vals[5]) + (1.0-omega)*exp(funcplus_vals[5])
        endogvar[endogenous_states[:bi_t]] = omega*func_vals[6] + (1.0-omega)*funcplus_vals[6]
        endogvar[endogenous_states[:u_t]] = omega*exp(func_vals[7]) + (1.0-omega)*exp(funcplus_vals[7])
    else
        endogvar[endogenous_states[:λ]] = exp(func_vals[1])
        endogvar[endogenous_states[:qk_t]] = exp(func_vals[2])
        bp::Float64 = func_vals[3]
        bww::Float64 = func_vals[4]
        endogvar[endogenous_states[:bc_t]] = exp(func_vals[5])
        endogvar[endogenous_states[:bi_t]] = func_vals[6]
        endogvar[endogenous_states[:u_t]] = exp(func_vals[7])
    end

    #vp - see (2.5) in GHLS (2017) Technical Appendix (TA). vp = \frac{π(X_{t-1},τ_t)}{\tilde{π}_{t-1}/2} where (X_{t-1},τ_t) is the minimum state vector. (see (2.1) and (2.2) in TA)
    endogvar[endogenous_states[:Vp_t]] = (sqrt(1.0+4.0*bp)+1.0)/2.0

    #vw = π_w(X_{t-1},τ_t)/\tilde{π}_{w,t} (see (2.6) in TA)
    endogvar[endogenous_states[:Vw_t]] = (sqrt(1.0+4.0*bww)+1.0)/2.0

    # Indexation term for price changes - (1.4) of TA --> Expectation of future inflation
    dptildem1::Float64 = (params[keys[:π_bar]]^params[keys[:ap]])*(endogvarm1[endogenous_states[:π_t]]^(1.0-params[keys[:ap]]))

    # Component of indexation term for wage changes - (1.10) in TA - see dw below
    dwtildem1::Float64 = (params[keys[:π_bar]]^params[keys[:aw]])*(endogvarm1[endogenous_states[:π_t]]^(1.0-params[keys[:aw]]))

    # Component of indexation term for wage changes - (1.10) in TA. Note the indexation is dwtildem1*gzwage
    # Note that aw is technically bw but they are always equal
    gzwage::Float64 = params[keys[:gz]]*techshk^(1.0-params[keys[:aw]])

    #dp = π(X_{t-1},τ_t) from (2.5) of TA
    endogvar[endogenous_states[:π_t]] = endogvar[endogenous_states[:Vp_t]]*dptildem1

    #dw = π_w (X_{t-1},τ_t) from (2.6) of TA
    endogvar[endogenous_states[:π_w]] = endogvar[endogenous_states[:Vw_t]]*dwtildem1*gzwage

    #muc (marginal utility of consumption), see last term in (2.8) of TA or (1.27) of TA
    endogvar[endogenous_states[:muc_t]] = endogvar[endogenous_states[:λ]] + (params[keys[:γ]]/params[keys[:gz]])*params[keys[:β]]*endogvar[endogenous_states[:bc_t]]

    #cc = c (X_{t-1},τ_t) (see (2.8) in TA)
    endogvar[endogenous_states[:c_t]] = params[keys[:γ]]*endogvarm1[endogenous_states[:c_t]]/(params[keys[:gz]]*techshk)+1.0/endogvar[endogenous_states[:muc_t]]

    # This is is term in the square root minus 1 divided by 4 of (2.9) of TA
    cquad_vi::Float64 = endogvar[endogenous_states[:bi_t]]/(endogvar[endogenous_states[:qk_t]]*invshk)-(1.0-endogvar[endogenous_states[:qk_t]]*invshk)/(params[keys[:ϕ_I]].value*endogvar[endogenous_states[:qk_t]]*invshk)

    #vi = i(X_{t-1},τ_t)/i_{t-1} from (2.9) in TA
    endogvar[endogenous_states[:Vi_t]] = 0.5*(1.0+sqrt(1.0+4.0*cquad_vi))

    #inv = i(X_{t-1},τ_t) from (2.9) in TA
    endogvar[endogenous_states[:i_t]] = endogvar[endogenous_states[:Vi_t]]*endogvarm1[endogenous_states[:i_t]]/techshk

    #A_{y,t} from (1.33) in TA
    aayy::Float64 = 1.0/gshk-(params[keys[:ϕ_p]]/2.0)*(endogvar[endogenous_states[:Vp_t]]-1.0)*(endogvar[endogenous_states[:Vp_t]]-1.0)

     #r^k from (1.46) in TA
    rkss::Float64 = params[keys[:gz]]/params[keys[:β]]-1.0+params[keys[:δ]]

    #Utilization Cost = a(u_t) from (1.14) in TA
    utilcost::Float64 = (rkss/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(endogvar[endogenous_states[:u_t]]-1.0))-1.0)

    #gdp = y_t - see (1.32) in TA
    endogvar[endogenous_states[:y_t]] = (1.0/aayy)*( endogvar[endogenous_states[:c_t]]+endogvar[endogenous_states[:i_t]] + utilcost*(endogvarm1[endogenous_states[:k_t]]/(params[keys[:gz]]*techshk)))

    #rw = w_t (1.34) in TA
    endogvar[endogenous_states[:w_t]] = endogvarm1[endogenous_states[:w_t]]*endogvar[endogenous_states[:π_w]]/(params[keys[:gz]]*techshk*endogvar[endogenous_states[:π_t]])

    #lab = N_t (1.35) in TA
    endogvar[endogenous_states[:N_t]] = endogvar[endogenous_states[:y_t]]^(1.0/(1.0-params[keys[:α]]))*(endogvar[endogenous_states[:u_t]]*endogvarm1[endogenous_states[:k_t]]/(params[keys[:gz]]*techshk))^(params[keys[:α]]/(params[keys[:α]]-1.0))/ashk

   #Changed llabss to log(m[:labss]), this is xhp = x_t^g from (1.17) in TA. This is same as loged version of the formula at the bottom of pg. 1978 in the main paper for GHLS (2017).
    endogvar[endogenous_states[:x_t]] = params[keys[:α]]*log(endogvar[endogenous_states[:u_t]]) + (1-params[keys[:α]])*(log(endogvar[endogenous_states[:N_t]])-log(labss))

    #Log of steady state nominal interest rate - see (1.43) in TA
    lrss::Float64 = log(params[keys[:gz]]*params[keys[:π_bar]]/params[keys[:β]])

    #Notional interest rate (Interest rate without a zero lower bound) - see (1.39) in TA
    endogvar[endogenous_states[:Rnotional_t]] = exp(lrss+params[keys[:ρ_R]]*(log(endogvarm1[endogenous_states[:Rnotional_t]])-lrss) + (1.0-params[keys[:ρ_R]])*(params[keys[:γ_π]]*log(endogvar[endogenous_states[:π_t]]/params[keys[:π_bar]]) + params[keys[:γ_g]]*log(endogvar[endogenous_states[:y_t]]*techshk/endogvarm1[endogenous_states[:y_t]]) + params[keys[:γ_x]]*endogvar[endogenous_states[:x_t]] ) + rrshk)

    #Marginal Cost: mc_t from (1.36) in TA
    endogvar[endogenous_states[:mc_t]] = endogvar[endogenous_states[:w_t]]*endogvar[endogenous_states[:N_t]]/((1.0-params[keys[:α]])*endogvar[endogenous_states[:y_t]])

    #rentalk: r_t^k from (1.37) in TA
    endogvar[endogenous_states[:rk_t]] = (params[keys[:α]]/(1.0-params[keys[:α]]))*(endogvar[endogenous_states[:w_t]]*endogvar[endogenous_states[:N_t]]*params[keys[:gz]]*techshk/(endogvar[endogenous_states[:u_t]]*endogvarm1[endogenous_states[:k_t]]))

    #cap: \bar{k}_{t+1} from (1.38) in TA
    endogvar[endogenous_states[:k_t]] = (1.0-params[keys[:δ]])*(endogvarm1[endogenous_states[:k_t]]/(params[keys[:gz]]*techshk)) + invshk*endogvar[endogenous_states[:i_t]]*(1.0- (params[keys[:ϕ_I]]/2.0)*(endogvar[endogenous_states[:Vi_t]]-1.0)*(endogvar[endogenous_states[:Vi_t]]-1.0) )

    #this is nomr (Nominal Interest Rate) - see (1.18) in TA
    endogvar[endogenous_states[:Rnominal_t]] = copy(endogvar[endogenous_states[:Rnotional_t]])

    #include shocks with endogvar since they are part of the current state
    endogvar[nendogvars+1:nendogvars+nexogvars] = currentshockvalues

    return nothing
end

"""
    decr!(endogvar::Vector{Float64}, approx::Approximation,endogvarm1::Vector{Float64},innovations::Vector{Float64},params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64}, labss::Float64, alphacoeff::Array{Float64,2}, exogenous_shocks::OrderedCollections.OrderedDict{Symbol, Int64}, endogenous_states::OrderedCollections.OrderedDict(Symbol, Int64}, zlbswitch::Bool)

# Arguments:
- `endogvar::Vector{Float64}`: A vector to contain the updated endogenous variables
- `approx::Approximation`: Approximation object for use in construction of function approximations
- `endogvarm1::Vector{Float64}`: State variables from previous period
- `innovations::Vector{Float64}`: Vector of innovations to shocks
- `params::Array{AbstractParameter{Float64}, 1}`: Vector of model parameters
- `keys::OrderedCollections.OrderedDict{Symbol, Int64}`: Map from human-readable names of parameters and steady-states to their indices
- `labss::Float64`: The steady state of labor, which is the only steady state required in the decision rule
- `alphacoeff::Array{Float64,2}`: The coefficients on Smolyak basis polynomials
- `exogenous_shocks::OrderedCollections.OrderedDict{Symbol, Int64}`: Maps shocks to indices
- `endogenous_states::OrderedCollections.OrderedDict{Symbol, Int64}`: Maps endogenous states to indices
- `zlbswitch::Bool`: Determines whether the model treats the zero lower bound as a binding constraint or not

# Description:
The decision rule, which updates endogenous variables and shocks given previous period's endogenous values and innovations innovations to shocks this period. Will first approximate the necessary functions before calling intermediatedec! to update state variables.
"""
function decr!(endogvar::Vector{Float64}, approx::Approximation, endogvarm1::Vector{Float64},innovations::Vector{Float64},params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64}, labss::Float64, alphacoeff::Array{Float64,2},exogenous_shocks::OrderedCollections.OrderedDict{Symbol,Int64},endogenous_states::OrderedCollections.OrderedDict{Symbol,Int64}, zlbswitch::Bool)

    #Initialize Variables
    shockindexall=ones(Int64,approx.nexogvars)
    shockindex=Array{Int64}(undef,approx.nexogshocks)
    shockindex_inter=Array{Int64}(undef,approx.nexogshocks)
    lmsv=Array{Float64}(undef,approx.nmsv)
    currentshockvalues=Array{Float64}(undef,approx.nexogvars)
    funcmatplus=Array{Float64}(undef,approx.nfunc,approx.ninter)
    weighttemp=Array{Float64}(undef,approx.nexogshocks)
    funcvals=Array{Float64}(undef,approx.nfunc)
    funcplus_vals=Array{Float64}(undef,approx.nfunc)
    xx=Array{Float64}(undef,approx.nmsv)
    polyvec=Array{Float64}(undef,approx.ngridpoints)
    omegaweight = 100000.0

    #Update shocks (all shocks are in deviation from steady state)
    currentshockvalues[1]::Float64 = params[keys[:ρ_η]]*endogvarm1[endogenous_states[:η_t]] + params[keys[:σ_η]]*innovations[1]
    currentshockvalues[2]::Float64 = params[keys[:ρ_μ]]*endogvarm1[endogenous_states[:μ_t]] + params[keys[:σ_μ]]*innovations[2]
    currentshockvalues[3]::Float64 = params[keys[:σ_Z]]*innovations[3]
    currentshockvalues[4]::Float64 = params[keys[:ρ_int]]*endogvarm1[endogenous_states[:Rnotionalshock_t]] + params[keys[:σ_R]]*innovations[4]
    currentshockvalues[5]::Float64 = params[keys[:ρ_g]]*endogvarm1[endogenous_states[:g_t]] + params[keys[:σ_g]]*innovations[5]
    currentshockvalues[6] = 0.0 #a shock

    # Finds shock index associated with current shock value (rounded down)
    @simd for i in 1:approx.nexogshocks
        if (currentshockvalues[i] < approx.shockbounds[i,1])
            shockindex[i] = 1
        elseif (currentshockvalues[i] > approx.shockbounds[i,2])
            shockindex[i] = floor(Int64,(approx.shockbounds[i,2]-approx.shockbounds[i,1])/approx.shockdistance[i] )
        else
            shockindex[i] = floor(Int64,1.0+(currentshockvalues[i]-approx.shockbounds[i,1])/approx.shockdistance[i])
        end
    end

    # Lowest possible state (all indices rounded down)
    shockindexall[1:approx.nexogshocks] = shockindex
    stateindex0 = exogstate(shockindexall,approx.nshockgrid,approx.nexogvars)

    # Highest possible state (all indices rounded up)
    shockindexall[1:approx.nexogshocks] = shockindex + approx.interpolatemat[:,approx.ninter]
    stateindex1 = exogstate(shockindexall,approx.nshockgrid,approx.nexogvars)

    # One possible function for each possible interpolation (i.e. every combination of rounding up/down)
    funcmat = Array{Float64}(undef,approx.nfunc,approx.ninter)
    weightvec = Array{Float64}(undef,approx.ninter)

    #Log the minimum state variables
    lmsv[1:approx.nmsv] = log.(endogvarm1[1:approx.nmsv])

    # Convert endogenous variables to [-1, 1] domain
    xx = msv2xx(lmsv,approx.nmsv,approx.slopeconmsv)

    # Get Smolyak basis polynomials evaluated at these values (this is T(ϕ(X_{t-1})))
    polyvec = smolyakpoly(approx.nmsv,approx.ngridpoints,approx.nindplus,approx.indplus,xx)

    prod_sd = prod(approx.shockdistance)

    @inbounds @simd for i in 1:approx.ninter
        @inbounds @simd for j in 1:approx.nexogshocks
            shockindexall[j] = shockindex[j] + approx.interpolatemat[j,i]

            # Shock grid points that are closer to actual shock values result in lower weights
            weighttemp[j]=(1-approx.interpolatemat[j,i])*(currentshockvalues[j]-approx.exoggrid[j,stateindex0]) + (approx.interpolatemat[j,i])*(approx.exoggrid[j,stateindex1]-currentshockvalues[j])
        end

        stateindex = exogstate(shockindexall,approx.nshockgrid,approx.nexogvars)
        stateindexplus = stateindex+approx.ns

        # Calculates $T(ϕ(X_{t-1})α_{l, j, k}
        @inbounds @simd for ifunc in 1:approx.nfunc
            @views funcmat[ifunc,i] = BLAS.dot(approx.ngridpoints, alphacoeff[(ifunc-1)*approx.ngridpoints+1:ifunc*approx.ngridpoints, stateindex], 1, polyvec, 1)
            @views funcmatplus[ifunc,i] = BLAS.dot(approx.ngridpoints, alphacoeff[(ifunc-1)*approx.ngridpoints+1:ifunc*approx.ngridpoints, stateindexplus],1, polyvec, 1)
        end

        # Give weight to inverse interpolation, so that in the end interpolations that are closer to actual shocks are given high weights, note this is Γ_{k}
        # Also note prod_sd is the most prod(weighttemp) can be
        weightvec[approx.ninter+1-i] = prod(weighttemp)/prod_sd
    end

    #In-place version of funcvals = funcmat*weightvec
    mul!(funcvals, funcmat, weightvec)

    zlbintermediate = false  #Start with evaluation of only first piece of each function
    omegapoly = 1.0 # Put all weight on first function

    # Update endogenous variables
    intermediatedec!(endogvar,approx.nendogvars,approx.nexogvars,labss, endogvarm1,currentshockvalues,funcvals,omegapoly,funcvals,zlbintermediate, endogenous_states, exogenous_shocks, params, keys)

    # See if we are at zlb
    if ( (endogvar[endogenous_states[:Rnotional_t]] < 1.0) & (zlbswitch == true) )
        zlbintermediate = true

        # If we are then calculate weight to put on first function as well as values of second piece of each function and then re-calculate updated endogenous variables
        omega = exp(omegaweight*log(endogvar[endogenous_states[:Rnotional_t]]))
        mul!(funcplus_vals, funcmatplus,  weightvec)
        intermediatedec!(endogvar,approx.nendogvars,approx.nexogvars, labss, endogvarm1,currentshockvalues, funcapp,omega,funcplus_vals,zlbintermediate, endogenous_states, exogenous_shocks, params, keys)
        endogvar[endogenous_states[:Rnominal_t]] = 1.0 #Nominal interest rate constrained to 1
    end

    return nothing
end

"""
    decrlin(endogvarm1::Vector{Float64},innovations::Vector{Float64},nendogvars::Int,nexogvars::Int,sigma::Array{Float64,2},pp::Array{Float64,2})

# Arguments:
- `endogvarm1::Vector{Float64}`: Previous period's endogenous variables and shock values.
- `innovations::Vector{Float64}`: Innovations to the shocks.
- `nendogvars::Int`: Number of endogenous variables.
- `nexogvars::Int`: Number of exogenous variables.
- `sigma::Array{Float64, 2}`: Innovation part of the linear transition equation
- `pp::Array{Float64, 2}`: Feedback (endogenous varaibles lags) part of the linear transition equation

Description:
Decision rule for linearized version, which calculates updated endogenous variables and shocks given lagged endogenous values and innovations.
"""
function decrlin(endogvarm1::Vector{Float64},innovations::Vector{Float64},nendogvars::Int,nexogvars::Int,sigma::Array{Float64,2},pp::Array{Float64,2},steady_states::Array{Float64,1})


    # Initialize variables
    xx = Vector{Float64}(undef,nendogvars+nexogvars)
    exogpart = Array{Float64}(undef,nendogvars+nexogvars)
    endogsteady = steady_states[1:nendogvars + nexogvars]

    # Get endogenous variables in terms of deviation from steady state
    endogvarm1[1:nendogvars] -= endogsteady[1:nendogvars]

    # Update values according to transition equation
    mul!(xx, pp, endogvarm1[1:nendogvars+nexogvars])
    mul!(exogpart,sigma,innovations)
    xx = xx + exogpart

    xx[1:nendogvars] += endogsteady[1:nendogvars]

    return xx
end

"""
    decr_euler(rkss::Float64, approx::Approximation, gridindex::Int64,shockpos::Int64, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64},alphacoeff::Array{Float64,2}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, zlbswitch::Bool)

# Arguments:
- `rkss::Float64`: Steady state for rental rate of capital
- `approx::Approximation`: Approximation object for use in construction of function and integral approximations
- `gridindex::Int64`: Index of grid point on Smolyak grid
- `shockpos::Int64`: Exogenous state (which is also the index of the grid point on the exogenous shock grid)
- `params::Array{AbstractParameter{Float64},1}`: Model parameters
- `keys::OrderedDict{Symbol,Int64}`: Maps from human-readable names of parameters to their indices
- `alphacoeff::Array{Float64,2}`: Alpha coefficients for function approximations
- `labss::Float64`: Steady state of labor
- `exogenous_shocks::OrderedDict{Symbol,Int64}`: Maps human-readable exogenous shocks to indices
- `endogenous_states::OrderedDict{Symbol,Int64}`: Maps from human-readable endogenous states to indices
- `zlbswitch::Bool`: Determines whether zero lower bound will be a binding constraint

# Description:
Gets new values of functions at given exogenous state (grid point on the exogenous shock grid) and grid point on the Smolyak grid using the previous values of the functions at that point. At the model solution we are at a fixed point so these values should be the same.
"""
function decr_euler(rkss::Float64, approx::Approximation, gridindex::Int64,shockpos::Int64, params::Array{AbstractParameter{Float64},1}, keys::OrderedDict{Symbol,Int64},alphacoeff::Array{Float64,2}, labss::Float64, exogenous_shocks::OrderedDict{Symbol,Int64},endogenous_states::OrderedDict{Symbol,Int64}, zlbswitch::Bool)

    #Initialize Variables
    zlbinfo  = approx.statezlbinfo[shockpos]
    funcvalsnew = Array{Float64}(undef,2*approx.nfunc)
    endogvar = Array{Float64}(undef,approx.nendogvars+approx.nexogvars)
    endogvarzlb = Array{Float64}(undef,approx.nendogvars+approx.nexogvars)
    endogvarp = Array{Float64}(undef,approx.nendogvars+approx.nexogvars)
    endogvarzlbp = Array{Float64}(undef,approx.nendogvars+approx.nexogvars)
    slopeconxxmsv = Array{Float64}(undef,2*approx.nmsv)
    xgridmsv = Array{Float64}(undef,approx.nmsv)
    abserror = Array{Float64}(undef,2*approx.nfunc)
    ev = Array{Float64}(undef,12)
    exp_eul = Array{Float64}(undef,12)
    currentshockvalues = Array{Float64}(undef,approx.nexogvars)
    funcvals = Array{Float64}(undef,2*approx.nfunc)
    endogvarm1 = Array{Float64}(undef,approx.nendogvars+approx.nexogvars)
    expectations = Array{Float64}(undef, 12)
    innovations = Array{Float64}(undef,approx.nexogvars)

    # Gets shock values for given position on shock grid
    currentshockvalues[1:approx.nexogshocks] = approx.exoggrid[1:approx.nexogshocks,shockpos]

    # Calculates T(ϕ(X_{t-1}))α_{l,j,k} for each function
    # Note Γ_{k} is one here for the shock grid point we are at and zero everywhere else and so we do not need to calculate
    shockpospoly = shockpos + approx.ns
    for ifunc in 1:approx.nfunc
        @views funcvals[ifunc] = BLAS.dot(approx.ngridpoints, alphacoeff[(ifunc-1)*approx.ngridpoints+1:ifunc*approx.ngridpoints,shockpos],1,approx.bbt[:,gridindex], 1)
        @views funcvals[approx.nfunc+ifunc] = BLAS.dot(approx.ngridpoints, alphacoeff[(ifunc-1)*approx.ngridpoints+1:ifunc*approx.ngridpoints,shockpospoly],1,approx.bbt[:,gridindex], 1)
    end

    # Get endogenous variable values for grid point on Smoylak grid and convert back from [-1, 1] domain
    xgridmsv = approx.xgrid[1:approx.nmsv,gridindex]
    slopeconxxmsv[1:approx.nmsv] = approx.slopeconxx[1:approx.nmsv]
    slopeconxxmsv[approx.nmsv+1:2*approx.nmsv] = approx.slopeconxx[approx.nmsv+1:approx.nmsv+approx.nmsv]
    endogvarm1[1:approx.nmsv] = exp.( msv2xx(xgridmsv,approx.nmsv,slopeconxxmsv) )

    zlbintermediate = false  # Start by assuming not at zlb
    omega = 1.0 # Since not at zlb only use first piece of function approximation

    # Update endogenous variables given shocks and function approximations
    intermediatedec!(endogvar,approx.nendogvars,approx.nexogvars, labss, endogvarm1,currentshockvalues, funcvals[1:approx.nfunc],omega,funcvals[1:approx.nfunc],zlbintermediate, endogenous_states, exogenous_shocks, params, keys)

    # If we are accounting for zlb and are at the zlb then need to re-get updated endogenous variables
    if ((zlbinfo != 0) & (zlbswitch == true))
        # Why do we not recalculate omega here/set zlbintermediate true?
        intermediatedec!(endogvarzlb, approx.nendogvars,approx.nexogvars, labss, endogvarm1,currentshockvalues, funcvals[approx.nfunc+1:2*approx.nfunc], omega,funcvals[approx.nfunc+1:2*approx.nfunc],zlbintermediate, endogenous_states, exogenous_shocks, params, keys)
        endogvarzlb[endogenous_variables[:Rnominal_t]] = 1.0
    end

    # Calculate conditional expectations
    for quadpoint in 1:approx.nquad
        innovations[1:approx.nexogshocks] = approx.ghnodes[:,quadpoint]

        decr!(endogvarp,approx,endogvar,innovations,params, keys, labss, alphacoeff, exogenous_shocks, endogenous_states, zlbswitch)

        techshkp = exp(endogvarp[endogenous_states[:z_t]])
        invshkp = exp(endogvarp[endogenous_states[:μ_t]])

        # Note below that when multiple equations are cited, they are both the same expression, just in different places

        # This is what's inside the expectation in from (1.26) in Technical Appendix of GHLS (2017), for V_{λ,t}, same as expectation in (2.1)
        ev[1] = endogvarp[endogenous_states[:λ]]/(endogvarp[endogenous_states[:π_t]]*techshkp)

        # a(u_t) - see equation (5) on pg. 1977 of GHLS (2017), same as (1.14) in Technical Appendix (TA)
        utilcostp = (rkss/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(endogvarp[13]-1.0))-1.0)

        # Value in expectation in (1.29) of TA, for V_{q,t} or (2.16) - same expectation
        ev[2] = (endogvarp[endogenous_states[:λ]]/techshkp)*( (endogvarp[endogenous_states[:rk_t]]*endogvarp[endogenous_states[:u_t]])-utilcostp+(1.0-params[keys[:δ]])*endogvarp[endogenous_states[:qk_t]])

        # Value in expectation in (2.18) and (2.3) of TA, for V_{π,j} and V_{π,t}
        ev[3] = endogvarp[endogenous_states[:λ]]*(endogvarp[endogenous_states[:Vp_t]]-1.0)*endogvarp[endogenous_states[:Vp_t]]*endogvarp[endogenous_states[:y_t]]

        # Value in expectation in (2.19) and(2.4) of TA, for V_{w,j} and V_{w,t}
        ev[4] = (endogvarp[endogenous_states[:Vw_t]]-1.0)*endogvarp[endogenous_states[:Vw_t]]

        # Value in expectation in (2.15) of TA, for V_{c,j} or (1.28) of TA
        ev[5] = endogvarp[endogenous_states[:muc_t]]/techshkp

        # Value in expectation in (2.17) of TA, for V_{i,j} or (1.31) of TA
        ev[6] = endogvarp[endogenous_states[:λ]]*endogvarp[endogenous_states[:qk_t]]*invshkp*(endogvarp[endogenous_states[:Vi_t]]-1.0)*endogvarp[endogenous_states[:Vi_t]]*endogvarp[endogenous_states[:Vi_t]]

        # Equation sources are same as above, just endogvarzlbp is used instead of endogvarp
        if ((zlbinfo != 0) & (zlbswitch == true))
            decr!(endogvarzlbp,approx,endogvarzlb,innovations, params, keys, labss, alphacoeff, exogenous_shocks, endogenous_states, zlbswitch)
            ev[7] = endogvarzlbp[endogenous_states[:λ]]/(endogvarzlbp[endogenous_states[:π_t]]*techshkp)
            utilcostp = (rkss/params[keys[:σ_a]])*(exp(params[keys[:σ_a]]*(endogvarzlbp[endogenous_states[:u_t]]-1.0))-1.0)
            ev[8] = (endogvarzlbp[endogenous_states[:λ]]/techshkp)*( (endogvarzlbp[endogenous_states[:rk_t]]*endogvarzlbp[endogenous_states[:u_t]])-utilcostp+(1-params[keys[:δ]])*endogvarzlbp[endogenous_states[:qk_t]] )
            ev[9] = endogvarzlbp[endogenous_states[:λ]]*(endogvarzlbp[endogenous_states[:Vp_t]]-1.0)*endogvarzlbp[endogenous_states[:Vp_t]]*endogvarzlbp[endogenous_states[:y_t]]
            ev[10] = (endogvarzlbp[endogenous_states[:Vw_t]]-1.0)*endogvarzlbp[endogenous_states[:Vw_t]]
            ev[11] = endogvarzlbp[endogenous_states[:muc_t]]/techshkp
            ev[12] = endogvarzlbp[endogenous_states[:λ]]*endogvarzlbp[endogenous_states[:qk_t]]*invshkp*(endogvarzlbp[endogenous_states[:Vi_t]]-1.0)*endogvarzlbp[endogenous_states[:Vi_t]]*endogvarzlbp[endogenous_states[:Vi_t]]
        else
            ev[7:12] = ev[1:6]
        end

        # Approximate expectation by taking weighted sum over integration nodes
        expectations += approx.ghweights[quadpoint]*ev
    end

    liqshk = exp(endogvar[endogenous_states[:η_t]])
    invshk = exp(endogvar[endogenous_states[:μ_t]])
    ep = params[keys[:ϵ_p]].scaledvalue

    #λ_t from equation (1.26) in TA
    exp_eul[1] =  (params[keys[:β]]/params[keys[:gz]])*liqshk*endogvar[endogenous_states[:Rnominal_t]]*expectations[1]

    #q_t from equation (1.29) in TA
    exp_eul[2] =  (params[keys[:β]]/params[keys[:gz]])*expectations[2]/endogvar[endogenous_states[:λ]]

    #V_{π_t} from (2.3) in TA
    exp_eul[3] =  params[keys[:β]]*expectations[3]/(endogvar[endogenous_states[:λ]]*endogvar[endogenous_states[:y_t]])+(params[keys[:ϵ_p]]/params[keys[:ϕ_p]])*(endogvar[endogenous_states[:mc_t]]-(params[keys[:ϵ_p]]-1.0)/params[keys[:ϵ_p]])

    #V_{w,t} from (2.4) in TA
    exp_eul[4] = params[keys[:β]]*expectations[4]+(params[keys[:ϵ_w]]/params[keys[:ϕ_w]])*endogvar[endogenous_states[:λ]]*endogvar[endogenous_states[:N_t]]*
           ( (params[keys[:ψ_L]]*endogvar[endogenous_states[:N_t]]^params[keys[:σ_L]]/endogvar[endogenous_states[:λ]])-((params[keys[:ϵ_w]]-1.0)/params[keys[:ϵ_w]])*endogvar[endogenous_states[:w_t]] )

    #V_{c,t} from (1.28) in TA
    exp_eul[5] = expectations[5]

    #V_{i,t} from (1.31) in TA (This not being multiplied by m[:ϕ_I]  since in cquad_vi of intermediatedec! we don't divide V_{i,t} by m[:ϕ_I])
    exp_eul[6] = (params[keys[:β]]/params[keys[:gz]])*expectations[6]/endogvar[endogenous_states[:λ]] - 0.5*endogvar[endogenous_states[:qk_t]]*invshk*(endogvar[endogenous_states[:Vi_t]]-1.0)*(endogvar[endogenous_states[:Vi_t]]-1.0)

    funcvalsnew[1] = log(exp_eul[1])
    funcvalsnew[2] = log(exp_eul[2])
    funcvalsnew[3] = exp_eul[3]
    funcvalsnew[4] = exp_eul[4]
    funcvalsnew[5] = log(exp_eul[5])
    funcvalsnew[6] = exp_eul[6]
    funcvalsnew[7] = log( 1.0 + (1/params[keys[:σ_a]])*log(endogvar[endogenous_states[:rk_t]]/rkss) ) # V_{u,t} from (2.20)

    # Below equations are the same as above, just for the zero lower bound case.
    if ((zlbinfo != 0) & (zlbswitch == true))
        exp_eul[7] =  (params[keys[:β]]/params[keys[:gz]])*liqshk*endogvarzlb[endogenous_states[:Rnominal_t]]*expectations[7]
        exp_eul[8] = (params[keys[:β]]/params[keys[:gz]])*expectations[8]/endogvarzlb[endogenous_states[:λ]]
        exp_eul[9] =  params[keys[:β]]*expectations[9]/(endogvarzlb[endogenous_states[:λ]]*endogvarzlb[endogenous_states[:y_t]])+(params[keys[:ϵ_p]]/params[keys[:ϕ_p]])*( endogvarzlb[endogenous_states[:mc_t]]-(params[keys[:ϵ_p]]-1.0)/params[keys[:ϵ_p]])
        exp_eul[10] = params[keys[:β]]*expectations[10]+(params[keys[:ϵ_w]]/params[keys[:ϕ_w]])*endogvarzlb[endogenous_states[:λ]]*endogvarzlb[endogenous_states[:N_t]]*
           ( (params[keys[:ψ_L]]*endogvarzlb[endogenous_states[:N_t]]^params[keys[:σ_L]]/endogvarzlb[endogenous_states[:λ]])-((params[keys[:ϵ_w]]-1.0)/params[keys[:ϵ_w]])*endogvarzlb[endogenous_states[:w_t]])
        exp_eul[11] = expectations[11]
        exp_eul[12] = (params[keys[:β]]/params[keys[:gz]])*expectations[12]/endogvarzlb[endogenous_states[:λ]] - 0.5*endogvarzlb[endogenous_states[:qk_t]]*invshk*(endogvarzlb[endogenous_states[:Vi_t]]-1.0)*(endogvarzlb[endogenous_states[:Vi_t]]-1.0)

        funcvalsnew[8] = log(exp_eul[7])
        funcvalsnew[9] = log(exp_eul[8])
        funcvalsnew[10] = exp_eul[9]
        funcvalsnew[11] = exp_eul[10]
        funcvalsnew[12] = log(exp_eul[11])
        funcvalsnew[13] = exp_eul[12]
        funcvalsnew[14] = log( 1.0 + (1/params[keys[:σ_a]])*log(endogvarzlb[endogenous_states[:rk_t]]/rkss) )
    else
        exp_eul[7:12] = exp_eul[1:6]
        funcvalsnew[8:14] = funcvalsnew[1:7]
    end

    # Gets average error across all approximated functions
    erravg = 0.0
    for ifunc in 1:2*approx.nfunc
        abserror[ifunc] = abs(funcvalsnew[ifunc]-funcvals[ifunc])
        erravg += abserror[ifunc]
    end

    erravg /= (2*approx.nfunc)

    return funcvalsnew, erravg

end

"""
    finite_grid!(shockgrid::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}, shockdistance::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}, shockbounds::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}, n::Int64,rho::Float64,sigmaep::Float64)

# Arguments:
- `shockgrid::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}: Array containing possible values of shock
- `shockdistance::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}: Distance between values of shocks on grid
- `shockbounds::SubArray{Float64, 1, Array{Float64,1}, Tuple{UnitRange{Int64}},true}: Minimum and maximum values of shock
- `n::Int`: Number of shock realizations.
- `rho::Float64`: AR(1) coefficient.
- `sigmaep::Float64`: Standard deviation of innovation.

# Description:
For a given shock, this populates the shockgrid array with the possible values of the shock, spanning -nu to nu and evenly spaced. It also records the shockdistance (distance between shock values) and shockbounds (min and max shock values).
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
    gen_shockgrid(nshockgrid::Array{Int,1}, nexogshocks::Int, ns::Int, nexogvars::Int, params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64})

- `nshockgrid::Array{Int, 1}`: Array indexing number of realizations for each shock (length nexogvars).
- `nexogshocks::Int`: Number of exogenous shock processes that are active (number of possible values is strictly greater than one)
- `ns::Int`: Number of exogenous states.
- `nexogvars::Int`: Number of exogenous shock processes.
- `params::Array{AbstractParameter{Float64},1}`: Model parameters
- `keys::OrderedCollections.OrderedDict{Symbol,Int64}`: Maps human-readable parameter names to indices

# Description:
Creates a grid of shock values for use in interpolation. For each exogenous shock, a grid point is associated with the value that that shock takes on at that grid point. In total there are  'ns' such grid points, with each point representing a distinct combination of shock values. Also returns the upper and lower bounds of each shock, as well as the distance between shock values for any two grid points, which is a constant for each shock since the grid is evenly spaced along each dimension (although different dimensions, which correspond to different shocks, could have different spacing).
"""
function gen_shockgrid(nshockgrid::Array{Int,1}, nexogshocks::Int, ns::Int, nexogvars::Int, params::Array{AbstractParameter{Float64},1}, keys::OrderedCollections.OrderedDict{Symbol,Int64})

    #Initiliaze Variables
    shockbounds = Array{Float64}(undef,nexogshocks,2)
    shockdistance = Array{Float64}(undef,nexogshocks)
    currentshockindex = Array{Float64}(undef,nexogshocks)
    nshocksum_vec = Array{Float64}(undef,nexogshocks)
    shockvalues = Array{Float64}(undef,sum(nshockgrid))
    exoggrid = zeros(nexogvars,ns) #Not undef since sometimes not every row is initialized (when nexogshocks < nexogvars)

    rhovec = [params[keys[:ρ_η]].value,params[keys[:ρ_μ]].value,0.0,params[keys[:ρ_int]].value,params[keys[:ρ_g]].value, 0.0]
    sigmavec = [params[keys[:σ_η]].scaledvalue,params[keys[:σ_μ]].scaledvalue,params[keys[:σ_Z]].scaledvalue,params[keys[:σ_R]].scaledvalue,params[keys[:σ_g]].scaledvalue,0.0] #we should remove ashk it is not used

    # Tracks the total number of shock values that have been inputted into shockvalues array
    num_shockvalues = 0

    # For each shock populate the shockvalues array with the values it can take on
    @simd for i in 1:nexogshocks

        # Use views so that array slices can be mutated
        @views finite_grid!(shockvalues[num_shockvalues+1:num_shockvalues+nshockgrid[i]], shockdistance[i:i], shockbounds[i, :], nshockgrid[i],rhovec[i],sigmavec[i])
        num_shockvalues += nshockgrid[i]
    end

    # Associates a combination of shocks with the indices of the values that the shocks take on at that combination
    exoggrid_indices = gen_exoggrid_indices(nshockgrid, nexogvars, ns)

    # For each distinct combination of shocks
    @simd for ss in 1:ns
        num_prev_shockvalues = 0

        # Put the shock value (rather than index) in to exoggrid
        for i in 1:nexogshocks
            # shockvalues is 1D (which makes it easier to accomodate different shocks having a different number of values) so need to offset index from exoggridindices by numshockvalues to get correct value
            exoggrid[i, ss] = shockvalues[num_prev_shockvalues + exoggrid_indices[i, ss]]
            num_prev_shockvalues += nshockgrid[i]
        end
    end

    return exoggrid,shockbounds,shockdistance

end
