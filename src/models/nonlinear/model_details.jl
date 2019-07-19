function float_dot(x:: SubArray{Float64,1,Array{Float64,2},Tuple{UnitRange{Int64},Int64},true},y::Array{Float64})

    dot=0.0
    @simd for i in 1:length(y)
        dot = dot+x[i]*y[i]
    end
    return dot
end

@views view(x,l,u,i) = x[l:u,i]  #More quickly takes slice of an array

function msv2xx(msv::Array{Float64}, nmsv::Int, slopeconmsv::Array{Float64})
    return slopeconmsv[1:nmsv] .* msv + slopeconmsv[nmsv+1:2*nmsv]
end

function exogposition(exogvec::Array{Int64}, nrvec::Array{Int64}, nlength::Int64)

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

function intermediatedec(m::GHLS,endogvarm1::Array{Float64},currentshockvalues::Array{Float64},polyvar::Array{Float64},omegapoly::Float64,polyvarplus::Array{Float64},zlbintermediate::Bool)

    #Input
    nvars = m.approx.nvars
    nexog = m.approx.nexog

    #Initilize Variables
    endogvar=Array{Float64}(undef,nvars+nexog)

    invshk = exp(currentshockvalues[m.exogenous_shocks[:μ_sh]])
    techshk = exp(currentshockvalues[m.exogenous_shocks[:ztil_sh]])
    rrshk = currentshockvalues[m.exogenous_shocks[:rm_sh]]
    gss = 1.0/(1.0-m[:shrgy])
    gshk = exp( log(gss) + currentshockvalues[m.exogenous_shocks[:g_sh]] )
    ashk = exp( currentshockvalues[6] ) # Which of the last 3 shocks is this?

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
    dptildem1 = (m[:π_bar]^m[:ap])*(dpm1^(1.0-m[:ap]))
    dwtildem1 = (m[:π_bar]^m[:aw])*(dpm1^(1.0-m[:aw]))
    gzwage = m[:gz]*techshk^(1.0-m[:bw])
    dp = vp*dptildem1
    dw = vw*dwtildem1*gzwage
    muc = lam + (m[:γ]/m[:gz])*m[:β]*bc
    cc = m[:γ]*ccm1/(m[:gz]*techshk)+1.0/muc
    cquad_vi = bi/(qq*invshk)-(1.0-qq*invshk)/(m[:ϕ_I].value*qq*invshk)

    vi = 0.5*(1.0+sqrt(1.0+4.0*cquad_vi))

    inv = vi*invm1/techshk
    aayy = 1.0/gshk-(m[:ϕ_p]/2.0)*(vp-1.0)*(vp-1.0)
    rkss = m[:gz]/m[:β]-1.0+m[:δ]
    utilcost = (rkss/m[:σ_a])*(exp(m[:σ_a]*(util-1.0))-1.0)
    gdp = (1.0/aayy)*( cc+inv + utilcost*(capm1/(m[:gz]*techshk)) )
    rw = rwm1*dw/(m[:gz]*techshk*dp)
    lab = gdp^(1.0/(1.0-m[:α]))*(util*capm1/(m[:gz]*techshk))^(m[:α]/(m[:α]-1.0))/ashk
    xhp = m[:α]*log(util) + (1-m[:α])*(log(lab)-log(m[:labss])) # Changed llabss to log(m[:labss])

    lrss = log(m[:gz]*m[:π_bar]/m[:β])
    notr = exp(lrss+m[:ρ_R]*(log(notrm1)-lrss) + (1.0-m[:ρ_R])*(m[:γ_π]*log(dp/m[:π_bar]) + m[:γ_g]*log(gdp*techshk/gdpm1) + m[:γ_x]*xhp ) + rrshk)

    mc = rw*lab/((1.0-m[:α])*gdp)
    rentalk = (m[:α]/(1.0-m[:α]))*(rw*lab*m[:gz]*techshk/(util*capm1))
    cap = (1.0-m[:δ])*(capm1/(m[:gz]*techshk)) + invshk*inv*(1.0- (m[:ϕ_I]/2.0)*(vi-1.0)*(vi-1.0) )
    nomr = copy(notr) # copy as to no make it a pointer

    #all variables are in levels except for shocks
    #shocks are in log-level deviations from SS
    #(1) cap, (2) cc, (3) inv, (4) rw, (5) notr, (6) dp, (7) gdp, (8) xhp, (9) nomr, (10) lam, (11) qq,
    #(12) lab, (13) util, (14) mc, (15) rentalk, (16) muc, (17) vi, (18) vp, (19) vw, (20) dw, (21) bc, (22) bi,
    #(23) liqshk, (24) invshk, (25) techshk, (26) rrshk, (27) gshk, (28) ashk

    endogvar = vcat([cap, cc, inv, rw, notr, dp, gdp, xhp, nomr, lam, qq, lab, util, mc, rentalk, muc, vi, vp, vw, dw, bc, bi], currentshockvalues) #Concatenate two arrays

    return endogvar
end

function decr(m::GHLS,endogvarm1::Array{Float64},innovations::Array{Float64},alphacoeff::Array{Float64})

    #Initilize Variables
    endogvar=Array{Float64}(undef,m.approx.nvars+m.approx.nexog)
    shockindexall=ones(Int64,m.approx.nexog-m.approx.nexogcont)
    shockindex=Array{Int64}(undef,m.approx.nexogshock)
    shockindex_inter=Array{Int64}(undef,m.approx.nexogshock)
    lmsv=Array{Float64}(undef,m.approx.nmsv+m.approx.nexogcont)
    currentshockvalues=Array{Float64}(undef,m.approx.nexog)
    funcmatplus=Array{Float64}(undef,m.approx.nfunc,m.approx.ninter)
    weighttemp=Array{Float64}(undef,m.approx.nexogshock)
    funcapp=Array{Float64}(undef,m.approx.nfunc)
    funcapp_plus=Array{Float64}(undef,m.approx.nfunc)
    xx=Array{Float64}(undef,m.approx.nmsv)
    polyvec=Array{Float64}(undef,m.approx.ngrid)

    omegaweight = 100000.0

    nmsvplus = m.approx.nmsv+m.approx.nexogcont

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
    currentshockvalues[1] = m[:ρ_η]*endogvarm1[m.endogenous_states[:b_t]] + m[:σ_η]*innovations[1]
    currentshockvalues[2] = m[:ρ_μ]*endogvarm1[m.endogenous_states[:μ_t]] + m[:σ_μ]*innovations[2]
    currentshockvalues[3] = m[:σ_Z]*innovations[3]
    currentshockvalues[4] = m[:ρ_int]*endogvarm1[m.endogenous_states[:mon_t]] + m[:σ_R]*innovations[4]
    currentshockvalues[5] = m[:ρ_g]*endogvarm1[m.endogenous_states[:g_t]] + m[:σ_g]*innovations[5]
    currentshockvalues[6] = 0.0*endogvarm1[28] + 0.0*innovations[6] # Which one of the three is this? Is it unk_t? Isn't this the a shock?

    #find position of shocks for interpolation
    @simd for i in 1:m.approx.nexogshock
        if (currentshockvalues[i] < m.approx.shockbounds[i,1])   #extrapolation below
            shockindex[i] = 1
        elseif (currentshockvalues[i] > m.approx.shockbounds[i,2])  #extrapolation above
            shockindex[i] = floor(Int64,(m.approx.shockbounds[i,2]-m.approx.shockbounds[i,1])/m.approx.shockdistance[i] )
        else
            shockindex[i] = floor(Int64,1.0+(currentshockvalues[i]-m.approx.shockbounds[i,1])/m.approx.shockdistance[i])  #interpolation case
        end
    end

    #loop for interpolating between shocks
    shockindexall[1:m.approx.nexogshock] = shockindex
    stateindex0 = exogposition(shockindexall,m.approx.nshockgrid,m.approx.nexog-m.approx.nexogcont)
    shockindexall[1:m.approx.nexogshock] = shockindex + m.approx.interpolatemat[:,m.approx.ninter]
    stateindex1 = exogposition(shockindexall,m.approx.nshockgrid,m.approx.nexog-m.approx.nexogcont)
    funcmat = zeros(m.approx.nfunc,m.approx.ninter)
    weightvec = zeros(m.approx.ninter)
    lmsv[1:m.approx.nmsv] = log.(endogvarm1[1:m.approx.nmsv])
    if (m.approx.nexogcont > 0)
        lmsv[m.approx.nmsv+1:m.approx.nmsv+m.approx.nexogcont] = currentshockvalues[m.approx.nexog-m.approx.nexogcont+1:m.approx.nexog]
    end

    xx = msv2xx(lmsv,nmsvplus,m.approx.slopeconmsv)

    polyvec = smolyakpoly(nmsvplus,m.approx.ngrid,m.approx.nindplus,m.approx.indplus,xx)

    prod_sd = prod(m.approx.shockdistance)

    for i in 1:m.approx.ninter # @inbounds, stops the bounds check
        @fastmath @inbounds @simd for j in 1:m.approx.nexogshock
            shockindexall[j] = shockindex[j] + m.approx.interpolatemat[j,i]
            weighttemp[j]=(1-m.approx.interpolatemat[j,i])*(currentshockvalues[j]-m.approx.exoggrid[j,stateindex0]) + (m.approx.interpolatemat[j,i])*(m.approx.exoggrid[j,stateindex1]-currentshockvalues[j])
        end
        stateindex = exogposition(shockindexall,m.approx.nshockgrid,m.approx.nexog-m.approx.nexogcont)
        stateindexplus = stateindex+m.approx.ns
        @fastmath @inbounds @simd for ifunc in 1:m.approx.nfunc
            funcmat[ifunc,i] = float_dot(view(alphacoeff,(ifunc-1)*m.approx.ngrid+1,ifunc*m.approx.ngrid,stateindex),polyvec)
            funcmatplus[ifunc,i] = float_dot(view(alphacoeff,(ifunc-1)*m.approx.ngrid+1,ifunc*m.approx.ngrid,stateindexplus),polyvec)
        end
        weightvec[m.approx.ninter-i+1] = prod(weighttemp)/prod_sd
    end

    funcapp = dgemv(1.0,funcmat, weightvec)

    zlbintermediate = false  #start with evaluation of 1 poly case (omegapoly and 2nd funcapp irrelevant)
    omegapoly = 1.0 #SINCE ZLBINTERMEDIATE IS FALSE
    endogvar=intermediatedec(m,endogvarm1,currentshockvalues,funcapp,omegapoly,funcapp,zlbintermediate)
    if ( (endogvar[m.endogenous_states[:rm_t]] < 1.0) & (m.approx.zlbswitch == true) ) #zlb case
        zlbintermediate = true
        omegapoly = exp(omegaweight*log(endogvar[m.endogenous_states[:rm_t]])) #now omegapoly and funcapp_plus relevant
        funcapp_plus = dgemv(1.0, funcmatplus,  weightvec)
        endogvar = intermediatedec(m,endogvarm1,currentshockvalues,funcapp,omegapoly,funcapp_plus,zlbintermediate)
        endogvar[9] = 1.0
    end

    return endogvar

end

function decrlin(endogvarm1::Array{Float64},innovations::Array{Float64},m::GHLS,sigma::Array{Float64},pp::Array{Float64})


    # Initilize variables
    nvars = m.approx.nvars
    nexog = m.approx.nexog
    endogvar = Array{Float64}(undef,nvars+nexog)
    xxm1 = zeros(nvars+nexog)
    xx = zeros(nvars+nexog)
    exogpart = Array{Float64}(undef,nvars+nexog)
    endogsteady = [i.value for i in m.steady_state[1:m.approx.nvars + m.approx.nexog]]

    xxm1[1:nvars] = endogvarm1[1:nvars]-endogsteady[1:nvars]
    xxm1[nvars+1:nvars+nexog] = endogvarm1[nvars+1:nvars+nexog]

    xx = dgemv(1.0,pp, xxm1)
    exogpart = dgemv(1.0,sigma, innovations)
    xx = xx + exogpart

    endogvar[1:nvars] = xx[1:nvars]+endogsteady[1:nvars]
    endogvar[nvars+1:nvars+nexog] = xx[nvars+1:nvars+nexog]

    return endogvar
end


function decr_euler(m::GHLS,gridindex::Int64,shockpos::Int64,alphacoeff::Array{Float64})

    #Initilize Variables
    zlbinfo  = m.approx.statezlbinfo[shockpos]
    polyappnew = Array{Float64}(undef,2*m.approx.nfunc)
    endogvar = Array{Float64}(undef,m.approx.nvars+m.approx.nexog)
    endogvarzlb = Array{Float64}(undef,m.approx.nvars+m.approx.nexog)
    endogvarp = Array{Float64}(undef,m.approx.nvars+m.approx.nexog)
    endogvarzlbp = Array{Float64}(undef,m.approx.nvars+m.approx.nexog)
    slopeconcont = Array{Float64}(undef,2*m.approx.nexogcont)
    xgridshock = Array{Float64}(undef,m.approx.nexogcont)
    slopeconxxmsv = Array{Float64}(undef,2*m.approx.nmsv)
    xgridmsv = Array{Float64}(undef,m.approx.nmsv)
    abserror = Array{Float64}(undef,2*m.approx.nfunc)
    ev = Array{Float64}(undef,12)
    exp_eul = Array{Float64}(undef,12)

    currentshockvalues = zeros(m.approx.nexog)
    polyapp = zeros(2*m.approx.nfunc)
    endogvarm1 = zeros(m.approx.nvars+m.approx.nexog)
    exp_var = zeros(12)
    innovations = zeros(m.approx.nexog)

    nmsvplus = m.approx.nmsv+m.approx.nexogcont
    if (m.approx.nexogcont > 0)
        slopeconcont[1:m.approx.nexogcont] = m.approx.slopeconxx[m.approx.nmsv+1:nmsvplus]
        slopeconcont[m.approx.nexogcont+1:2*m.approx.nexogcont] = m.approx.slopeconxx[nmsvplus+m.approx.nmsv+1:2*nmsvplus]
        xgridshock = m.approx.xgrid[m.approx.nmsv+1:nmsvplus,gridindex]
        currentshockvalues[m.approx.nexog-m.approx.nexogcont+1:m.approx.nexog] = msv2xx(xgridshock,m.approx.nexogcont,slopeconcont)
    end

    currentshockvalues[1:m.approx.nexogshock] = m.approx.exoggrid[1:m.approx.nexogshock,shockpos]

    shockpospoly = shockpos + m.approx.ns
    for ifunc in 1:m.approx.nfunc
        polyapp[ifunc] = dot(alphacoeff[(ifunc-1)*m.approx.ngrid+1:ifunc*m.approx.ngrid,shockpos],m.approx.bbt[:,gridindex])
        polyapp[m.approx.nfunc+ifunc] = dot(alphacoeff[(ifunc-1)*m.approx.ngrid+1:ifunc*m.approx.ngrid,shockpospoly],m.approx.bbt[:,gridindex])
    end

    xgridmsv = m.approx.xgrid[1:m.approx.nmsv,gridindex]
    slopeconxxmsv[1:m.approx.nmsv] = m.approx.slopeconxx[1:m.approx.nmsv]
    slopeconxxmsv[m.approx.nmsv+1:2*m.approx.nmsv] = m.approx.slopeconxx[nmsvplus+1:nmsvplus+m.approx.nmsv]
    endogvarm1[1:m.approx.nmsv] = exp.( msv2xx(xgridmsv,m.approx.nmsv,slopeconxxmsv) )
    zlbintermediate = false  #force evaluation of 1 poly case at date t
    omegapoly = 1.0 #SET OMEGAPOLY TO 1 SINCE ZLBINTERMEDIATE IS FALSE

    endogvar=intermediatedec(m,endogvarm1,currentshockvalues,polyapp[1:m.approx.nfunc],
                    omegapoly,polyapp[1:m.approx.nfunc],zlbintermediate)

    if ((zlbinfo != 0) & (m.approx.zlbswitch == true))
        endogvarzlb= intermediatedec(m,endogvarm1,currentshockvalues,polyapp[m.approx.nfunc+1:2*m.approx.nfunc],
                                omegapoly,polyapp[m.approx.nfunc+1:2*m.approx.nfunc],zlbintermediate)
        endogvarzlb[9] = 1.0
    end

    for ss in 1:m.approx.nquad
        innovations[1:m.approx.nexogshock] = m.approx.ghnodes[:,ss]
        if (m.approx.nexogcont > 0)
            innovations[m.approx.nexog-m.approx.nexogcont+1:m.approx.nexog] =  m.approx.ghnodes[m.approx.nexogshock+1:m.approx.nexogshock+m.approx.nexogcont,ss]
        end

        endogvarp=decr(m,endogvar,innovations,alphacoeff) # NOT SURE WHAT TO DO WITH THIS

        techshkp = exp(endogvarp[25])
        invshkp = exp(endogvarp[24])
        ev[1] = endogvarp[10]/(endogvarp[6]*techshkp)

        utilcostp = (m[:rkss]/m[:σ_a])*(exp(m[:σ_a]*(endogvarp[13]-1.0))-1.0) # a(u_t)
        ev[2] = (endogvarp[10]/techshkp)*( (endogvarp[15]*endogvarp[13])-utilcostp+(1.0-m[:δ])*endogvarp[11])
        ev[3] = endogvarp[10]*(endogvarp[18]-1.0)*endogvarp[18]*endogvarp[7]
        ev[4] = (endogvarp[19]-1.0)*endogvarp[19]
        ev[5] = endogvarp[16]/techshkp
        ev[6] = endogvarp[10]*endogvarp[11]*invshkp*(endogvarp[17]-1.0)*endogvarp[17]*endogvarp[17]

        if ((zlbinfo != 0) & (m.approx.zlbswitch == true))
            endogvarzlbp=decr(m,endogvarzlb,innovations,alphacoeff, test)
            ev[7] = endogvarzlbp[10]/(endogvarzlbp[6]*techshkp)
            utilcostp = (m[:rkss]/m[:σ_a])*(exp(m[:σ_a]*(endogvarzlbp[13]-1.0))-1.0)
            ev[8] = (endogvarzlbp[10]/techshkp)*( (endogvarzlbp[15]*endogvarzlbp[13])-utilcostp+(1-params[16])*endogvarzlbp[11] )
            ev[9] = endogvarzlbp[10]*(endogvarzlbp[18]-1.0)*endogvarzlbp[18]*endogvarzlbp[7]
            ev[10] = (endogvarzlbp[19]-1.0)*endogvarzlbp[19]
            ev[11] = endogvarzlbp[16]/techshkp
            ev[12] = endogvarzlbp[10]*endogvarzlbp[11]*invshkp*(endogvarzlbp[17]-1.0)*endogvarzlbp[17]*endogvarzlbp[17]
        else
            ev[7:12] = ev[1:6]
        end
        exp_var = exp_var + m.approx.ghweights[ss]*ev
    end

    liqshk = exp(endogvar[23]) # Change to m.endogenous_states[:b_t]?
    invshk = exp(endogvar[24]) # Change to m.endogenous_states[:μ_t]?
    ep = m[:ϵ_p] # Is this constant elasticity of demand?

    exp_eul[1] =  (m[:β]/m[:gz])*liqshk*endogvar[9]*exp_var[1] # Change endogvar?
    exp_eul[2] =  (m[:β]/m[:gz])*exp_var[2]/endogvar[10]
    exp_eul[3] =  m[:β]*exp_var[3]/(endogvar[10]*endogvar[7])+(m[:ϵ_p]/m[:ϕ_p])*(endogvar[14]-(m[:ϵ_p]-1.0)/m[:ϵ_p])
    exp_eul[4] = m[:β]*exp_var[4]+(m[:ϵ_w]/m[:ϕ_w])*endogvar[10]*endogvar[12]*
           ( (m[:ψ_L]*endogvar[12]^m[:σ_L]/endogvar[10])-((m[:ϵ_w]-1.0)/m[:ϵ_w])*endogvar[4] )
    exp_eul[5] = exp_var[5]
    exp_eul[6] = (m[:β]/m[:gz])*exp_var[6]/endogvar[10] - 0.5*endogvar[11]*invshk*(endogvar[17]-1.0)*(endogvar[17]-1.0)

    polyappnew[1] = log(exp_eul[1])
    polyappnew[2] = log(exp_eul[2])
    polyappnew[3] = exp_eul[3]
    polyappnew[4] = exp_eul[4]
    polyappnew[5] = log(exp_eul[5])
    polyappnew[6] = exp_eul[6]
    polyappnew[7] = log( 1.0 + (1/m[:σ_a])*log(endogvar[15]/m[:rkss]) )

    if ((zlbinfo != 0) & (m.approx.zlbswitch == true))
        exp_eul[7] =  (m[:β]/m[:gz])*liqshk*endogvarzlb[9]*exp_var[7]
        exp_eul[8] = (m[:β]/m[:gz])*exp_var[8]/endogvarzlb[10]
        exp_eul[9] =  m[:β]*exp_var[9]/(endogvarzlb[10]*endogvarzlb[7])+(m[:ϵ_p]/m[:ϕ_p])*( endogvarzlb[14]-(m[:ϵ_p]-1.0)/m[:ϵ_p])
        exp_eul[10] = m[:β]*exp_var[10]+(m[:ϵ_w]/m[:ϕ_w])*endogvarzlb[10]*endogvarzlb[12]*
           ( (m[:ψ_L]*endogvarzlb[12]^m[:σ_L]/endogvarzlb[10])-((m[:ϵ_w]-1.0)/m[:ϵ_w])*endogvarzlb[4])
        exp_eul[11] = exp_var[11]
        exp_eul[12] = (m[:β]/m[:gz])*exp_var[12]/endogvarzlb[10] - 0.5*endogvarzlb[11]*invshk*(endogvarzlb[17]-1.0)*(endogvarzlb[17]-1.0)

        polyappnew[8] = log(exp_eul[7])
        polyappnew[9] = log(exp_eul[8])
        polyappnew[10] = exp_eul[9]
        polyappnew[11] = exp_eul[10]
        polyappnew[12] = log(exp_eul[11])
        polyappnew[13] = exp_eul[12]
        polyappnew[14] = log( 1.0 + (1/m[:σ_a])*log(endogvarzlb[15]/m[:rkss]) )
    else
        exp_eul[7:12] = exp_eul[1:6]
        polyappnew[8:14] = polyappnew[1:7]
    end

    errsum = 0.0
    errmax = 0.0
    imax = 0

    for ifunc in 1:2*m.approx.nfunc
        abserror[ifunc] = abs(polyappnew[ifunc]-polyapp[ifunc])
        errsum = errsum + abserror[ifunc]
        if (abserror[ifunc] > errmax)
            errmax = abserror[ifunc]
            imax = copy(ifunc)
        end
    end

    errsum = errsum/(2*m.approx.nfunc)

    return polyappnew, errsum

end


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

function get_shockdetails(m::GHLS)

    #Input
    number_shock_values = m.approx.number_shock_values
    ngridshocks = m.approx.nshockgrid
    exogvarinfo = m.approx.exogvarinfo

    #Initilize Variables
    shockbounds = Array{Float64}(undef,m.approx.nexogshock,2)
    shockdistance = Array{Float64}(undef,m.approx.nexogshock)
    currentshockindex = Array{Float64}(undef,m.approx.nexogshock)
    nshocksum_vec = Array{Float64}(undef,m.approx.nexogshock)
    exoggrid = zeros(m.approx.nexog-m.approx.nexogcont,m.approx.ns)

    rhovec = [m[:ρ_η].value,m[:ρ_μ].value,0.0,m[:ρ_int].value,m[:ρ_g].value, 0.0] # Unsure about rhoa
    sigmavec = [m[:σ_η].scaledvalue,m[:σ_μ].scaledvalue,m[:σ_Z].scaledvalue,m[:σ_R].scaledvalue,m[:σ_g].scaledvalue,0.0] # Unsure about sdeva

    nshocksum = 0
    shockvalues = zeros(number_shock_values)

    for i in 1:m.approx.nexogshock
        xshock = Array{Float64}(undef,ngridshocks[i])
        shockdistance[i],xshock=finite_grid(ngridshocks[i],rhovec[i],sigmavec[i])
        shockbounds[i,1] = xshock[1]
        shockbounds[i,2] = xshock[ngridshocks[i]]
        shockvalues[nshocksum+1:nshocksum+ngridshocks[i]] = xshock
        nshocksum = nshocksum + ngridshocks[i]
    end

    for ss in 1:m.approx.ns
        currentshockindex = exogvarinfo[:,ss]
        nshocksum_vec = zeros(Int64,m.approx.nexogshock)
        nall = ngridshocks[1]
        shockpos = currentshockindex[1]
        exoggrid[1,ss] = shockvalues[currentshockindex[1]]
        for i in 2:m.approx.nexogshock
            nshocksum_vec[i] = nshocksum_vec[i-1] + ngridshocks[i-1]
            shockpos = shockpos + nall*(currentshockindex[i]-1)
            nall = nall*ngridshocks[i]
            exoggrid[i,ss] = shockvalues[nshocksum_vec[i] + currentshockindex[i]]
        end
    end

    return exoggrid,shockbounds,shockdistance

end

function calc_premium(m::GHLS,endogvar::Array{Float64},alphacoeff::Array{Float64})

    # Initilize Variables
    endogvarp = Array{Float64}(undef,m.approx.nvars+m.approx.nexog)
    innovations=zeros(m.approx.nexog)

    exp_var = 0.0
    for ss in 1:m.approx.nquad
        innovations[1:m.approx.nexogshock] = m.approx.ghnodes[:,ss]
        endogvarp=decr(m,endogvar,innovations,alphacoeff, false)
        techshkp = exp(endogvarp[25])
        invshkp = exp(endogvarp[4])
        utilcostp = (m[:rkss]/m[:σ_a])*(exp(m[:σ_a]*(endogvarp[13]-1.0))-1.0)
        ev = ( (endogvarp[15]*endogvarp[13])-utilcostp+(1.0-m[:δ])*endogvarp[11])*endogvarp[6]
        exp_var = exp_var + m.approx.ghweights[ss]*ev
    end

    premium = exp_var/(endogvar[11]*endogvar[9])

    return premium

end
