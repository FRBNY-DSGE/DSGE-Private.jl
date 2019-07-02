#include("GHLS.jl")

#module model_details

#    using LinearAlgebra

#    export decr, decrlin, decr_euler, get_shockdetails, calc_premium, msv2xx, exogposition

function msv2xx(msv, nmsv, slopeconmsv)

    #Input
    nmsv :: Int
    msv :: Array{Float64}
    slopeconmsv :: Array{Float64}

    return slopeconmsv[1:nmsv] .* msv + slopeconmsv[nmsv+1:2*nmsv]

function exogposition(exogvec, nrvec, nlength)

    # Input
    nlength :: Int64
    nrvec :: Array{Int64}
    exogvec :: Array{Int64}

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

function intermediatedec(m,nvars,nexog,endogvarm1,currentshockvalues,polyvar,omegapoly,polyvarplus,zlbintermediate)

    #Input
    m :: GHLS
    nvars :: Int64
    nexog :: Int64
    endogvarm1 :: Array{Float64}
    currentshockvalues :: Array{Float64}
    polyvar :: Array{Float64}
    omegapoly :: Float64
    polyvarplus :: Array{Float64}
    zlbintermediate :: Bool

    #Initilize Variables
    endogvar=Array{Float64}(undef,nvars+nexog,1)

    invshk = exp(currentshockvalues[m.exogenous_shocks[:μ_sh]])
    techshk = exp(currentshockvalues[m.exogenous_shocks[:ztil_sh]])
    rrshk = currentshockvalues[m.exogenous_shocks[:rm_sh]]
    gss = 1.0/(1.0-m[:shrgy])
    gshk = exp( log(gss) + currentshockvalues[m[:g_sh]] )
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
    cquad_vi = bi/(qq*invshk)-(1.0-qq*invshk)/(m[:ϕ_I]*qq*invshk)

    vi = 0.5*(1.0+sqrt(1.0+4.0*cquad_vi))

    inv = vi*invm1/techshk
    aayy = 1.0/gshk-(m[:ϕ_p]/2.0)*(vp-1.0)*(vp-1.0)
    rkss = m[:gz]/m[:β]-1.0+m[:δ]
    utilcost = (rkss/m[:σ_a])*(exp(m[:σ_a]*(util-1.0))-1.0)
    gdp = (1.0/aayy)*( cc+inv + utilcost*(capm1/(m[:gz]*techshk)) )
    rw = rwm1*dw/(m[:gz]*techshk*dp)
    lab = gdp^(1.0/(1.0-m[:α]))*(util*capm1/(m[:gz]*techshk))^(m[:α]/(m[:α]-1.0))/ashk
    xhp = m[:α]*log(util) + (1-m[:α])*(log(lab)-m[:labss]) # Changed llabss to m[:labss]

    lrss = log(m[:gz]*m[:π_bar]/m[:β])
    notr = exp(lrss+m[:ρ_R]*(log(notrm1)-lrss) + (1.0-m[:ρ_R])*(m[:γ_π]*log(dp/m[:π_bar]) + m[:γ_g]*log(gdp*techshk/gdpm1) + m[:γ_xhp]*xhp ) + rrshk)

    mc = rw*lab/((1.0-m[:α])*gdp)
    rentalk = (m[:α]/(1.0-m[:α]))*(rw*lab*m[:gz]*techshk/(util*capm1))
    cap = (1.0-[:δ])*(capm1/(m[:gz]*techshk)) + invshk*inv*(1.0- (m[:ϕ_I]/2.0)*(vi-1.0)*(vi-1.0) )
    nomr = copy(notr) # copy as to no make it a pointer

    #all variables are in levels except for shocks
    #shocks are in log-level deviations from SS
    #(1) cap, (2) cc, (3) inv, (4) rw, (5) notr, (6) dp, (7) gdp, (8) xhp, (9) nomr, (10) lam, (11) qq,
    #(12) lab, (13) util, (14) mc, (15) rentalk, (16) muc, (17) vi, (18) vp, (19) vw, (20) dw, (21) bc, (22) bi,
    #(23) liqshk, (24) invshk, (25) techshk, (26) rrshk, (27) gshk, (28) ashk

    endogvar = vcat([cap, cc, inv, rw, notr, dp, gdp, xhp, nomr, lam, qq, lab, util, mc, rentalk, muc, vi, vp, vw, dw, bc, bi], currentshockvalues) #Concatenate two arrays

    return endogvar
end

function decr(m,endogvarm1,innovations,poly,alphacoeff)

    #Input
    m :: GHLS
    endogvarm1 :: Array{Float64}
    innovations :: Array{Float64}
    poly :: SmolyakApproximation
    alphacoeff :: Array{Float64}

    #Initilize Variables
    endogvar=Array{Float64}(undef,poly[:nvars]+poly[:nexog],1)
    shockindexall=ones(Int64,poly[:nexog]-poly[:nexogcont])
    shockindex=Array{Int64}(undef,poly[:nexogshock])
    shockindex_inter=Array{Int64}(undef,poly[:nexogshock])
    lmsv=Array{Float64}(undef,poly[:nmsv]+poly[:nexogcont])
    currentshockvalues=Array{Float64}(undef,poly[:nexog])
    funcmatplus=Array{Float64}(undef,poly[:nfunc],poly[:ninter])
    weighttemp=Array{Float64}(undef,poly[:nexogshock])
    funcapp=Array{Float64}(undef,poly[:nfucn])
    funcapp_plus=Array{Float64}(undef,poly[:nfunc])
    xx=Array{Float64}(undef,poly[:nmsv])
    polyvec=Array{Float64}(undef,poly[:ngrid])

    omegaweight = 100000.0

    nmsvplus = poly[:nmsv]+poly[:nexogcont]

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
    currentshockvalues[1] = m[:ρ_η]*m[:b_t] + m[:σ_η]*innovations[1]
    currentshockvalues[2] = m[:ρ_μ]*m[:μ_t] + m[:σ_μ]*innovations[2]
    currentshockvalues[3] = m[:σ_Z]*innovations[3]
    currentshockvalues[4] = m[:ρ_int]*m[:mon_t] + m[:σ_R]*innovations[4]
    currentshockvalues[5] = m[:ρ_g]*m[:g_t] + m[:σ_g]*innovations[5]
    currentshockvalues[6] = 0.0*endogvarm1[28] + 0.0*innovations[6] # Which one of the thee is this? Is it unk_t?

    #find position of shocks for interpolation
    @simd for i in 1:poly[:nexogshock]
        if (currentshockvalues[i] < poly[:shockbounds][i,1])   #extrapolation below
            shockindex[i] = 1
        elseif (currentshockvalues[i] > poly[:shockbounds][i,2])  #extrapolation above
            shockindex[i] = floor(Int64,(poly[:shockbounds][i,2]-poly[:shockbounds][i,1])/poly[:shockdistance][i] )
        else
            shockindex[i] = floor(Int64,1.0+(currentshockvalues[i]-poly[:shockbounds][i,1])/poly[:shockdistance][i])  #interpolation case
        end
    end

    #loop for interpolating between shocks
    shockindexall[1:poly[:nexogshock]] = shockindex
    stateindex0 = exogposition(shockindexall,poly[:nshockgrid],poly[:nexog]-poly[:nexogcont])
    shockindexall[1:poly[:nexogshock]] = shockindex + poly[:interpolatemat][:,poly[:ninter]]
    stateindex1 = exogposition(shockindexall,poly[:nshockgrid],poly[:nexog]-poly[:nexogcont])
    funcmat = zeros(poly[:nfunc],poly[:ninter])
    weightvec = zeros(poly[:ninter],1)
    lmsv[1:poly[:nmsv]] = log.(endogvarm1[1:poly[:nmsv]])
    if (poly[:nexogcont] > 0)
        lmsv[poly[:nmsv]+1:poly[:nmsv]+poly[:nexogcont]] = currentshockvalues[poly[:nexog]-poly[:nexogcont]+1:poly[:nexog]]
    end
    xx = msv2xx(lmsv,nmsvplus,poly[:slopeconmsv])

    polyvec = smolyakpoly(nmsvplus,poly[:ngrid],poly[:nindplus],poly[:indplus],xx)
    prod_sd = prod(poly[:shockdistance])
    for i in 1:poly[:ninter] # @inbounds, stops the bounds check
        @fastmath @inbounds @simd for j in 1:poly[:nexogshock]
            shockindexall[j] = shockindex[j] + poly[:interpolatemat][j,i]
            weighttemp[j]=(1-poly[:interpolatemat][j,i])*(currentshockvalues[j]-poly[:exoggrid][j,stateindex0]) + (poly[:interpolatemat][j,i])*(poly[:exoggrid][j,stateindex1]-currentshockvalues[j])
        end
        stateindex = exogposition(shockindexall,poly[:nshockgrid],poly[:nexog]-poly[:nexogcont])
        stateindexplus = stateindex+poly[:ns]
        @fastmath @inbounds @simd for ifunc in 1:poly[:nfunc]
            funcmat[ifunc,i] = dot(alphacoeff[(ifunc-1)*poly[:ngrid]+1:ifunc*poly[:ngrid],stateindex],polyvec)
            funcmatplus[ifunc,i] = dot(alphacoeff[(ifunc-1)*poly[:ngrid]+1:ifunc*poly[:ngrid],stateindexplus],polyvec)
        end
        weightvec[poly[:ninter]-i+1] = prod(weighttemp)/prod_sd
    end

    funcapp = funcmat * weightvec

    zlbintermediate = false  #start with evaluation of 1 poly case (omegapoly and 2nd funcapp irrelevant)
    omegapoly = 1.0 #SINCE ZLBINTERMEDIATE IS FALSE
    endogvar=intermediatedec(m,poly[:nparams],poly[:nvars],poly[:nexog],poly[:nfunc],endogvarm1,currentshockvalues,funcapp,m[:labss],omegapoly,funcapp,zlbintermediate)

    if ( (endogvar[m.endogenous_states[:rm_t]] < 1.0) & (poly[:zlbswitch] == true) ) #zlb case
        zlbintermediate = true
        omegapoly = exp(omegaweight*log(endogvar[m.endogenous_states[:rm_t]])) #now omegapoly and funcapp_plus relevant
        funcapp_plus = funcmatplus * weightvec
        endogvar = intermediatedec(m,poly[:nvars],poly[:nexog],endogvarm1,currentshockvalues,funcapp,omegapoly,funcapp_plus,zlbintermediate)
        endogvar[9] = 1.0
    end

    return endogvar

end

function decrlin(endogvarm1,innovations,m::GHLS)

    # Input
    innovations :: Array{Float64}
    endogvarm1 :: Array{Float64}

    # Initilize variables
    nvars = m.approx[:nvars]
    nexog = m.approx[:nexog]
    endogvar = Array{Float64}(undef,nvars)
    xxm1 = zeros(nvars)
    xx = zeros(nvars)
    exogpart = Array{Float64}(undef,nvars)

    xxm1[1:nvars-nexog] = endogvarm1[1:nvars-nexog]-m.approx[:endogsteady][1:nvars-nexog]
    xxm1[nvars-nexog+1:nvars] = endogvarm1[nvars-nexog+1:nvars]

    Γ0, Γ1, C, Ψ, Π = eqcond(m)

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

    QQQ = zeroes(8, 8)
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

    xx = pp * xxm1
    exogpart = sigma * innovations
    xx = xx + exogpart

    endogvar[1:nvars-nexog] = xx[1:nvars-nexog]+m.approx[:endogsteady][1:nvars-nexog]
    endogvar[nvars-nexog+1:nvars] = xx[nvars-nexog+1:nvars]

    return endogvar
end


function decr_euler(m,shockpos,poly,alphacoeff,zlbinfo)

    #Input
    m :: GHLS
    poly :: SmolyakApproximation
    shockpos :: Int64
    gridindex :: Int64
    bbt :: Array{Float64}
    xgrid :: Array{Float64}
    alphacoeff :: Array{Float64}
    slopeconxx :: Array{Float64}
    zlbinfo :: Int64

    #Initilize Variables
    polyappnew = Array{Float64}(undef,2*poly[:nfunc])
    endogvar = Array{Float64}(undef,poly[:nvars]+poly[:nexog],1)
    endogvarzlb = Array{Float64}(undef,poly[:nvars]+poly[:nexog],1)
    endogvarp = Array{Float64}(undef,poly[:nvars]+poly[:nexog],1)
    endogvarzlbp = Array{Float64}(undef,poly[:nvars]+poly[:nexog],1)
    slopeconcont = Array{Float64}(undef,2*poly[:nexogcont],1)
    xgridshock = Array{Float64}(undef,poly[:nexogcont],1)
    slopeconxxmsv = Array{Float64}(undef,2*poly[:nmsv],1)
    xgridmsv = Array{Float64}(undef,poly[:nmsv],1)
    abserror = Array{Float64}(undef,2*poly[:nfunc],1)
    ev = Array{Float64}(undef,12,1)
    exp_eul = Array{Float64}(undef,12,1)

    currentshockvalues = zeros(poly[:nexog])
    polyapp = zeros(2*poly[:nfunc],1)
    endogvarm1 = zeros(poly[:nvars]+poly[:nexog],1)
    exp_var = zeros(12)
    innovations = zeros(poly[:nexog])

    nmsvplus = poly[:nmsv]+poly[:nexogcont]
    if (poly[:nexogcont] > 0)
        slopeconcont[1:poly[:nexogcont]] = poly[:slopeconxx][poly[:nmsv]+1:nmsvplus]
        slopeconcont[poly[:nexogcont]+1:2*poly[:nexogcont]] = poly[:slopeconxx][nmsvplus+poly[:nmsv]+1:2*nmsvplus]
        xgridshock = poly[:xgrid][poly[:nmsv]+1:nmsvplus,poly[:gridindex]]
        currentshockvalues[poly[:nexog]-poly[:nexogcont]+1:poly[:nexog]] = msv2xx(xgridshock,poly[:nexogcont],slopeconcont)
    end

    currentshockvalues[1:poly[:nexogshock]] = poly[:exoggrid][1:poly[:nexogshock],shockpos]
    shockpospoly = shockpos + poly[:ns]
    for ifunc in 1:poly[:nfunc]
        polyapp[ifunc] = dot(alphacoeff[(ifunc-1)*poly[:ngrid]+1:ifunc*poly[:ngrid],shockpos],poly[:bbt][:,poly[:gridindex]])
        polyapp[poly[:nfunc]+ifunc] = dot(alphacoeff[(ifunc-1)*poly[:ngrid]+1:ifunc*poly[:ngrid],shockpospoly],poly[:bbt][:,poly[:gridindex]])
    end

    xgridmsv = poly[:xgrid][1:poly[:nmsv],poly[:gridindex]]
    slopeconxxmsv[1:poly[:nmsv]] = poly[:slopeconxx][1:poly[:nmsv]]
    slopeconxxmsv[poly[:nmsv]+1:2*poly[:nmsv]] = poly[:slopeconxx][nmsvplus+1:nmsvplus+poly[:nmsv]]
    endogvarm1[1:poly[:nmsv]] = exp.( msv2xx(xgridmsv,poly[:nmsv],slopeconxxmsv) )
    zlbintermediate = false  #force evaluation of 1 poly case at date t
    omegapoly = 1.0 #SET OMEGAPOLY TO 1 SINCE ZLBINTERMEDIATE IS FALSE

    endogvar=intermediatedec(m,poly[:nvars],poly[:nexog],
                    endogvarm1,currentshockvalues,polyapp[1:poly[:nfunc]],
                    omegapoly,polyapp[1:poly[:nfunc]],zlbintermediate)

    if ((zlbinfo != 0) & (poly[:zlbswitch] == true))
        endogvarzlb= intermediatedec(m,poly[:nvars],poly[:nexog],
                                endogvarm1,currentshockvalues,polyapp[poly[:nfunc]+1:2*poly[:nfunc]],
                                omegapoly,polyapp[poly[:nfunc]+1:2*poly[:nfunc]],zlbintermediate)
        endogvarzlb[9] = 1.0
    end

    for ss in 1:poly[:nquad]
        innovations[1:poly[:nexogshock]] = poly[:ghnodes][:,ss]
        if (poly[:nexogcont] > 0)
            innovations[poly[:nexog]-poly[:nexogcont]+1:poly[:nexog]] =  poly[:ghnodes][poly[:nexogshock]+1:poly[:nexogshock]+poly[:nexogcont],ss]
        end
        endogvarp=decr(m,endogvar,innovations,poly,alphacoeff) # NOT SURE WHAT TO DO WITH THIS
        techshkp = exp(endogvarp[25])
        invshkp = exp(endogvarp[24])
        ev[1] = endogvarp[10]/(endogvarp[6]*techshkp)

        utilcostp = (m[:rkss]/m[:σ_a])*(exp(m[:σ_a]*(endogvarp[13]-1.0))-1.0) # a(u_t)
        ev[2] = (endogvarp[10]/techshkp)*( (endogvarp[15]*endogvarp[13])-utilcostp+(1.0-m[:δ])*endogvarp[11])
        ev[3] = endogvarp[10]*(endogvarp[18]-1.0)*endogvarp[18]*endogvarp[7]
        ev[4] = (endogvarp[19]-1.0)*endogvarp[19]
        ev[5] = endogvarp[16]/techshkp
        ev[6] = endogvarp[10]*endogvarp[11]*invshkp*(endogvarp[17]-1.0)*endogvarp[17]*endogvarp[17]

        if ((zlbinfo != 0) & (poly[:zlbswitch] == true))
            endogvarzlbp=decr(m,endogvarzlb,innovations,poly,alphacoeff)
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
        exp_var = exp_var + poly[:ghweights][ss]*ev
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

    if ((zlbinfo != 0) & (poly[:zlbswitch] == true))
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

    for ifunc in 1:2*poly[:nfunc]
        abserror[ifunc] = abs(polyappnew[ifunc]-polyapp[ifunc])
        errsum = errsum + abserror[ifunc]
        if (abserror[ifunc] > errmax)
            errmax = abserror[ifunc]
            imax = copy(ifunc)
        end
    end
    #errsum = errsum/convert(Float64,(2*poly.nfunc))
    errsum = errsum/(2*poly[:nfunc])

    return polyappnew, errsum, errmax

end


function finite_grid(n,rho,sigmaep)

    #Input
    n :: Int64
    rho :: Float64
    sigmaep :: Float64

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


function get_shockdetails(m,poly::SmolyakApproximation,number_shock_values,ngridshocks,exogvarinfo)

    #Input
    m :: GHLS
    number_shock_values :: Int64
    ngridshocks :: Array{Int64}
    exogvarinfo :: Array{Int64}

    #Initilize Variables
    shockbounds = Array{Float64}(undef,poly[:nexogshock],2)
    shockdistance = Array{Float64}(undef,poly[:nexogshock],1)
    currentshockindex = Array{Float64}(undef,poly[:nexogshock],1)
    nshocksum_vec = Array{Float64}(undef,poly[:nexogshock],1)
    exoggrid = zeros(poly[:nexog]-poly[:nexogcont],poly[:ns])

    rhovec = [m[:ρ_η],m[:ρ_μ],0.0,m[:ρ_int],m[:ρ_g], 0.0] # Unsure about rhoa
    sigmavec = [m[:σ_η],m[:σ_μ],m[:σ_Z],m[:σ_R],m[:σ_g],0.0] # Unsure about sdeva

    nshocksum = 0
    shockvalues = zeros(number_shock_values)

    for i in 1:nexogshock
        xshock = Array{Float64}(undef,ngridshocks[i])
        shockdistance[i],xshock=finite_grid(ngridshocks[i],rhovec[i],sigmavec[i])
        shockbounds[i,1] = xshock[1]
        shockbounds[i,2] = xshock[ngridshocks[i]]
        shockvalues[nshocksum+1:nshocksum+ngridshocks[i]] = xshock
        nshocksum = nshocksum + ngridshocks[i]
    end

    for ss in 1:ns
        currentshockindex = exogvarinfo[:,ss]
        nshocksum_vec = zeros(Int64,poly[:nexogshock],1)
        nall = ngridshocks[1]
        shockpos = currentshockindex[1]
        exoggrid[1,ss] = shockvalues[currentshockindex[1]]
        for i in 2:nexogshock
            nshocksum_vec[i] = nshocksum_vec[i-1] + ngridshocks[i-1]
            shockpos = shockpos + nall*(currentshockindex[i]-1)
            nall = nall*ngridshocks[i]
            exoggrid[i,ss] = shockvalues[nshocksum_vec[i] + currentshockindex[i]]
        end
    end

    return exoggrid,shockbounds,shockdistance

end

function calc_premium(m,endogvar,poly,alphacoeff)

    # Input
    poly :: SmolyakPolynomial
    m :: GHLS
    endogvar :: Array{Float64}
    alphacoeff :: Array{Float64}

    # Initilize Variables
    endogvarp = Array{Float64}(undef,poly[:nvars]+poly[:nexog],1)
    innovations=zeros(poly[:nexog],1)

    exp_var = 0.0
    for ss in 1:poly[:nquad]
        innovations[1:poly[:nexogshock]] = poly[:ghnodes][:,ss]
        endogvarp=decr(m,endogvar,innovations,poly,alphacoeff)
        techshkp = exp(endogvarp[25])
        invshkp = exp(endogvarp[4])
        utilcostp = (m[:rkss]/m[:σ_a])*(exp(m[:σ_a]*(endogvarp[13]-1.0))-1.0)
        ev = ( (endogvarp[15]*endogvarp[13])-utilcostp+(1.0-m[:δ])*endogvarp[11])*endogvarp[6]
        exp_var = exp_var + poly[:ghweights][ss]*ev
    end

    premium = exp_var/(endogvar[11]*endogvar[9])

    return premium

end


end
