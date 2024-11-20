"""
This code replicates the matlab code provided in make_irfs_taylor.m to obtain dynamic solutions to the model after already obtaining the gx and hx from solving the system using a complex schur decomposition
"""
function compute_irfs_taylor(m::OnionModel, gx::AbstractMatrix, hx::AbstractMatrix, U11i::AbstractMatrix, C::AbstractMatrix)

#=
I need NK, NK_norm, T
Also definitions of matrices in the function heading?

I know NK_Norm is just NK -1
=#
    T = get_setting(m, :T)
    NK = get_setting(m, :n_predetermined_variables)
    NK_norm = NK - 1
    n = get_setting(m, :n_sectors)
    tau0 = 1 #normalization, used in kanzig_irfs.m

    numer = -transpose(m[:gam].value) * ((eye(n) - get_setting(m, :inpshare))\m[:taxshare].value)
    denom = transpose(m[:gam].value) * ((eye(n) - get_setting(m, :inpshare))\m[:labshare].value)
    dcstar_dτ = numer/denom

    X = zeros(NK_norm, T+25) #Add 24 extra months to get 2 year rate
    X0 = [zeros(n+2, 1); tau0]
    X[:, 1] = X0

    for t in 1:T+24
        X[:, t+1] = hx*X[:, t]
    end

    Y = gx*X

    X = round.(X, digits=12)
    Y = round.(Y, digits=12)

    ct     = vcat(0,Y[m.endogenous_states[:c] - NK,1:T-1])                 #[0 Y[m.endogenous_states[:c] - NK,1:T-1]]
    wt     = X[m.endogenous_states[:lw],1:T]
    piit   = [zeros(n,1) Y[(m.endogenous_states[:π_1] - NK) : (m.endogenous_states[Symbol("π_$n")] - NK) ,1:T-1]]
    pict   =  vcat(0,Y[m.endogenous_states[:πc] - NK,1:T-1])                        #[0 Y[m.endogenous_states[:πc] - NK,1:T-1]]
    piwt   =  vcat(0,Y[m.endogenous_states[:πw] - NK,1:T-1])                                      #[0 Y[m.endogenous_states[:πw] - NK,1:T-1]]
    taut   = vcat(0,X[m.endogenous_states[:τ],1:T-1])                                  #[0 X[m.endogenous_states[:τ],1:T-1]]
    tauhatt = vcat(0,X[m.endogenous_states[:τ],1:T-1])                                 #[0 X[m.endogenous_states[:τ],1:T-1]] # same
    snormt = X[m.endogenous_states[:ls_2]:m.endogenous_states[Symbol("ls_$n")],1:T]
    st     = C*snormt
    it     = X[m.endogenous_states[:li],1:T+24]



# now solve for other variables

    # first make \Theta^ij_t
    cit = transpose(repeat(ct,1,n)) - (m[:ζ].value)*st # sectoral final consumption #WAS CT,N,1

    sIt = get_setting(m, :ω_tilde)*st           #(m[:ω_tilde].value)*st # indices of relative price of intermediate inputs
    # (different for each sector)

    sEt = get_setting(m, :ωE_tilde)*st            #(m[:ωE_tilde].value)*st
    sNt = get_setting(m, :ωN_tilde)*st            #(m[:ωN_tilde].value)*st
#We need to take a stand on what carbon taxes are in steady state
# for now, let's assume they are zero (so \widetilde{M}^i=M^i)
    xt = zeros(n,T)

    for t=1:T
        #=
     note, I define THETA as in climate_multisector.pdf: it pertains to
     industry i's use of the product of industry j. So what we need in
     this market clearing condition is its transpose, THETA'.
     note also we've replaced consumption ct w real wages wt
        =#

        #Note this is Capital Theta, \Theta
        Θ = (m[:ξ].value*st[:,t]') .+ (m[:η].value * (ones(size(m[:totintshare].value)) - m[:totintshare].value) .- m[:ν].value).*sIt[:,t] + (m[:ν].value - m[:ξ].value)*sEt[:,t].*m[:energy].value' + (m[:ν].value - m[:ξ].value)*sNt[:,t].*(ones(size(m[:energy].value)) - m[:energy].value)' .- (m[:η].value * m[:labshare].value * wt[t])
        px_Θ = sum(get_setting(m, :IO2).*(Θ'),dims=2)
        xt[:,t] = (eye(n) - get_setting(m, :IO2))\(-px_Θ + m[:pc_px].value.*cit[:,t])
    end
    et = m[:ei].value'*xt  # log-deviation of aggregate emissions

    irf = Dict()
    #Definitions of IRFs below:

    irf["Y"] = Y
    irf["X"] = X
    #irf["THETA"] = Θ
    irf["ct"] = ct
    irf["piit"] = piit
    irf["pict"] = pict
    irf["taut"] = taut
    irf["st"] = st
    irf["cit"] = cit
    irf["xt"] = xt
    irf["et"] = et
    irf["cstar"] = dcstar_dτ*tauhatt
    irf["cgap"] = irf["ct"] - irf["cstar"]
    irf["pit_core"] = (m[:gam_core].value)'*irf["piit"]
    irf["pit_food"] = (m[:gam_food].value)'*irf["piit"]
    irf["pit_energy"] = (m[:gam_energy].value)'*irf["piit"]
    irf["pit_services"] = (m[:gam_services].value)' * irf["piit"]
    irf["pit_coreservices"] = (m[:gam_coreservices].value)' * irf["piit"]
    irf["pit_goods"] = (m[:gam_goods].value)'*irf["piit"]
    irf["pit_coregoods"] = (m[:gam_coregoods].value)'*irf["piit"]
    irf["piwt"] = piwt
    irf["wt"] = wt

    # variables needed for kanzig irfs

    irf["cumpt_energy"] = cumsum(irf["pit_energy"], dims =2)  # % energy price level
    irf["cumpt_cpi"]    = cumsum(irf["pict"], dims =1)        # % consumer price level
    irf["cumpt_core"]   = cumsum(irf["pit_core"], dims =2)    # % core cpi level
    irf["cumpt_oil"]    = cumsum(irf["piit"][Int(m[:oil].value),:], dims =1) #% oil price

    irf["cumpt_oilr"] = irf["cumpt_oil"] - irf["cumpt_cpi"]
    irf["cumpt_services"] = cumsum(irf["pit_services"], dims=2)

    irf["it_2yr"] = zeros(1,T)
    for t=2:T
        irf["it_2yr"][t] = sum(it[t:t+23])/2
    end
    irf["it"] = it[1:T]
    irf["ann_it"] = 12*it[1:T]

    irf["cumpt_food"] = cumsum(irf["pit_food"],dims=1)

    return irf

end
