function parse_irfs(m::OnionModel, states::AbstractArray, pseudo::AbstractArray, obs::AbstractArray, C::AbstractMatrix; add_addl_zero::Bool = true)

    T = get_setting(m, :T)
    NK = get_setting(m, :n_back_states) + get_setting(m, :n_exo_states)
    @show NK
    n = get_setting(m, :n_sectors)
    tau0 = 1 #normalization, used in kanzig_irfs.m

    numer = -transpose(m[:gam].value) * ((eye(n) - get_setting(m, :inpshare))\m[:taxshare].value)
    denom = transpose(m[:gam].value) * ((eye(n) - get_setting(m, :inpshare))\m[:labshare].value)
    dcstar_dτ = numer/denom

    #Option to not add 0 before state responses
if !add_addl_zero
    ct     =  states[m.endogenous_states[:c],1:T]           #[0 Y[m.endogenous_states[:c] - NK,1:T-1]]
    wt     = states[m.endogenous_states[:lw],1:T]

    piit   = states[(m.endogenous_states[:π_1]) : (m.endogenous_states[Symbol("π_$n")]) ,1:T]
    pict   =  states[m.endogenous_states[:πc],1:T]                     #[0 Y[m.endogenous_states[:πc] - NK,1:T-1]]
    piwt   = states[m.endogenous_states[:πw],1:T]                                  #[0 Y[m.endogenous_states[:πw] - NK,1:T-1]]
    taut   = states[m.endogenous_states[:τ],1:T]                                   #[0 X[m.endogenous_states[:τ],1:T-1]]
    tauhatt = states[m.endogenous_states[:τ],1:T]                               #[0 X[m.endogenous_states[:τ],1:T-1]] # same
    snormt = states[m.endogenous_states[:ls_2]:m.endogenous_states[Symbol("ls_$(n)")],1:T]
    st     = C*snormt
    it     = states[m.endogenous_states[:li],1:T+24]
else
    ct     =  vcat(0,states[m.endogenous_states[:c],1:T-1])           #[0 Y[m.endogenous_states[:c] - NK,1:T-1]]
    wt     = states[m.endogenous_states[:lw],1:T]

    piit   = [zeros(n,1) states[(m.endogenous_states[:π_1]) : (m.endogenous_states[Symbol("π_$n")]) ,1:T-1]]
    pict   =  vcat(0,states[m.endogenous_states[:πc],1:T-1])                     #[0 Y[m.endogenous_states[:πc] - NK,1:T-1]]
    piwt   = vcat(0,states[m.endogenous_states[:πw],1:T-1])                                  #[0 Y[m.endogenous_states[:πw] - NK,1:T-1]]
    taut   = vcat(0,states[m.endogenous_states[:τ],1:T-1])                                   #[0 X[m.endogenous_states[:τ],1:T-1]]
    tauhatt = vcat(0,states[m.endogenous_states[:τ],1:T-1])                               #[0 X[m.endogenous_states[:τ],1:T-1]] # same
    snormt = states[m.endogenous_states[:ls_2]:m.endogenous_states[Symbol("ls_$(n)")],1:T]
    st     = C*snormt
    it     = states[m.endogenous_states[:li],1:T+24]
end


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
    #irf["pit_services"] = (m[:gam_services].value)' * irf["piit"]
    #irf["pit_coreservices"] = (m[:gam_coreservices].value)' * irf["piit"]
    #irf["pit_goods"] = (m[:gam_goods].value)'*irf["piit"]
    #irf["pit_coregoods"] = (m[:gam_coregoods].value)'*irf["piit"]
    irf["piwt"] = piwt
    irf["wt"] = wt

    # variables needed for kanzig irfs

    irf["cumpt_energy"] = cumsum(irf["pit_energy"], dims =2)  # % energy price level
    irf["cumpt_cpi"]    = cumsum(irf["pict"], dims =1)        # % consumer price level
    irf["cumpt_core"]   = cumsum(irf["pit_core"], dims =2)    # % core cpi level
    irf["cumpt_oil"]    = cumsum(irf["piit"][get_setting(m, :oil),:], dims =1) #% oil price

    irf["cumpt_oilr"] = irf["cumpt_oil"] - irf["cumpt_cpi"]
    #irf["cumpt_services"] = cumsum(irf["pit_services"], dims=2)
    #irf["cumpt_food"] = cumsum(irf["pit_food"],dims=1)

    irf["it_2yr"] = zeros(1,T)
    for t=2:T
        irf["it_2yr"][t] = sum(it[t:t+23])/2
    end
    irf["it"] = it[1:T]
    irf["ann_it"] = 12*it[1:T]



    return irf

end
