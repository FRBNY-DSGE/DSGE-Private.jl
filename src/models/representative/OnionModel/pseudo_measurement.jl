function pseudo_measurement(m::OnionModel{T},
                            TTT::Matrix{T},
                            RRR::Matrix{T},
                            CCC::Vector{T}) where {T <:AbstractFloat}

    subspect_ind = isletter(subspec(m)[end]) ? length(subspec(m)) - 1 : length(subspec(m))

    endo      = m.endogenous_states
    endo_addl = m.endogenous_states_augmented
    pseudo    = m.pseudo_observables

    _n_states = n_states_augmented(m)
    _n_pseudo = n_pseudo_observables(m)

    ZZ_pseudo = zeros(_n_pseudo, _n_states)
    DD_pseudo = zeros(_n_pseudo)


    #Gamma weighted sum of CPI
    #no_nan_sects = setdiff(1:get_setting(m, :n_sectors), get_setting(m, :nan_sects))
    #gam_sum = sum(m[:gam].value[no_nan_sects])
    for i in 1:get_setting(m, :n_sectors) #no_nan_sects
        ZZ_pseudo[pseudo[:pseudo_CPI], endo[Symbol("π_$i")]] = m[:gam].value[i] #/gam_sum
    end


    # Core CPI
    core_gam_sum = sum(m[:gam].value[get_setting(m, :core_sectors)])

    for i in get_setting(m, :core_sectors)
        println("------------------------")
        println(i)
        println(m[:gam].value[i])
        println(m[:gam].value[i]/core_gam_sum)
        ZZ_pseudo[pseudo[:core_cpi], endo[Symbol("π_$i")]] = m[:gam].value[i]/core_gam_sum
    end
    # Energy CPI


    # Food CPI


    # Services CPI
    return PseudoMeasurement(ZZ_pseudo, DD_pseudo)
end
