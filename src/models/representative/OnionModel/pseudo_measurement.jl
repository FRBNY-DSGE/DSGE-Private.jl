function pseudo_measurement(m::OnionModel{T},
                            TTT::Matrix{T},
                            RRR::Matrix{T},
                            CCC::Vector{T}) where {T <:AbstractFloat}

    subspec_int = parse(Int, subspec(m)[3:end])

    endo      = m.endogenous_states
    endo_addl = m.endogenous_states_augmented
    pseudo    = m.pseudo_observables

    _n_states = n_states_augmented(m)
    _n_pseudo = n_pseudo_observables(m)

    ZZ_pseudo = zeros(_n_pseudo, _n_states)
    DD_pseudo = zeros(_n_pseudo)


    #Gamma weighted sum of CPI
    for i in 1:get_setting(m, :n_sectors)
        ZZ_pseudo[pseudo[:pseudo_CPI], endo[Symbol("π_$i")]] = get_setting(m, :gam)[i]
    end

    for i in 1:get_setting(m, :n_sectors)
        ZZ_pseudo[pseudo[:pseudo_KCPI], endo[Symbol("π_$i")]] = get_setting(m, :Kgam)[i]
    end

    # Consumption growth pseudo observable for ss117 IRFs
    if subspec_int ∈ [117]
        ZZ_pseudo[pseudo[:pseudo_cons], endo[:c_t]] = 1.0
        ZZ_pseudo[pseudo[:pseudo_cons], endo_addl[:c_t1]] = -1.0
    end


    #=
    # Core CPI
    core_gam_sum = sum(m[:gam].value[get_setting(m, :core_sectors)])

    for i in get_setting(m, :core_sectors)
        ZZ_pseudo[pseudo[:Pseudo_core_cpi], endo[Symbol("π_$i")]] = m[:gam].value[i]/core_gam_sum
    end


    # Core CPI
    core_gam_sum = sum(m[:Kgam].value[get_setting(m, :core_sectors)])

    for i in get_setting(m, :core_sectors)
        ZZ_pseudo[pseudo[:Pseudo_core_Kcpi], endo[Symbol("π_$i")]] = m[:Kgam].value[i]/core_gam_sum
    end


    # Energy CPI


    # Food CPI


    # Services CPI
    core_serve_gam_sum = sum(m[:Kgam].value[get_setting(m, :core_service_sectors)])

    for i in get_setting(m, :core_service_sectors)
        ZZ_pseudo[pseudo[:Pseudo_core_services_Kcpi], endo[Symbol("π_$i")]] = m[:Kgam].value[i]/core_serve_gam_sum
    end

    # Core Goods CPI
    core_good_gam_sum = sum(m[:Kgam].value[get_setting(m, :core_goods_sectors)])

    for i in get_setting(m, :core_goods_sectors)
        ZZ_pseudo[pseudo[:Pseudo_core_goods_Kcpi], endo[Symbol("π_$i")]] = m[:Kgam].value[i]/core_good_gam_sum
    end
=#

     ## Long Run Inflation
    #=
    ZZ_pseudo[pseudo[:LongRunInflation],endo[:π_star_t]] = 1.
    DD_pseudo[pseudo[:LongRunInflation]]                 = 100. *(m[:π_star]-1.)
    =#

    return PseudoMeasurement(ZZ_pseudo, DD_pseudo)
end
