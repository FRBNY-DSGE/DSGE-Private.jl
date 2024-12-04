function init_pseudo_observable_mappings!(m::OnionModel)
    subspec_ind = isletter(subspec(m)[end]) ? length(subspec(m)) - 1 : length(subspec(m))

    pseudo_names = [:core_cpi, :core_Kcpi, :pseudo_CPI, :pseudo_KCPI,:core_goods_Kcpi, :core_services_Kcpi ]

    pseudo = OrderedDict{Symbol, PseudoObservable}()
    for k in pseudo_names
        pseudo[k] = PseudoObservable(k)
    end


    # Fill in names and reverse transforms

    pseudo[:pseudo_CPI].name = "Pseudo CPI"
    pseudo[:pseudo_CPI].longname = "Pseudo CPI aggregated from gamma weighted sum of sectoral inflation"
    pseudo[:pseudo_CPI].rev_transform = identity
    #pseudo[:pseudo_CPI].fwd_transform = "


    pseudo[:pseudo_KCPI].name = "Pseudo K-CPI"
    pseudo[:pseudo_KCPI].longname = "Pseudo K-CPI aggregated from gamma weighted sum of sectoral inflation"
    pseudo[:pseudo_KCPI].rev_transform = identity
    #pseudo[:pseudo_CPI].fwd_transform = "

    pseudo[:core_cpi].name = "Core CPI"
    pseudo[:core_cpi].longname = "Core CPI aggregated from sectoral inflation"
    pseudo[:core_cpi].rev_transform = identity


    pseudo[:core_Kcpi].name = "Core K-CPI"
    pseudo[:core_Kcpi].longname = "Core K-CPI aggregated from sectoral inflation"
    pseudo[:core_Kcpi].rev_transform = identity


    pseudo[:core_services_Kcpi].name = "K-CPI Core Services"
    pseudo[:core_services_Kcpi].longname = "K-CPI Core Services"

    pseudo[:core_goods_Kcpi].name = "K-CPI Core Goods"
    pseudo[:core_goods_Kcpi].longname = "K-CPI Core Goods"

#=
    pseudo[:LongRunInflation].name = "Long Run Inflation"
    pseudo[:LongRunInflation].longname = "Long Run Inflation"
    pseudo[:LongRunInflation].rev_transform = quartertoannual
=#

    m.pseudo_observable_mappings = pseudo

end
