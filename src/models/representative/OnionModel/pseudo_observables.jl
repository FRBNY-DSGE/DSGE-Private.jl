function init_pseudo_observable_mappings!(m::OnionModel)

    subspec_int = parse(Int, subspec(m)[3:end])

    pseudo_names = [:pseudo_CPI, :pseudo_KCPI, :pseudo_cons]

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

#=
    pseudo[:Pseudo_core_cpi].name = "Pseudo Core CPI"
    pseudo[:Pseudo_core_cpi].longname = "Core CPI aggregated from sectoral inflation"
    pseudo[:Pseudo_core_cpi].rev_transform = identity


    pseudo[:Pseudo_core_Kcpi].name = "Pseudo Core K-CPI"
    pseudo[:Pseudo_core_Kcpi].longname = "Core K-CPI aggregated from sectoral inflation"
    pseudo[:Pseudo_core_Kcpi].rev_transform = identity


    pseudo[:Pseudo_core_services_Kcpi].name = "Pseudo K-CPI Core Services"
    pseudo[:Pseudo_core_services_Kcpi].longname = "K-CPI Core Services"

    pseudo[:Pseudo_core_goods_Kcpi].name = "Pseudo K-CPI Core Goods"
    pseudo[:Pseudo_core_goods_Kcpi].longname = "K-CPI Core Goods"

    pseudo[:LongRunInflation].name = "Long Run Inflation"
    pseudo[:LongRunInflation].longname = "Long Run Inflation"
    pseudo[:LongRunInflation].rev_transform = quartertoannual
=#
    # ss117 consumption growth pseudo observable for IRFs
    pseudo[:pseudo_cons].name = "Consumption Growth"
    pseudo[:pseudo_cons].longname = "Consumption Growth pseudo observable"
    pseudo[:pseudo_cons].rev_transform = identity


    m.pseudo_observable_mappings = pseudo

end
