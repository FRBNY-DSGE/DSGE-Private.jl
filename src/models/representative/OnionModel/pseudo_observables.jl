function init_pseudo_observable_mappings!(m::OnionModel)
    subspec_ind = isletter(subspec(m)[end]) ? length(subspec(m)) - 1 : length(subspec(m))

    pseudo_names = [:core_cpi, :pseudo_CPI]

    pseudo = OrderedDict{Symbol, PseudoObservable}()
    for k in pseudo_names
        pseudo[k] = PseudoObservable(k)
    end


    # Fill in names and reverse transforms

    pseudo[:pseudo_CPI].name = "Pseudo CPI"
    pseudo[:pseudo_CPI].longname = "Pseudo CPI aggregated from gamma weighted sum of sectoral inflation"
    pseudo[:pseudo_CPI].rev_transform = identity #loggrowthtopct_annualized
    #pseudo[:pseudo_CPI].fwd_transform = "

    pseudo[:core_cpi].name = "Core CPI"
    pseudo[:core_cpi].longname = "Core CPI aggregated from sectoral inflation"
    pseudo[:core_cpi].rev_transform = identity #loggrowthtopct_annualized

    m.pseudo_observable_mappings = pseudo

end
