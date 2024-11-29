function init_pseudo_observable_mappings!(m::OnionModel)
    subspec_ind = isletter(subspec(m)[end]) ? length(subspec(m)) - 1 : length(subspec(m))

    pseudo_names = [:core_cpi]

    pseudo = OrderedDict{Symbol, PseudoObservable}()
    for k in pseudo_names
        pseudo[k] = PseudoObservable(k)
    end


    # Fill in names and reverse transforms

    pseudo[:core_cpi].name = "Core CPI"
    pseudo[:core_cpi].longname = "Core CPI aggregated from sectoral inflation"

    m.pseudo_observable_mappings = pseudo

end
