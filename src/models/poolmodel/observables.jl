function init_observable_mappings!(m::PoolModel)

    observables = OrderedDict{Symbol,Observable}()

    ############################################################################
    ## 1. Model 904 predictive density
    ############################################################################
    observables[:Model904] = Observable(:Model904, [:wrongorigmatlab, :p904],
                                        x -> x, x -> x,
                                        "Model 904 predictive density",
                                        "Model 904 conditional predictive density scores")

    ############################################################################
    ## 2. Model 805 predictive density
    ############################################################################
    observables[:Model805] = Observable(:Model805, [:wrongorigmatlab, :p805],
                                        x -> x, x -> x,
                                        "Model 805 predictive density",
                                        "Model 805 conditional predictive density scores")

    m.observable_mappings = observables
end
