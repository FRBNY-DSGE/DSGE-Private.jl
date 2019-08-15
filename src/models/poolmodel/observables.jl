function init_observable_mappings!(m::PoolModel)
    observables = OrderedDict{Symbol,Observable}()

    if subspec(m) == "ss1"
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
    elseif subspec(m) == "ss2"
        ############################################################################
        ## 1. Model 904 predictive density
        ############################################################################
        observables[:Model904] = Observable(:Model904, [:right805orig904, :p904],
                                            x -> x, x -> x,
                                            "Model 904 predictive density",
                                            "Model 904 conditional predictive density scores")

        ############################################################################
        ## 2. Model 805 predictive density
        ############################################################################
        observables[:Model805] = Observable(:Model805, [:right805orig904, :p805],
                                            x -> x, x -> x,
                                            "Model 805 predictive density",
                                            "Model 805 conditional predictive density scores")

    end

    m.observable_mappings = observables
end
