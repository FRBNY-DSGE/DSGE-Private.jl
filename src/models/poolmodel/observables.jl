function init_observable_mappings!(m::PoolModel)
    observables = OrderedDict{Symbol,Observable}()

    identity_transform_805 = function(levels)
        return levels[:p805]
    end
    identity_transform_904 = function(levels)
        return levels[:p904]
    end

    if subspec(m) == "ss1"
        ############################################################################
        ## 1. Model 904 predictive density
        ############################################################################

        observables[:Model904] = Observable(:Model904, [:p904__wrongorigmatlab],
                                            identity_transform_904,
                                            identity_transform_904,
                                            "Model 904 predictive density",
                                            "Model 904 conditional predictive density scores")

        ############################################################################
        ## 2. Model 805 predictive density
        ############################################################################
        observables[:Model805] = Observable(:Model805, [:p805__wrongorigmatlab],
                                            identity_transform_805,
                                            identity_transform_805,
                                            "Model 805 predictive density",
                                            "Model 805 conditional predictive density scores")
    elseif subspec(m) == "ss2"
        ############################################################################
        ## 1. Model 904 predictive density
        ############################################################################
        observables[:Model904] = Observable(:Model904, [:p904__right805orig904],
                                            identity_transform_904,
                                            identity_transform_904,
                                            "Model 904 predictive density",
                                            "Model 904 conditional predictive density scores")

        ############################################################################
        ## 2. Model 805 predictive density
        ############################################################################
        observables[:Model805] = Observable(:Model805, [:p805__right805orig904],
                                            identity_transform_805,
                                            identity_transform_805,
                                            "Model 805 predictive density",
                                            "Model 805 conditional predictive density scores")

    end

    m.observable_mappings = observables
end
