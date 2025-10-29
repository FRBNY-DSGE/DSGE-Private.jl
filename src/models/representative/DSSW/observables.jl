function init_observable_mappings!(m::DSSW)

    observables = OrderedDict{Symbol,Observable}()
    #population_mnemonic = get(get_setting(m, :population_mnemonic))

    observables[:output_growth] = Observable(:output_growth, [:output_growth], identity, identity,
                                             "log real GDP per capita", "log real GDP per capita")

    observables[:consumption] = Observable(:consumption, [:consumption], identity, identity,
                                           "log nominal consumption deflated by GDP deflator per capita", "log nominal consumption deflated by GDP deflator per capita")

    observables[:investment_growth] = Observable(:investment_growth, [:investment_growth], identity, identity,
                                                 "log nominal investment deflated by GDP deflator per capita", "log nominal investment deflated by GDP deflator per capita")

    observables[:hours] = Observable(:hours, [:hours], identity, identity,
                                     "log hours worked per capita", "log hours worked per capita")

    observables[:wage_growth] = Observable(:wage_growth, [:wage_growth], identity, identity,
                                           "log hourly real wage", "log hourly real wage")

    observables[:inflation] = Observable(:inflation, [:inflation], identity, identity,
                                         "log GDP deflator", "log GDP deflator")

    observables[:nominal_rate] = Observable(:nominal_rate, [:nominal_rate], identity, identity,
                                            "nominal short-term interest rate", "nominal short-term interest rate")

    if get_setting(m, :n_coint) > 0
        observables[:consumption_coint] = Observable(:consumption_coint, [:consumption_coint], identity, identity,
                                                "cointegration: consumption - output", "cointegration: consumption - output")
        observables[:investment_coint] = Observable(:investment_coint, [:investment_coint], identity, identity,
                                                "cointegration: investment - output", "cointegration: investment - output")
        observables[:wage_coint] = Observable(:wage_coint, [:wage_coint], identity, identity,
                                                "cointegration: real wage - output", "cointegration: real wage - output")
    end


    m.observable_mappings = observables
end
