function init_observable_mappings!(m::PLT) # do not edit inputs
    # The goal of this function is to populate the m.observable_mappings field
    # with an OrderedDict mapping name (Symbol) to an Observable instance.
    # You should probably edit almost the entirety of this function

    observables = OrderedDict{Symbol,Observable}() # do not edit this line
    population_mnemonic = get(get_setting(m, :population_mnemonic)) # up to you whether you want to keep this line

    ############################################################################
    ## 1. Real GDP Growth
    ############################################################################
    # gdp_fwd_transform = function (levels)
    #     # FROM: Level of GDP (from FRED)
    #     # TO: Quarter-to-quarter percent change of real GDP per capita

    #     levels[!,:temp] = percapita(m, :GDP, levels)
    #     gdp = 1000 * nominal_to_real(:temp, levels)
    #     oneqtrpctchange(gdp)
    # end

    function identity(x)
        return x
    end
    # gdp_rev_transform = loggrowthtopct_annualized_percapita


    ############################################################################
    ## 2. CPI Inflation
    ############################################################################

    # cpi_fwd_transform = function (levels)
    #     # FROM: CPI urban consumers index (from FRED)
    #     # TO: Annualized quarter-to-quarter percent change of CPI index

    #     quartertoannual(oneqtrpctchange(levels[!,:CPIAUCSL]))
    # end

    # cpi_rev_transform = loggrowthtopct_annualized

    observables[:i_t] = Observable(:i_t, [:CPIAUCSL__FRED],
                                        identity, identity,
                                        "i_t",
                                        "i_t")

    observables[:π_t] = Observable(:π_t, [:DFF__FRED],
                                               identity, identity,
                                               "pi_t",
                                               "pi_t")

   observables[:x_t] = Observable(:x_t, [:GDP__FRED, population_mnemonic, :GDPDEF__FRED],
                                     identity, identity,
                                      "x_t", "x_t")

    observables[:p_t] = Observable(:p_t, [:DFF__FRED],
                                               identity, identity,
                                               "p_t",
                                               "p_t")

    # observables[:u_t] = Observable(:u_t, [:DFF__FRED],
    #                                            identity, identity,
    #                                            "u_t",
    #                                            "u_t")

    # observables[:r_t] = Observable(:r_t, [:DFF__FRED],
    #                                            identity, identity,
    #                                            "r_t",
    #                                            "r_t")

    ############################################################################
    ## 3. Nominal short-term interest rate (3 months)
    ############################################################################

    # nominalrate_fwd_transform = function (levels)
    #     # FROM: Nominal effective federal funds rate (aggregate daily data at a
    #     #       quarterly frequency at an annual rate)
    #     # TO:   Nominal effective fed funds rate, at a quarterly rate annualized

    #     levels[!,:DFF]
    # end

    # nominalrate_rev_transform = identity




    m.observable_mappings = observables # do not edit this line
end
