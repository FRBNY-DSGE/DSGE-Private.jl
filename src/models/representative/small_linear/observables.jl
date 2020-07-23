function init_observable_mappings!(m::SmallLinear)

    observables = OrderedDict{Symbol,Observable}()
    population_mnemonic = get(get_setting(m, :population_mnemonic))
    population_mnemonic_f = get(get_setting(m, :population_mnemonic_f))

    ############################################################################
    ## 1. Real GDP Growth Domestic
    ############################################################################
    gdp_fwd_transform = function (levels)
        # FROM: Level of GDP (from FRED)
        # TO: Quarter-to-quarter percent change of real GDP per capita

        levels[!,:temp] = percapita(m, :GDP, levels)
        gdp =  nominal_to_real(:temp, levels)
        oneqtrpctchange(gdp)
    end

    gdp_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_gdp] = Observable(:obs_gdp, [:GDP__FRED, population_mnemonic, :GDPDEF__FRED],
                                       gdp_fwd_transform, gdp_rev_transform,
                                       "Real GDP Growth Domestic", "Real GDP Growth Per Capita Domestic")

    ############################################################################
    ## 2. CPI Inflation Domestic
    ############################################################################

    cpi_fwd_transform = function (levels)
        # FROM: CPI urban consumers index (from FRED)
        # TO: Annualized quarter-to-quarter percent change of CPI index

        quartertoannual(oneqtrpctchange(levels[!,:CPIAUCSL]))
    end

    cpi_rev_transform = loggrowthtopct_annualized

    observables[:obs_cpi] = Observable(:obs_cpi, [:CPIAUCSL__FRED],
                                        cpi_fwd_transform, cpi_rev_transform,
                                        "CPI Inflation Domestic",
                                        "CPI Inflation Domestic")

    ############################################################################
    ## 3. Nominal short-term interest rate (3 months) Domestic
    ############################################################################

    nominalrate_fwd_transform = function (levels)
        # FROM: Nominal effective federal funds rate (aggregate daily data at a
        #       quarterly frequency at an annual rate)
        # TO:   Nominal effective fed funds rate, at a quarterly rate annualized
        annualtoquarter(levels[!,:DFF])
    end

    nominalrate_rev_transform = quartertoannual

    observables[:obs_nominalrate] = Observable(:obs_nominalrate, [:DFF__FRED],
                                               nominalrate_fwd_transform, nominalrate_rev_transform,
                                               "Nominal FFR",
                                               "Nominal Effective Fed Funds Rate")

    ############################################################################
    ## 1. Real GDP Growth Foreign
    ############################################################################
    gdp_f_fwd_transform = function (levels)
        # FROM: Level of GDP (from FRED)
        # TO: Quarter-to-quarter percent change of real GDP per capita

        levels[!,:temp] = percapita(:EUNNGDP, levels, :LFEMTTTTEZQ647S)
        gdp = nominal_to_real(:temp, levels)
        oneqtrpctchange(gdp)
    end

    gdp_f_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_gdp_f] = Observable(:obs_gdp_f, [:EUNNGDP__FRED, population_mnemonic_f, :NAGIGP01EZQ661S__FRED],
                                       gdp_f_fwd_transform, gdp_f_rev_transform,
                                       "Real GDP Growth Euro", "Real GDP Growth Per Capita Euro Area")

    ############################################################################
    ## 2. CPI Inflation Foreign
    ############################################################################

    cpi_f_fwd_transform = function (levels)
        # FROM: CPI index (from FRED)
        # TO: Annualized quarter-to-quarter percent change of CPI index

        quartertoannual(oneqtrpctchange(levels[!,:CP0000EZ19M086NEST]))
    end

    cpi_f_rev_transform = loggrowthtopct_annualized

    observables[:obs_cpi_f] = Observable(:obs_cpi_f, [:CP0000EZ19M086NEST__FRED],
                                        cpi_f_fwd_transform, cpi_f_rev_transform,
                                        "CPI Inflation Euro",
                                        "CPI Inflation Euro Area")

    ############################################################################
    ## 3. Nominal short-term interest rate (3 months) Foreign
    ############################################################################

    nominalrate_f_fwd_transform = function (levels)
        # FROM: Nominal rate (aggregate daily data at a
        #       quarterly frequency at an annual rate)
        # TO:   Nominal rate, at a quarterly rate annualized

        annualtoquarter(levels[!,:INTDSREZM193N])
    end

    nominalrate_f_rev_transform = quartertoannual

    observables[:obs_nominalrate_f] = Observable(:obs_nominalrate_f, [:INTDSREZM193N__FRED],
                                               nominalrate_f_fwd_transform, nominalrate_f_rev_transform,
                                               "Nominal Rate Euro",
                                               "Nominal Discount Rate Euro Area")

    m.observable_mappings = observables
end
