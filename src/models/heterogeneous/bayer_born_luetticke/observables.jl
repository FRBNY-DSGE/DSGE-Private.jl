function init_observable_mappings!(m::BayerBornLuetticke)

    observables = OrderedDict{Symbol,Observable}()
    population_mnemonic = get(get_setting(m, :population_mnemonic))

    iden_fwd_transform = function (levels, name)
        return levels[!, name]
    end

    ############################################################################
    ## 1. GDP growth per capita
    ############################################################################
    gdp_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_gdp] = Observable(:obs_gdp, [:Ygrowth__BBL, population_mnemonic], # add population_mnenomic here just to make pop data available
                                       x -> iden_fwd_transform(x, :Ygrowth), gdp_rev_transform,
                                       "Real GDP Growth", "Real GDP Growth Per Capita")

    ############################################################################
    ## 2. Consumption growth per-capita
    ############################################################################

    cons_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_consumption] = Observable(:obs_consumption, [:Cgrowth__BBL],
                                               x -> iden_fwd_transform(x, :Cgrowth), cons_rev_transform,
                                               "Real Consumption Growth", "Real Consumption Growth Per Capita")

    ############################################################################
    ## 3. Investment growth per capita
    ############################################################################

    invst_rev_transform = loggrowthtopct_annualized_percapita

    observables[:obs_investment] = Observable(:obs_investment, [:Igrowth__BBL],
                                              x -> iden_fwd_transform(x, :Igrowth), invst_rev_transform,
                                              "Real Investment Growth",
                                              "Real Investment Growth Per Capita")

    ############################################################################
    ## 4. Wage growth
    ############################################################################

    wages_rev_transform = loggrowthtopct_annualized

    observables[:obs_wages] = Observable(:obs_wages, [:wgrowth__BBL],
                                         x -> iden_fwd_transform(x, :wgrowth), wages_rev_transform,
                                         "Real Wage Growth",
                                         "Real Wage Growth in Non-Farm Business Sector")

    ############################################################################
    ## 5. Hours per capita
    ############################################################################

    hrs_rev_transform = logleveltopct_annualized_percapita

    observables[:obs_hours] = Observable(:obs_hours, [:N__BBL],
                                         x -> iden_fwd_transform(x, :N), hrs_rev_transform,
                                         "Hours Growth",
                                         "Hours Growth Per Capita")

    ############################################################################
    ## 6. GDP Deflator (Inflation)
    ############################################################################

    gdpdef_rev_transform = loggrowthtopct_annualized

    observables[:obs_gdpdeflator] = Observable(:obs_gdpdeflator, [:pi__BBL],
                                               x -> iden_fwd_transform(x, :pi), gdpdef_rev_transform,
                                               "GDP Deflator",
                                               "GDP Deflator Inflation")

    ############################################################################
    ## 7. Nominal interest rate (shadow)
    ############################################################################

    nomrate_rev_transform = quartertoannual

    observables[:obs_nominalrate] = Observable(:obs_nominalrate, [:RB__BBL],
                                               x -> iden_fwd_transform(x, :RB), nomrate_rev_transform,
                                               "Nominal Interest Rate",
                                               "Nominal Interest Rate, augmented by shadow rate from Wu and Xia (2016) during ZLB")

    ############################################################################
    # 8. Wealth inequality
    ############################################################################

    observables[:obs_W90share] = Observable(:obs_W90share, [:w90share__BBL],
                                            x -> iden_fwd_transform(x, :w90share), identity,
                                            "Top 90% Wealth Share",
                                            "90th Percentile of Net Personal Wealth Distribution")

    ############################################################################
    # 9. Income inequality
    ############################################################################

    observables[:obs_I90share] = Observable(:obs_I90share, [:I90share__BBL],
                                            x -> iden_fwd_transform(x, :I90share), identity,
                                            "Top 90% Income Share",
                                            "Top 90% Percentile of Pre-Tax National Income Distribution")

    ############################################################################
    # 10. Idiosyncratic income risk
    ############################################################################

    observables[:obs_sigmasq] = Observable(:obs_sigmasq, [:sigma2__BBL],
                                           x -> iden_fwd_transform(x, :sigma2), identity,
                                           "Idiosyncratic Income Risk",
                                           "Variance of Idiosyncratic Income")

    ############################################################################
    # 11. Tax progressivity
    ############################################################################

    observables[:obs_taxprogressivity] = Observable(:obs_taxprogressivity, [:tauprog__BBL],
                                                    x -> iden_fwd_transform(x, :tauprog), identity,
                                                    "Tax Progressivity",
                                                    "Tax Progressivity from Ferriere and Navarro (2018)")

    m.observable_mappings = observables
end
