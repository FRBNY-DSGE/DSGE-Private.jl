function init_observable_mappings!(m::OnionModel)
    observables = OrderedDict{Symbol, Observable}()
    population_mnemonic = get(get_setting(m, :population_mnemonic))

    consumption_fwd_transform = function (levels)
        # FROM: Nominal consumption
        # TO:   Real consumption, approximate quarter-to-quarter percent change,
        #       per capita, adjusted for population filtering

        levels[!,:temp] = percapita(m, :PCE, levels)
        cons = 1000 * nominal_to_real(:temp, levels)
        oneqtrpctchange(cons)
    end

    consumption_rev_transform = loggrowthtopct_annualized_percapita

    # Consumption - this is currently quarterly, we are still undecided between quarterly and monthly
    observables[:consumption_growth] = Observable(:consumption_growth,
                                                  [:PCE__FRED, population_mnemonic],
                                                  consumption_fwd_transform,
                                                  consumption_rev_transform,
                                                  "Demeaned Consumption Growth",
                                                  "Demeaned Consumption Growth") #adjusted for population filtering as well?

    wages_fwd_transform = function (levels)
            # FROM: Nominal compensation per hour (:COMPNFB from FRED)
            # TO: quarter to quarter percent change of real compensation (using GDP deflator)

            oneqtrpctchange(nominal_to_real(:COMPNFB, levels))
        end

    wages_rev_transform = loggrowthtopct_annualized

    # Real Wage Growth - this is currently quarterly, we are still undecided between quarterly and monthly
    observables[:real_wage_growth] = Observable(:real_wage_growth,
                                                [:COMPNFB__FRED, :GDPDEF__FRED],
                                                wages_fwd_transform,
                                                wages_rev_transform,
                                                "Demeaned Real Wage Growth",
                                                "Demeaned Real Wage Growth")


    cpi_fwd_transform = function(levels)
        oneqtrpctchange(levels[!,:CPIAUCSL])
    end

    cpi_rev_transform = loggrowthtopct_annualized

    # CPI Inflation -
    observables[:cpi_inflation] = Observable(:cpi_inflation, [:CPIAUCSL__FRED],
                                   cpi_fwd_transform,
                                   cpi_rev_transform,
                                   "CPI Inflation",
                                   "CPI Inflation")

    nominalrate_fwd_transform = function (levels)
        # FROM: Nominal effective federal funds rate (aggregate daily data at a
        #       quarterly frequency at an annual rate)
        # TO:   Nominal effective fed funds rate, at a quarterly rate

        annualtoquarter(levels[!,:DFF])
    end

    nominalrate_rev_transform = quartertoannual

    # FFR - this is currently quarterly, we are still undecided between quarterly and monthly
    observables[:NominalFFR] = Observable(:NominalFFR,
                                   [:DFF__FRED],
                                   nominalrate_fwd_transform,
                                   nominalrate_rev_transform,
                                   "Demeaned Nominal FFR",
                                   "Demeaned Nominal FFR")


    # CPI Sectoral Inflation
    inflation_sector_names = get_setting(m, :sector_names)

    for i in 1:get_setting(m, :n_sectors)

        cpisector_fwd_transform = function(levels)
            100 * levels[!, Symbol("$(inflation_sector_names[i])")]
        end

        if get_setting(m, :data_quarter_or_month) == :quarter
            observables[Symbol("Inflation, $(inflation_sector_names[i])")] = Observable(Symbol("Inflation, $(inflation_sector_names[i])"),
                                                                                        [Symbol("$(inflation_sector_names[i])__CPIQUARTER")],
                                                                                        cpisector_fwd_transform,
                                                                                        identity,
                                                                                        "CPI Sector $(i) Inflation, $(inflation_sector_names[i])",
                                                                                        "CPI Sector $(i) Inflation, $(inflation_sector_names[i])")
        else
            observables[Symbol("Inflation, $(inflation_sector_names[i])")] = Observable(Symbol("Inflation, $(inflation_sector_names[i])"),
                                                                                        [Symbol("$(inflation_sector_names[i])__CPIMONTH")],
                                                                                        cpisector_fwd_transform,
                                                                                        identity,
                                                                                        "CPI Sector $(i) Inflation, $(inflation_sector_names[i])",
                                                                                        "CPI Sector $(i) Inflation, $(inflation_sector_names[i])")
        end
    end

    #m <= Setting(:forward_looking_observables, [:inflation_expectations_10year])
    #observables[:inflation_expectations_10year] = Observable(:inflation_expectations_10year, [])



    m.observable_mappings = observables

end
