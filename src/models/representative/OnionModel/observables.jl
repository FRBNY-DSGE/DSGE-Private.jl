function init_observable_mappings!(m::OnionModel)
    observables = OrderedDict{Symbol, Observable}()
    population_mnemonic = get(get_setting(m, :population_mnemonic))

    demean = function (levels)
        cons = all(ismissing.(levels)) ? missing : mean(skipmissing(levels))
        for i in 1:length(levels)
            levels[i] = ismissing(levels[i]) ? levels[i] : levels[i] - cons
        end
        levels
    end

    #Auxillary function to subtract 2.3 annualized from CPI rather than the mean
    demean2 = function (levels)
        cons = 2.3/4        #all(ismissing.(levels)) ? missing : mean(skipmissing(levels))
        for i in 1:length(levels)
            levels[i] = ismissing(levels[i]) ? levels[i] : levels[i] - cons
        end
        levels
    end

    consumption_fwd_transform = function (levels)
        # FROM: Nominal consumption
        # TO:   Real consumption, approximate quarter-to-quarter percent change,
        #       per capita, adjusted for population filtering

        levels[!,:temp] = percapita(m, :PCE, levels)
        cons = 1000 * nominal_to_real(:temp, levels)
        demean(oneqtrpctchange(cons))
    end

    consumption_rev_transform = identity     # loggrowthtopct_annualized_percapita

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

            demean(oneqtrpctchange(nominal_to_real(:COMPNFB, levels)))
        end

    wages_rev_transform = identity #loggrowthtopct_annualized

    # Real Wage Growth - this is currently quarterly, we are still undecided between quarterly and monthly
    observables[:real_wage_growth] = Observable(:real_wage_growth,
                                                [:COMPNFB__FRED, :GDPDEF__FRED],
                                                wages_fwd_transform,
                                                wages_rev_transform,
                                                "Demeaned Real Wage Growth",
                                                "Demeaned Real Wage Growth")

if get_setting(m, :marco_test_num) == 69
    cpi_fwd_transform = function(levels)
        demean(oneqtrpctchange(levels[!,:CPIAUCSL]))
    end

    #cpi_rev_transform = loggrowthtopct_annualized
    cpi_rev_transform = identity

    # CPI Inflation -
    observables[:cpi_inflation] = Observable(:cpi_inflation, [:CPIAUCSL__FRED],
                                   cpi_fwd_transform,
                                   cpi_rev_transform,
                                   "CPI Inflation",
                                             "CPI Inflation")
end


#=
    #Core servies
     core_service_cpi_fwd_transform = function(levels)
        demean(oneqtrpctchange(levels[!,:CUSR0000SASLE]))
    end

    core_service_cpi_rev_transform = identity
    observables[:cpi_core_services] = Observable(:cpi_core_services, [:CUSR0000SASLE__FRED],    #[:CUSR0000SACL1E__FRED],
                                   core_service_cpi_fwd_transform,
                                   core_service_cpi_rev_transform,
                                   "CPI Core Services",
                                             "CPI Core Services")

    #Core goods
     core_goods_cpi_fwd_transform = function(levels)
        demean(oneqtrpctchange(levels[!,:CUSR0000SACL1E]))
    end

    core_goods_cpi_rev_transform = identity
    observables[:cpi_core_goods] = Observable(:cpi_core_goods, [:CUSR0000SACL1E__FRED],
                                   core_goods_cpi_fwd_transform,
                                   core_goods_cpi_rev_transform,
                                   "CPI Core Goods",
                                              "CPI Core Goods")


    #Energy
    energy_cpi_fwd_transform = function(levels)
        demean(oneqtrpctchange(levels[!, :CPIENGSL]))
    end


    energy_cpi_rev_transform = identity
    observables[:cpi_energy] = Observable(:cpi_energy, [:CPIENGSL__FRED],
                                          energy_cpi_fwd_transform,
                                          energy_cpi_rev_transform,
                                          "CPI Energy",
                                          "CPI Energy")


    =#



    # CPI Categorical Inflation
    inflation_subgroup_names = collect(keys(get_setting(m,:subgroup_names)))
subgroup_data_source = collect(values(get_setting(m,:subgroup_names)))

    for i in 1:length(inflation_subgroup_names)

        cpisector_fwd_transform = function(levels)
            demean(levels[!, Symbol("$(inflation_subgroup_names[i])")])
        end
            observables[Symbol("Inflation, $(inflation_subgroup_names[i])")] = Observable(Symbol("$(inflation_subgroup_names[i])"),
                                                                                        [subgroup_data_source[i]],
                                                                                        cpisector_fwd_transform,
                                                                                        identity,
                                                                                        "$(inflation_subgroup_names[i])",
                                                                                        "CPI: $(inflation_subgroup_names[i])")
    end


    nominalrate_fwd_transform = function (levels)
        # FROM: Nominal effective federal funds rate (aggregate daily data at a
        #       quarterly frequency at an annual rate)
        # TO:   Nominal effective fed funds rate, at a quarterly rate

        demean(annualtoquarter(levels[!,:DFF]))
    end

    nominalrate_rev_transform = identity #quartertoannual

    # FFR - this is currently quarterly, we are still undecided between quarterly and monthly
    observables[:NominalFFR] = Observable(:NominalFFR,
                                   [:DFF__FRED],
                                   nominalrate_fwd_transform,
                                   nominalrate_rev_transform,
                                   "Demeaned Nominal FFR",
                                          "Demeaned Nominal FFR")


############################################################################
#Fernald TFP
############################################################################
if get_setting(m, :marco_test_num) == 68 || get_setting(m, :marco_test_num) == 70 || get_setting(m, :marco_test_num) == 71 || get_setting(m, :marco_test_num) == 100

    tfp_rev_transform = identity #quartertoannual
    tfp_fwd_transform =  function (levels)
        # FROM: Fernald's unadjusted TFP series
        # TO:   De-meaned unadjusted TFP series, adjusted by Fernald's estimated alpha
        # Note: We only want to calculate the mean of unadjusted/adjusted TFP over the
        #       periods between date_presample_start(m) - 1 and
        #       date_mainsample_end(m), though we may end up transforming
        #       additional periods of data.

        start_date = Dates.lastdayofquarter(date_presample_start(m) - Dates.Month(3))
        end_date   = date_mainsample_end(m)
        date_range = start_date .<= levels[1:end, :date] .<= end_date
        tfp_unadj_inrange = levels[date_range, :TFPKQ]

        tfp_unadj      = levels[!,:TFPKQ]
        tfp_unadj_inrange_nonmissing = tfp_unadj_inrange[.!ismissing.(tfp_unadj_inrange)]
        tfp_unadj_inrange_nonmissing = tfp_unadj_inrange_nonmissing[.!isnan.(tfp_unadj_inrange_nonmissing)]
        tfp_unadj_mean = isempty(tfp_unadj_inrange_nonmissing) ? missing : mean(tfp_unadj_inrange_nonmissing)
        (tfp_unadj .- tfp_unadj_mean) ./ (4*(1 .- levels[!,:TFPJQ]))
    end

    observables[:obs_tfp] = Observable(:obs_tfp, [:TFPKQ__DLX, :TFPJQ__DLX],
                                       tfp_fwd_transform, tfp_rev_transform,
                                       "Total Factor Productivity Growth (Fernald)",
                                       "Fernald's TFP, adjusted by Fernald's estimated alpha")
end





#=
    # CPI Sectoral Inflation
    inflation_sector_names = get_setting(m, :sector_names)

    for i in 1:get_setting(m, :n_sectors)

        cpisector_fwd_transform = function(levels)
            demean2(100 * levels[!, Symbol("$(inflation_sector_names[i])")])
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
=#
    #m <= Setting(:forward_looking_observables, [:inflation_expectations_10year])
#observables[:inflation_expectations_10year] = Observable(:inflation_expectations_10year, [])



if get_setting(m, :marco_test_num) == 70 || get_setting(m, :marco_test_num) == 100
    ############################################################################
    # 10. Long term inflation expectations
    ############################################################################

    longinflation_fwd_transform = function (levels)
        # FROM: SPF: 10-Year average yr/yr CPI inflation expectations (annual percent)
        # TO:   FROM, less 0.5
        # Note: We subtract 0.5 because 0.5% inflation corresponds to
        #       the assumed long-term rate of 2 percent inflation, but the
        #       data are measuring expectations of actual inflation.

        demean(annualtoquarter(levels[!,:ASACX10]))

    end

#Demean here by subtracting 2.3 as well

longinflation_rev_transform = identity         #loggrowthtopct_annualized

observables[:obs_longinflation] = Observable(:obs_longinflation, [:ASACX10__DLX],
                                                 longinflation_fwd_transform, longinflation_rev_transform,
                                                 "10-year average inflation expectations",
                                                 "10-year average yr/yr CPI inflation expectations")


end

m.observable_mappings = observables

end
