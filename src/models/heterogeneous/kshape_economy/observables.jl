function init_observable_mappings!(m::mBBQ)
    observables = OrderedDict{Symbol,Observable}()
    _init_original_observable_mappings!(m, observables)
    m.observable_mappings = observables
end

function _init_original_observable_mappings!(m::mBBQ, observables::OrderedDict{Symbol, Observable})
    iden_fwd_transform = (levels, name) -> levels[!, name]
    iden_rev_transform = x -> x

    ############################################################################
    ## 1. GDP growth per capita
    ############################################################################
    observables[:obs_gdp] = Observable(
        :obs_gdp,
        [:ygpc],
        x -> iden_fwd_transform(x, :ygpc),
        iden_rev_transform,
        "Real GDP Growth per capita",
        "Identity mapping",
    )

    ############################################################################
    ## 2. Consumption growth per-capita
    ############################################################################
    observables[:obs_consumption] = Observable(
        :obs_consumption,
        [:cgpc],
        x -> iden_fwd_transform(x, :cgpc),
        iden_rev_transform,
        "Real Consumption Growth per capita",
        "Identity mapping",
    )

    ############################################################################
    ## 3. Investment growth per capita
    ############################################################################
    observables[:obs_investment] = Observable(
        :obs_investment,
        [:igpc],
        x -> iden_fwd_transform(x, :igpc),
        iden_rev_transform,
        "Real Investment Growth",
        "Identity mapping",
    )

    ############################################################################
    ## 4. Inflation
    ############################################################################

    ############################################################################
    ## 5. Nominal interest rate (shadow)
    ############################################################################
    observables[:obs_nominalrate] = Observable(
        :obs_nominalrate,
        [:rcb],
        x -> iden_fwd_transform(x, :rcb),
        iden_rev_transform,
        "Nominal Interest Rate",
        "Identity mapping",
    )

    ############################################################################
    ## 6. Wage growth
    ############################################################################
    observables[:obs_wages] = Observable(
        :obs_wages,
        [:wg],
        x -> iden_fwd_transform(x, :wg),
        iden_rev_transform,
        "Real Wage Growth",
        "Identity mapping",
    )

    ############################################################################
    ## 7. Unemployment rate
    ############################################################################
    observables[:obs_unemployment_rate] = Observable(
        :obs_unemployment_rate,
        [:u],
        x -> iden_fwd_transform(x, :u),
        iden_rev_transform,
        "Unemployment Rate",
        "Identity mapping",
    )
    ############################################################################
    ## 8. Lump sum transfer
    ############################################################################
    observables[:obs_lumpsum_transfer] = Observable(
        :obs_lumpsum_transfer,
        [:lt],
        x -> iden_fwd_transform(x, :lt),
        iden_rev_transform,
        "Lump Sum Transfer",
        "Identity mapping",
    )
    ############################################################################
    ## 9. Profits
    ############################################################################
    observables[:obs_profit] = Observable(
        :obs_profit,
        [:p],
        x -> iden_fwd_transform(x, :p),
        iden_rev_transform,
        "Profits",
        "Identity mapping",
    )
    ############################################################################
    ## 10. Central bank assets
    ############################################################################
    observables[:obs_central_bank_assets] = Observable(
        :obs_central_bank_assets,
        [:xcb],
        x -> iden_fwd_transform(x, :xcb),
        iden_rev_transform,
        "Central Bank Assets",
        "Identity mapping",
    )
    ############################################################################
    ## 11. Real stock returns
    ############################################################################
    observables[:obs_stock_returns] = Observable(
        :obs_stock_returns,
        [:sp500],
        x -> iden_fwd_transform(x, :sp500),
        iden_rev_transform,
        "SP500 returns",
        "Identity mapping",
    )
    observables
end
