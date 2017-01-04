"""
```
transform_data(m::AbstractModel, levels::DataFrame; cond_type::Symbol = :none,
    verbose::Symbol = :low)
```

Transform data loaded in levels and order columns appropriately for the DSGE
model. Returns DataFrame of transformed data.

The DataFrame `levels` is output from `load_data_levels`. The series in levels are
transformed as specified in `m.observable_mappings`.

- To prepare for per-capita transformations, population data are filtered using
  `hpfilter`. The series in `levels` to use as the population series is given by
  the `population_mnemonic` setting. If `use_population_forecast(m)`, a
  population forecast is appended to the recorded population levels before the
  filtering. Both filtered and unfiltered population levels and growth rates are
  added to the `levels` data frame.
- The transformations are applied for each series using the `levels` DataFrame
  as input.

Conditional data (identified by `cond_type in [:semi, :full]`) are handled
slightly differently: If `use_population_forecast(m)`, we drop the first period
of the population forecast because we treat the first forecast period
(`date_forecast_start(m)` as if it were data. We also only apply transformations
for the observables given in `cond_full_names(m)` or `cond_semi_names(m)`.
"""
function transform_data(m::AbstractModel, levels::DataFrame; cond_type::Symbol = :none, verbose::Symbol = :low)

    population_mnemonic = parse_population_mnemonic(m)[1]
    n_obs, _ = size(levels)

    # Step 1: HP filter population forecasts, if they're being used

    # population_recorded: historical population, unfiltered
    # population_all: full unfiltered series (including forecast)
    # dlpopulation_forecast: growth rates of population forecasts pre-filtering

    population_recorded = levels[:,[:date, population_mnemonic]]
    population_all, dlpopulation_forecast, n_population_forecast_obs = if use_population_forecast(m)
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            println("Loading population forecast...")
        end

        # load population forecast
        population_forecast_file = inpath(m, "data", "population_forecast_$(data_vintage(m)).csv")
        pop_forecast = readtable(population_forecast_file)

        rename!(pop_forecast, :POPULATION,  population_mnemonic)
        DSGE.na2nan!(pop_forecast)
        DSGE.format_dates!(:date, pop_forecast)

        # for conditional data, start "forecast" one period later
        # (first real forecast period treated as data)
        if cond_type in [:semi, :full]
            pop_forecast = pop_forecast[2:end, :]
        end

        # use our "real" series as current value
        pop_all = vcat(population_recorded, pop_forecast[2:end, :])

        # return values
        pop_all[population_mnemonic],
        difflog(pop_forecast[population_mnemonic]),
        length(pop_forecast[population_mnemonic])
    else
        population_recorded[:,population_mnemonic], [NaN], 1
    end

    # hp filter
    population_all = convert(Array, population_all)
    filtered_population, _ = hpfilter(population_all, 1600)

    # filtered series (levels)
    filtered_population_recorded = filtered_population[1:end-n_population_forecast_obs+1]
    filtered_population_forecast = filtered_population[end-n_population_forecast_obs+1:end]

    # filtered growth rates
    dlpopulation_recorded          = difflog(population_recorded[population_mnemonic])
    dlfiltered_population_recorded = difflog(filtered_population_recorded)

    levels[:filtered_population]          = filtered_population_recorded
    levels[:filtered_population_growth]   = dlfiltered_population_recorded
    levels[:unfiltered_population_growth] = dlpopulation_recorded

    # Step 2: apply transformations to each series
    transformed = DataFrame()
    transformed[:date] = levels[:date]

    data_transforms = collect_data_transforms(m)

    for series in keys(data_transforms)
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            println("Transforming series $series...")
        end
        f = data_transforms[series]
        transformed[series] = f(levels)
    end

    sort!(transformed, cols = :date)

    # NaN out observables not used for (semi)conditional forecasts
    if cond_type in [:semi, :full]
        cond_names = if cond_type == :semi
            cond_semi_names(m)
        elseif cond_type == :full
            cond_full_names(m)
        end

        cond_names_nan = setdiff(names(transformed), [cond_names; :date])
        T = eltype(transformed[:, cond_names_nan])
        transformed[transformed[:, :date] .>= date_forecast_start(m), cond_names_nan] = convert(T, NaN)
    end

    return transformed
end


"""
```
transform_data_reduced_form(m::AbstractModel, levels::DataFrame; cond_type::Symbol=:none, verbose::Symbol = :low)
```
Transform data loaded in levels and order columns appropriately for replicating Laubach
Williams. Returns DataFrame of transformed data.
The DataFrame `levels` is output from `load_data_levels`. The series in levels are
transformed as specified in `m.data_transforms`.
- The transformations are applied for each series using the `levels` DataFrame as input.
"""
function transform_data_reduced_form(m::AbstractModel, levels::DataFrame; cond_type::Symbol=:none,verbose::Symbol = :low)

    n_obs, _ = size(levels)
    obs = keys(m.observables)

    transformed = DataFrame()
    transformed[:date] = levels[:date]

    data_transforms = collect_data_transforms(m)

    # Step 1: apply transformations to non-lagged series
    for series in keys(data_transforms)
        if VERBOSITY[verbose] >= VERBOSITY[:high]
            println("Transforming series $series...")
        end
        f = data_transforms[series]
        transformed[series] = f(levels)
    end

    # Step 2: create lags of series by hand
    if :obs_gdp_t in names(levels)
        transformed[:obs_gdp_t1] = lag(transformed,:obs_gdp_t)
        transformed[:obs_gdp_t2] = lag(transformed,:obs_gdp_t1)
        y_keys = [:obs_gdp_t, :obs_π_t]
    elseif :obs_unemp_t1 in obs
        transformed[:obs_unemp_t1] = lag(transformed,:obs_unemp_t)
        transformed[:obs_unemp_t2] = lag(transformed,:obs_unemp_t1)
        y_keys = [:obs_unemp_t, :obs_π_t]
    else
        y_keys = [:obs_unemp_t, :obs_π_t]
    end

    if :obs_r_t1 in obs
        transformed[:obs_r_t1]   = lag(transformed,:obs_r_t)
        transformed[:obs_r_t2]   = lag(transformed,:obs_r_t1)
    end
    if :obs_π_t1 in obs
        transformed[:obs_π_t1]   = lag(transformed,:obs_π_t)

        for t in 2:8
            transformed[symbol("obs_π_t$t")] = lag(transformed,symbol("obs_π_t$(t-1)"))
        end
    end

    if :obs_π_o_t1 in obs && :obs_π_e_t1 in obs
        transformed[:obs_π_o_t1]  = lag(transformed,:obs_π_o_t)
        transformed[:obs_π_e_t1]  = lag(transformed,:obs_π_e_t)
    end

    for col in names(transformed)
        transformed[col] = fill_nan(transformed,col)
    end

    # ensure order matches order in mLaubachWilliams.jl
    X_dict = copy(m.observables)
    [delete!(X_dict,key) for key in y_keys]
    X_keys = [entry[2] for entry in sort(collect(zip(values(X_dict),keys(X_dict))))]
    ordered_cols = [:date; y_keys; X_keys]
    ordered_cols = [Symbol(x) for x in ordered_cols]
    #println("ordered_cols: $ordered_cols")
    transformed = transformed[:,ordered_cols]

    sort!(transformed, cols = :date)


end

"""
Returns the lagged version of df[col]
it fills the leading value with a NaN
"""
function lag(df::DataFrame, col::Symbol)
    n = size(df)[1]
    lagged = ones(n,1)
    lagged[1] = NaN
    for i in 2:n
        lagged[i] = df[i-1,col]
    end
    return(vec(lagged))
end



"""
replaces the NaNs in df[col] with the
last observed value. Meant for use with
lagged series.
"""
function fill_nan(df::DataFrame, col::Symbol)
    n = size(df)[1]
    filled = ones(n,1)
    nan_ind = findin(df[col],NaN)
    if !isempty(nan_ind)
        fill_val_ind = nan_ind[end]+1
        fill_val = df[col][fill_val_ind]
        for i in nan_ind
            df[col][i] = fill_val
        end
    end
    return(df[col])
end

function collect_data_transforms(m; direction=:fwd)

    data_transforms = OrderedDict{Symbol,Function}()

    # Parse vector of observable mappings into data_transforms dictionary
    for obs in keys(m.observable_mappings)
        data_transforms[obs] = getfield(m.observable_mappings[obs], symbol(string(direction) * "_transform"))
    end

    data_transforms

end