"""
```
plot_altpolicies(models, var, class, cond_type; title = "", kwargs...)

plot_altpolicies(models, vars, class, cond_type;
    forecast_string = "", altpol_string = "",
    start_date = iterate_quarters(date_mainsample_end(models[1], -4)),
    end_date = iterate_quarters(date_forecast_start(models[1], 20)),
    untrans = false, fourquarter = false,
    plotroot = figurespath(models[1], \"forecast\"), 
    titles = [], verbose = :low, kwargs...)
```

Plot `var` or `vars` forecasts under the alternative policies in `models`.

### Inputs

- `models::Vector{AbstractModel}`: vector of models, where each model has a
  different alternative policy
- `var::Symbol` or `vars::Vector{Symbol}`: variable(s) to be plotted,
  e.g. `:obs_gdp` or `[:obs_gdp, :obs_nominalrate]`
- `class::Symbol`
- `cond_type::Symbol`

### Keyword Arguments

- `forecast_string::String`
- `altpol_string::String`: identifies the group of alternative policies being
  plotted together. Required if `length(models) > 1`
- `start_date::Date`
- `end_date::Date`
- `untrans::Bool`: whether to plot untransformed forecast
- `fourquarter::Bool`: whether to plot four-quarter forecast
- `plotroot::String`: if nonempty, plots will be saved in that directory
- `title::String` or `titles::Vector{String}`
- `verbose::Symbol`

See `?histforecast` for additional keyword arguments, all of which can be passed
into `plot_altpolicies`.
"""
function plot_altpolicies{T<:AbstractModel}(models::Vector{T}, var::Symbol, class::Symbol,
                                            cond_type::Symbol; title::String = "",
                                            kwargs...)
    plots = plot_altpolicies(models, [var], class, cond_type;
                             titles = isempty(title) ? String[] : [title],
                             kwargs...)
    return plots[var]
end

function plot_altpolicies{T<:AbstractModel}(models::Vector{T}, vars::Vector{Symbol}, class::Symbol,
                                            cond_type::Symbol;
                                            forecast_string::String = "",
                                            altpol_string::String = "",
                                            start_date::Date = iterate_quarters(date_mainsample_end(models[1]), -4),
                                            end_date::Date = iterate_quarters(date_forecast_start(models[1]), 20),
                                            untrans::Bool = false,
                                            fourquarter::Bool = false,
                                            plotroot::String = figurespath(models[1], "forecast"),
                                            titles::Vector{String} = String[],
                                            verbose::Symbol = :low,
                                            kwargs...)
    # Determine output_vars
    if untrans && fourquarter
        error("Only one of untrans or fourquarter can be true")
    elseif untrans
        hist_prod  = :histut
        fcast_prod = :forecastut
    elseif fourquarter
        hist_prod  = :hist4q
        fcast_prod = :forecast4q
    else
        hist_prod  = :hist
        fcast_prod = :forecast
    end

    # Get titles if not provided
    if isempty(titles)
        detexify_title = typeof(Plots.backend()) == Plots.GRBackend
        titles = map(var -> describe_series(models[1], var, class, detexify = detexify_title), vars)
    end

    # Check if that altpol_string is provided if there are multiple altpolicies being plotted
    n_altpolicies = length(models)
    if n_altpolicies != 1 && isempty(altpol_string)
        error("Must provide nonempty altpol_key if plotting multiple alternative policies")
    end

    # Loop through variables
    plots = OrderedDict{Symbol, Plots.Plot}()
    for (var, title) in zip(vars, titles)
        # Loop through alternative policies
        plots[var] = plot()
        for m in models
            # Get AltPolicy
            altpolicy = alternative_policy(m)

            # Read in MeansBands
            hist     = read_mb(m, :mode, cond_type, Symbol(hist_prod, class), forecast_string = forecast_string)
            forecast = read_mb(m, :mode, cond_type, Symbol(fcast_prod, class), forecast_string = forecast_string)

            # Call recipe
            names  = Dict{Symbol, String}(:hist => "", :forecast => string(altpolicy))
            colors = Dict{Symbol, Any   }(:forecast => altpolicy.color)
            styles = Dict{Symbol, Symbol}(:forecast => altpolicy.linestyle)
            ylabel = series_ylabel(m, var, class, untrans = untrans, fourquarter = fourquarter)

            plots[var] = histforecast!(var, hist, forecast;
                                       start_date = start_date, end_date = end_date,
                                       names = names, colors = colors, styles = styles,
                                       ylabel = ylabel, title = title, kwargs...)
        end

        # Save plot
        if !isempty(plotroot)
            # Temporarily turn on printing altpolicy in models[1] if it's not already
            # on, so it will appear in the output file name
            old_print = models[1].settings[:alternative_policy].print
            models[1].settings[:alternative_policy].print = true

            # Save plot
            output_file = get_forecast_filename(plotroot, filestring_base(models[1]), :mode, cond_type,
                                                Symbol("altpol", fcast_prod, "_", detexify(var)),
                                                forecast_string = forecast_string,
                                                fileformat = plot_extension())
            output_file = replace(output_file, string(alternative_policy(models[1])), altpol_string)
            save_plot(plots[var], output_file, verbose = verbose)

            # Revert to previous print setting
            models[1].settings[:alternative_policy].print = old_print
        end
    end

    return plots
end