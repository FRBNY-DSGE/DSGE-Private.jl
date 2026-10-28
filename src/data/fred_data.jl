"""
```
load_fred_data(m::AbstractDSGEModel; start_date="1959-03-31", end_date=prev_quarter())
```

Checks in `inpath(m, raw)` for a FRED dataset corresponding to `data_vintage(m)`.
If a FRED vintage exists on disk, any required FRED series that is contained therein will be
imported. All missing series will be downloaded directly from FRED using the *FredData*
package. The full dataset is written to the appropriate data vintage file and returned.

# Arguments
- `m::AbstractDSGEModel`: the model object
- `start_date`: starting date.
- `end_date`: ending date.
- `mnemonics`: series to load. Defaults to the model's `__FRED` series.

# Notes

The FRED API reports observations according to the quarter-start date. `load_fred_data`
returns data indexed by quarter-end date for compatibility with other datasets.
"""
function load_fred_data(m::AbstractDSGEModel;
                        start_date::Dates.Date = Dates.Date("1959-01-01", "y-m-d"),
                        end_date::Dates.Date   = prev_quarter(),
                        verbose::Symbol  = :low,
                        mnemonics::Union{Nothing, Vector{Symbol}} = nothing)

    data_series = parse_data_series(m)
    if !isnothing(mnemonics)
        # Caller chose the series (e.g. Haver series loaded from FRED when haver_<vint>.csv is missing)
    elseif haskey(data_series, :FRED)
        mnemonics = data_series[:FRED]
    else
        # Then no FRED data used, return DataFrame with dates
        return DataFrame(date = get_quarter_ends(start_date,end_date))
    end
    vint = data_vintage(m)
    dateformat = "yymmdd"

    # Set up dataset and labels
    missing_series = Vector{Symbol}()
    data = []

    datafile = inpath(m, "raw", "fred_$vint.csv")
    if isfile(datafile)

        # Read in dataset and check that the file contains data for the proper dates
        data = CSV.read(datafile, DataFrame)

        # Convert dates from strings to dates for date arithmetic
        format_dates!(:date, data)

        qstart = lastdayofquarter(start_date)
        qend = lastdayofquarter(end_date)

        if !in(qstart, data[!,:date]) || !in(qend, data[!,:date])
            println(verbose, :low, "FRED dataset on disk missing start or end date. Fetching data from FRED.")
            data = DataFrame(date = get_quarter_ends(start_date,end_date))
            missing_series = mnemonics
        else
            # If we have the right dates but the series isn't in the file, add it to
            # missing_series
            for series in mnemonics
                if !in(series, propertynames(data))
                    push!(missing_series,series)
                end
            end
        end
    else
        missing_series = mnemonics
        data = DataFrame(date = get_quarter_ends(start_date,end_date))
    end

    # Get the missing data series
    if !isempty(missing_series)

        # Have to do this wacky parsing to prepend the century to the data vintage
        vint_date = if parse(Int,vint[1:2]) < 59
            Year(2000) + Dates.Date(vint, dateformat)
        else
            Year(1900) + Dates.Date(vint, dateformat)
        end

        fredseries = Array{FredSeries, 1}(undef, length(missing_series))
        f = Fred()

        # A failed request (rate limit, timeout, maintenance page) used to fall back to
        # today's unvintaged series, or skip the series, and the file was written anyway,
        # so a pull could silently cache revised (look-ahead) or missing data. Now each
        # request is retried, then it is an error and nothing is written. Today's series is
        # used only when ALFRED says it keeps no vintage of the series, or when the
        # :fred_allow_unvintaged_fallback setting is true (e.g. old-vintage reruns).
        allow_unvintaged = get_setting_or(m, :fred_allow_unvintaged_fallback, false)
        for (i,s) in enumerate(missing_series)
            println(verbose, :low, "Fetching FRED series $s at vintage $vint_date...")
            fredseries[i] = fetch_fred_vintage(f, s, start_date, end_date, vint_date;
                                               allow_unvintaged = allow_unvintaged, verbose = verbose)
        end
        #= Old: fell back to the unvintaged series on any error, skipped series that still
           failed, and printed raw errors (which contain the API key)
        for (i,s) in enumerate(missing_series)
            println(verbose, :low, "Fetching FRED series $s...")
            try
                fredseries[i] = get_data(f, string(s); frequency="q",
                                                       observation_start=string(start_date),
                                                       observation_end=string(end_date),
                                                       vintage_dates=string(vint_date))
            catch err
                if :msg in fieldnames(typeof(err))
                    @warn err.msg
                else
                    show(err)
                end
                @warn "FRED series $s could not be fetched at vintage $vint."

                try
                    println(verbose, :low, "Fetching FRED series $s without vintage...")
                    fredseries[i] = get_data(f, string(s); frequency="q",
                                                           observation_start=string(start_date),
                                                           observation_end=string(end_date))
                catch err
                    if :msg in fieldnames(typeof(err))
                        @warn err.msg
                    else
                        show(err)
                    end
                    @warn "FRED series $s could not be fetched."
                    continue
                end
            end
        end
        =#

        # Extract dataframe from each series and merge on date
        has_oj = isdefined(DataFrames, :outerjoin)
        for i = 1:length(fredseries)
            if isassigned(fredseries, i)
                series = fredseries[i]
                series_id = Symbol(series.id)
                rename!(series.df, :value => series_id)
                map!(x->lastdayofquarter(x), series.df[!, :date], series.df[!, :date])
                data = has_oj ? outerjoin(data, series.df[!, [:date, series_id]], on = :date) :
                    join(data, series.df[!, [:date, series_id]], on = :date, kind = :outer)
            end
        end

        # Change the dates to be the last day of each quarter
        n_rows, n_cols = size(data)

        for i = 1:n_rows
            data[i,:date] = Dates.lastdayofquarter(data[i,:date])
        end

        if !m.testing
            CSV.write(datafile, data, missingstring = "")
            println(verbose, :low, "Updated data from FRED written to $datafile.")
        end
    end

    # Make sure to only return the series and dates that are specified for this
    # model (there may be additional series in the file)

    rows = start_date .<= data[!, :date] .<= end_date
    cols = [:date; mnemonics]
    return data[rows, cols]
end

# Seconds to wait between attempts at a FRED request
const FRED_RETRY_WAITS = [5, 15, 30, 60, 120]

get_setting_or(m::AbstractDSGEModel, key::Symbol, default) =
    (haskey(m.settings, key) || (m.testing && haskey(m.test_settings, key))) ? get_setting(m, key) : default

# FRED errors echo the request URL, which contains the API key: never print them raw.
# `short = true` keeps only the first line (no stacktrace fragment), for the retry lines.
function fred_error_message(err; n::Int = 300, short::Bool = false)
    msg = replace(sprint(showerror, err), r"api_key=[^&\s\"]*" => "api_key=<redacted>")
    short && (msg = first(split(strip(msg), '\n')))
    return first(msg, n)
end
#= Old: no single-line form
fred_error_message(err; n::Int = 300) =
    first(replace(sprint(showerror, err), r"api_key=[^&\s\"]*" => "api_key=<redacted>"), n)
=#

# Errors that a retry cannot fix: FRED's "Bad Request" (unknown series, bad arguments,
# invalid API key, no ALFRED vintage), "Not Found", and a vintage date outside FRED's
# range. Rate limits (429), server errors, timeouts and maintenance pages are retried.
fred_error_is_permanent(msg::AbstractString) =
    occursin("Bad Request", msg) || occursin("Not Found", msg) ||
    occursin(r"StatusError\(40[04]", msg) || occursin("cannot be outside the range", msg)

"""
```
fetch_fred_vintage(f, s, start_date, end_date, vint_date; allow_unvintaged = false, verbose = :low,
                   waits = FRED_RETRY_WAITS)
```
Fetch FRED series `s` as of `vint_date`, retrying failed requests (waiting `waits` seconds);
errors a retry cannot fix (`fred_error_is_permanent`) fail at once.
If every attempt fails it is an error, unless ALFRED reports that it keeps no vintage
of `s` or `allow_unvintaged` is true: then today's (unvintaged) series is fetched, with
a warning that it is not real-time.
"""
function fetch_fred_vintage(f, s::Symbol, start_date::Dates.Date, end_date::Dates.Date, vint_date::Dates.Date;
                            allow_unvintaged::Bool = false, verbose::Symbol = :low,
                            waits::Vector{<:Real} = FRED_RETRY_WAITS)
    kwargs = (frequency = "q", observation_start = string(start_date), observation_end = string(end_date))
    msg = ""
    for (attempt, wait) in enumerate(vcat(waits, 0))
        try
            return get_data(f, string(s); kwargs..., vintage_dates = string(vint_date))
        catch err
            isa(err, InterruptException) && rethrow()
            msg = fred_error_message(err)
            # A permanent error (incl. "does not exist in ALFRED") is not retried
            fred_error_is_permanent(msg) && break
            #= Old: only the ALFRED case skipped the retries (up to ~230 s per series)
            occursin("does not exist in ALFRED", msg) && break
            =#
            attempt > length(waits) && break
            println(verbose, :low, "FRED request for $s failed (attempt $attempt): $(fred_error_message(err; n = 200, short = true)). Retrying in $wait s...")
            sleep(wait)
        end
    end
    if occursin("does not exist in ALFRED", msg) || allow_unvintaged
        @warn "FRED series $s could not be fetched at vintage $vint_date ($(first(msg, 150))); " *
              "using today's unvintaged series, which is not real-time"
        try
            return get_data(f, string(s); kwargs...)
        catch err
            isa(err, InterruptException) && rethrow()
            error("FRED series $s could not be fetched at vintage $vint_date or unvintaged: " * fred_error_message(err))
        end
    end
    error("FRED series $s could not be fetched at vintage $vint_date after $(length(waits) + 1) " *
          "attempts; no data written. Last error: $msg. Set :fred_allow_unvintaged_fallback to use today's " *
          "series instead (look-ahead).")
end
