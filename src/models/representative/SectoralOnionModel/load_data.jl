function load_onion_data(m::OnionModel; quarter_or_month::Symbol = get_setting(m, :data_quarter_or_month), first_run::Bool = false)

    sectoral_inflation_path = get_setting(m, :dataroot) * "raw/cpi" * (quarter_or_month == :quarter ? "quarter" : "month") * "_$(get_setting(m, :data_vintage)).csv"

    inflation_df = CSV.read(sectoral_inflation_path, DataFrame)

    if first_run

        inflation_df[!,:date] = map(x -> DSGE.quartertodate("$(x[1])-$(x[2])"), zip(inflation_df[!, :year], inflation_df[!, :quarter]))

        select!(inflation_df, Not([:year, :quarter]))
        CSV.write(sectoral_inflation_path, inflation_df)
    end

    df = load_data(m; try_disk = false, check_empty_columns = false, cond_type = :full, fomc_dates = implied_fomc_meeting_dates(Date(2028, 12, 31)), verbose = :low, inflation_df = inflation_df)

    return df
end



function load_data(m::OnionModel; cond_type::Symbol = :none, try_disk::Bool = true,
                   verbose::Symbol=:low, check_empty_columns::Bool = true,
                   summary_statistics::Symbol = :low, add_vals = (false, Date(2020,12,31)),
                   fomc_dates::Vector{Int64} = Vector{Int64}(),
                   builtin_mods::Vector{Symbol} = Vector{Symbol}(),
                   custom_mods::Vector{Any} = [], cm_ffr::DataFrame = DataFrame(),
                   recreate_data = false,
                   inflation_df = DataFrame())


    # Check if already downloaded
    if try_disk && has_saved_data(m; cond_type=cond_type)
        filename = get_data_filename(m, cond_type)
        print(verbose, :low, "Reading dataset $(filename) from disk...")
        df = read_data(m; cond_type = cond_type, check_empty_columns = check_empty_columns)
        if isvalid_data(m, df; cond_type = cond_type, check_empty_columns = check_empty_columns)
            println(verbose, :low, "dataset from disk valid")
        else
            println(verbose, :low, "dataset from disk not valid")
            recreate_data = true
        end
    else
        recreate_data = true
    end

    # Download routines
    if recreate_data
        println(verbose, :low, "Creating dataset...")

        levels = load_data_levels(m; verbose=verbose, add_vals = add_vals)

        if cond_type in [:semi, :full]
            cond_levels = load_cond_data_levels(m; verbose=verbose)
            levels, cond_levels = DSGE.reconcile_column_names(levels, cond_levels)
            levels = vcat(levels, cond_levels)
        end
        df = transform_data(m, levels; cond_type=cond_type, verbose=verbose)

        # Conditional OIS data
        if :obs_nominalrate1 in cond_semi_names(m) || :obs_nominalrate1 in cond_full_names(m)
            ois_data = CSV.read(inpath(m, "raw", "ois_$(data_vintage(m)).csv"), DataFrame, copycols = true)
            dates = DSGE.get_quarter_ends(iterate_quarters(date_mainsample_end(m), 1), date_conditional_end(m))
            n_cond = length(dates)

            ois_data_want = ois_data[date_mainsample_end(m) .< ois_data[!, :date] .<= date_conditional_end(m), [Symbol("ant$i") for i in 1:n_mon_anticipated_shocks(m)]]

            df[date_mainsample_end(m) .< df[!, :date] .<= date_conditional_end(m), [Symbol("obs_nominalrate$i") for i in 1:n_mon_anticipated_shocks(m)]] .= Matrix{Float64}(ois_data_want)
        end

        # Conditional SPD Data
        if any(occursin.("obs_exp_nominalrate", string.(vcat(cond_semi_names(m), cond_full_names(m)))))
            spd_data = CSV.read(inpath(m, "raw", "spd_raw_$(data_vintage(m)).csv"), DataFrame, copycols = true)
            # Transform into usable form
            spd_data = transform_spd_data(spd_data, fomc_dates = fomc_dates)
            CSV.write(inpath(m, "raw", "spd_$(data_vintage(m)).csv"), spd_data)

            dates = DSGE.get_quarter_ends(iterate_quarters(date_mainsample_end(m), 1), date_conditional_end(m))
            n_cond = length(dates)

            spd_data_want = spd_data[date_mainsample_end(m) .< spd_data[!, :date] .<= date_conditional_end(m), [Symbol("exp_ant$i") for i in expected_ffr(m)]]

            df[date_mainsample_end(m) .< df[!, :date] .<= date_conditional_end(m), [Symbol("obs_exp_nominalrate$i") for i in expected_ffr(m)]] .= Matrix{Float64}(spd_data_want)
        end

        # Ensure that only appropriate rows make it into the returned DataFrame.
        start_date = date_presample_start(m)
        end_date   = if cond_type in [:semi, :full]
            date_conditional_end(m)
        else
            date_mainsample_end(m)
        end
        df = df[start_date .<= df[!,:date] .<= end_date, :]

        DSGE.missing_cond_vars!(m, df; cond_type = cond_type,
                           check_empty_columns = check_empty_columns)

        # check that dataset is valid
        DSGE.isvalid_data(m, df; check_empty_columns = check_empty_columns)

        if !m.testing
            save_data(m, df; cond_type=cond_type)
        end
        println(verbose, :low, "dataset creation successful")

        # print summary statistics
        if summary_statistics == :low || summary_statistics == :high
            str_nondate_names = [string(name) for name in propertynames(df[:,2:end])]
            freq_nan_empty = zeros(size(df,2) - 1)
            for (colnum, name) in enumerate(propertynames(df[:,2:end]))
                is_missing_in_col = ismissing.(df[!,name])
                is_nan_in_col = isnan.(df[!,name][.!is_missing_in_col])
                n_miss = count(is_missing_in_col)
                n_nan  = count(is_nan_in_col)
                freq_nan_empty[colnum] = (n_nan + n_miss) / length(is_missing_in_col)
                println("$(name), Frequency of missing/NaNs: $(freq_nan_empty[colnum])")
                if summary_statistics == :high
                    colmean = mean(df[!,name][.!is_missing_in_col][.!is_nan_in_col])
                    colstd  = std(df[!,name][.!is_missing_in_col][.!is_nan_in_col])
                    println("$(name), Column Mean: $(colmean)")
                    println("$(name), Standard deviation: $(colstd)")
                end
            end
        end
    end

    # DSGE in-built adjustments to the dataframe.
    for mod in builtin_mods
        if mod == :post_covid
            df = DSGE.post_covid_data_mods(m, df, cond_type, fomc_dates; cm_ffr)
        end
        if mod == :pgbpid_estim
            df = ss104_estimation(df)
        end
    end

    # Custom adjustments to the dataframe
    for func in custom_mods
        df = func(m, df)
    end

    return df
end
