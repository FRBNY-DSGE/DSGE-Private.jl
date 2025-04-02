"""
'''
post_covid_data_mods(m, df, cond_type, fomc_dates; cm_ffr::DataFrame = DataFrame(),
                              spd_expect_data::Bool = true, keep_Q1_spd::Bool = true, add_22Q1_ffr::Bool = true,
                              expected_ffr::Vector{Integer} = [1,2,3,4,5,6],
                              all_ffr_qs::Vector{Integer} = [1,2,3,4,5,6],
                              fcast_date::Date = date_forecast_start(m))
'''

Neccesary manual adjustments to the dataframe when implementing subspecs 97 and above to address the covid and post-covid era.

"""
function post_covid_data_mods(m, df, cond_type, fomc_dates; cm_ffr::DataFrame = DataFrame(),
                              spd_expect_data::Bool = true, keep_Q1_spd::Bool = true, add_22Q1_ffr::Bool = false,
                              expected_ffr::Vector{Int64} = [1,2,3,4,5,6],
                              all_ffr_qs::Vector{Int64} = [1,2,3,4,5,6],
                              fcast_date::Date = date_forecast_start(m))

    for i in 1:n_mon_anticipated_shocks(m)
        df[end, "obs_nominalrate$(i)"] = NaN
        if (cond_type == :semi || cond_type == :full)
            df[end-1, "obs_nominalrate$(i)"] = NaN
        end
    end

    df = df[date_presample_start(m) .<= df[!, :date],:]

    # Set ZLB forecasted periods to missing
    if spd_expect_data
        exp_inds = findall(x -> occursin("obs_exp_nominalrate", x), names(df))
        if !keep_Q1_spd
            df[1:end-1, exp_inds] .= missing
        else
            q2_expected_ffr = expected_ffr #.- 1
            q3_expected_ffr = expected_ffr
            deleteat!(q2_expected_ffr, q2_expected_ffr .== 0)
            q4_expected_ffr = expected_ffr #.+ 1
            deleteat!(q4_expected_ffr, q4_expected_ffr .== 0)
            # handle the missings
            for i in all_ffr_qs[findall(!in(expected_ffr), all_ffr_qs)]
                df[end-1, Symbol("obs_exp_nominalrate$(i)")] = missing
            end
            for i in all_ffr_qs[findall(!in(q2_expected_ffr), all_ffr_qs)]
                df[end, Symbol("obs_exp_nominalrate$(i)")] = missing
            end
            for i in all_ffr_qs[findall(!in(q4_expected_ffr), all_ffr_qs)]
                df[end-2, Symbol("obs_exp_nominalrate$(i)")] = missing
            end
            for i in all_ffr_qs
                df[1:end-3, Symbol("obs_exp_nominalrate$(i)")] .= missing
            end
        end
        rm_inds = findall(x -> ismissing(x) || x <= 0.033, Matrix(df[:,exp_inds]))
        for k in rm_inds
            df[k[1],k[2]+minimum(exp_inds)-1] = missing
        end
    end

    if spd_expect_data && fcast_date >= Date(2022,4,1) && cond_type == :none
        antffr_may = [0.19939,0.435,0.523,0.643,0.746,0.783]
        for i in mon_ant_ait_shocks
            df[end,"obs_exp_nominalrate$(i)"] = antffr_may[i]
        end
    end

    if spd_expect_data && add_22Q1_ffr && !isempty(cm_ffr)
        col_ind = 2 #it's 4 if using 220314.csv
        exp_cm_vec = zeros(10)
        if false## Last day
            exp_cm_vec[2] = cm_ffr[findfirst(x -> x == "9/30/2022", cm_ffr[!,:Maturity]), col_ind]
            exp_cm_vec[6] = cm_ffr[findfirst(x -> x == "9/30/2023", cm_ffr[!,:Maturity]), col_ind]
            exp_cm_vec[10] = cm_ffr[findfirst(x -> x == "9/30/2024", cm_ffr[!,:Maturity]), col_ind]
        else ## Average
            cm_ffr[!,:Maturity] .= Date.(cm_ffr[!,:Maturity], DateFormat("m/d/y"))
            exp_cm_vec[2] = mean(skipmissing(cm_ffr[findall(x -> x <= Date(2022,9,30) && x >= Date(2022,7,1), cm_ffr[!,:Maturity]), col_ind]))
            exp_cm_vec[3] = mean(skipmissing(cm_ffr[findall(x -> x > Date(2022,9,30) && x <= Date(2022,12,31), cm_ffr[!,:Maturity]), col_ind]))
            exp_cm_vec[6] = mean(skipmissing(cm_ffr[findall(x -> x <= Date(2023,9,30) && x >= Date(2023,7,1), cm_ffr[!,:Maturity]), col_ind]))
            exp_cm_vec[10] = mean(skipmissing(cm_ffr[findall(x -> x <= Date(2024,9,30) && x >= Date(2024,7,1), cm_ffr[!,:Maturity]), col_ind]))
        end

        for i in expected_ffr
            df[end, Symbol("obs_exp_nominalrate$(i)")] = exp_cm_vec[i] / 4.0
        end
    end


    pgap_ygap_init_date = Date(2020, 6, 30)
    start_zlb_date = Date(2020, 12, 31)

    df[!, :obs_pgap] .= NaN
    df[!, :obs_ygap] .= NaN

    ind_init = findfirst(df[!, :date] .== pgap_ygap_init_date)
    df[ind_init, :obs_pgap] = -0.125
    df[ind_init, :obs_ygap] = -12.

    start_ind = findfirst(df[!, :date] .== start_zlb_date)
    if !isnothing(start_ind)
        end_ind = findfirst(df[!, :date] .== Date(2021,12,31))#(cond_type == :none ? date_conditional_end(m) : date_forecast_start(m))) # ZLB ends in 2021Q4
        inds_tempzlb = start_ind:(isnothing(end_ind) ? size(df, 1) : end_ind)
        df[inds_tempzlb, :obs_nominalrate] .= NaN
        df[inds_tempzlb, [Symbol("obs_nominalrate$i") for i in 1:n_mon_anticipated_shocks(m)]] .= NaN
    end

    return df
end




function ss104_estimation(df)
    return df
end
