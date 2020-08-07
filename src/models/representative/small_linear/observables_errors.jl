# Prints recommended parameter values for "measurement error" as a proportion of current data vintage observable standard deviations
function print_observable_errors(df::DataFrame; prop::Float64=0.20)
    nan2missing!(df)
    println(string("e_y: ", prop*std(skipmissing(df.obs_gdp), corrected=true)))

    println(string("e_y_f: ", prop*std(skipmissing(df.obs_gdp_f), corrected=true)))

    println(string("e_π: ", prop*std(skipmissing(df.obs_cpi), corrected=true)))

    println(string("e_π_f: ", prop*std(skipmissing(df.obs_cpi_f), corrected=true)))

    println(string("e_r_n: ", prop*std(skipmissing(df.obs_nominalrate), corrected=true)))

    println(string("e_r_n_f: ", prop*std(skipmissing(df.obs_nominalrate_f), corrected=true)))

end
