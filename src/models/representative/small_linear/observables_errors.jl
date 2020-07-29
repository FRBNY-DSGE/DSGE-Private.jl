# Creates parameters for "measurement error" of observables as a proportion of observable sequence standard deviation
function init_observable_errors!(m::SmallLinear; prop::Float64=0.20)
    df = load_data(m)
    nan2missing!(df)
    m <= parameter(:e_y, prop*std(skipmissing(df.obs_gdp), corrected=true), fixed=true,
                   description="e_y: Measurement error on domestic GDP growth.",
                   tex_label="e_y")

    m <= parameter(:e_y_f, prop*std(skipmissing(df.obs_gdp_f), corrected=true), fixed=true,
                   description="e_y_f: Measurement error on foreign GDP growth.",
                   tex_label="e_y^f")

    m <= parameter(:e_π, prop*std(skipmissing(df.obs_cpi), corrected=true), fixed=true,
                   description="e_π: Measurement error on domestic inflation.",
                   tex_label="e_\\pi")

    m <= parameter(:e_π_f, prop*std(skipmissing(df.obs_cpi_f), corrected=true), fixed=true,
                   description="e_π_f: Measurement error on foreign inflation.",
                   tex_label="e_\\pi^f")

    m <= parameter(:e_r_n, prop*std(skipmissing(df.obs_nominalrate), corrected=true), fixed=true,
                   description="e_r_n: Measurement error on the domestic interest rate.",
                   tex_label="e_r^n")

    m <= parameter(:e_r_n_f, prop*std(skipmissing(df.obs_nominalrate_f), corrected=true), fixed=true,
                   description="e_r_n_f: Measurement error on the foreign interest rate.",
                   tex_label="e_r^{n,f}")

end
