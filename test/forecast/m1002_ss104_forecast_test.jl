using DSGE, ModelConstructors, JLD2, CSV, DataFrames, Dates, Test
using ClusterManagers, HDF5, Plots, ModelConstructors, Nullables, Random, NLsolve

GR.inline("pdf")
include("/data/dsge_data_dir/dsgejl/$USER/proc/includeall.jl")
include("/data/dsge_data_dir/SystemwideDSGE/Current/System/sxg_var_decomp.jl")

# Forecasting
run_forecast             = true
forecast_or_decomp_type  = :mode
compute_mainfcast        = true
make_rd_draft_tables     = false
make_rd_draft_plots      = false

# Model Changeables
data_vint         = "250424"

data_vint_old     = "250227"

cond_vint         = "250424"
cond_vint_old     = "250227"

fcast_date        = DSGE.quartertodate("2025-Q1")
fcast_date_old    = DSGE.quartertodate("2025-Q1")

cond_type         = :full

forecast_string      = "ss104_test_2025Q1"


# Others
cond_full_names = [:obs_gdp, :obs_corepce, :obs_spread,
                   :obs_longrate, :obs_nominalrate,
                   :obs_exp_nominalrate1,
                   :obs_exp_nominalrate2,
                   :obs_exp_nominalrate3,
                   :obs_exp_nominalrate4,
                   :obs_exp_nominalrate5,
                   :obs_exp_nominalrate6]

cond_semi_names = [:obs_spread, :obs_longrate]


# Building the Model
custom_settings = [Setting(:date_forecast_start, fcast_date),
                   Setting(:date_conditional_end, fcast_date)]

m = Model1002("ss104", custom_settings = custom_settings)
usual_settings!(m, data_vint, cdvt = cond_vint, fcast_date = fcast_date, cond_full_names = cond_full_names, cond_semi_names = cond_semi_names)

overrides = forecast_input_file_overrides(m)
overrides[:mode] = "/data/dsge_data_dir/dsgejl/output_data/m1002/ss97/estimate/raw/" *
    "smc_cloud_bridge=true_fcastdate=2021Q2_iter=1_mhsteps=3_modeadj=1_nparts=15000_phi=0.95_priorwt=0.0_spreadadj=2_vint=210506.jld2"
overrides[:full] = "/data/dsge_data_dir/dsgejl/output_data/m1002/ss97/estimate/raw/" *
"smc_cloud_bridge=true_fcastdate=2021Q2_iter=1_mhsteps=3_modeadj=1_nparts=15000_phi=0.95_priorwt=0.0_spreadadj=2_vint=210506.jld2"
overrides[:semi] = "/data/dsge_data_dir/dsgejl/output_data/m1002/ss97/estimate/raw/" *
"smc_cloud_bridge=true_fcastdate=2021Q2_iter=1_mhsteps=3_modeadj=1_nparts=15000_phi=0.95_priorwt=0.0_spreadadj=2_vint=210506.jld2"

# Loading the dataframe
fomc_dates = implied_fomc_meeting_dates(Date(2028, 12, 31))
df = load_data(m; try_disk = false, check_empty_columns = false, cond_type = cond_type, fomc_dates = fomc_dates, builtin_mods = [:post_covid], cm_ffr = DataFrame())

# Stuff that still needs to be simplified
qtrs_before_last = DSGE.subtract_quarters(date_forecast_start(m), Date(2019, 12, 31))
irf_regime = get_setting(m, :cur_model_regime)

use_changing_systems = irf_regime >= 10


# Parameter draws
para  = forecast_params_wrapper!(m, forecast_or_decomp_type; rd_draft = make_rd_draft_tables || make_rd_draft_plots)


output_vars = Vector{Symbol}()

if compute_mainfcast
    append!(output_vars, [:histobs, :histpseudo, :histstdshocks, :forecastpseudo, :forecastobs,
                          :forecaststdshocks, :forecastutobs, :forecastutpseudo, :forecast4qpseudo,
                          :forecast4qobs, :hist4qobs, :hist4qpseudo, :histstates, :forecaststates,
                          :histstdshocks])
end

# Forecasting!
if run_forecast

    if forecast_or_decomp_type == :mode

        usual_forecast(m, :mode, cond_type, output_vars; df = df,
                           forecast_string = forecast_string,
                           density_bands = [0.5, 0.6, 0.68, 0.7, 0.8, 0.9],
                           mb_matrix = true, check_empty_columns = false,
                           params = para,#modal_params,
                           catch_smoother_lapack = true,
                           pseudo2data =
                           Dict{Symbol, Symbol}(:PseudoGDP => :obs_gdp, :PseudoCorePCE => :obs_corepce),
                           rerun_smoother = true, bdd_fcast = true,
                           set_regime_vals_altpolicy = (x,n) ->
                           model2para_covid_set_regime_vals(x, n;
                                                            start_regime = qtrs_before_last+1),
                           n_back = 0, back_shocks = [:AIT_sh], use_changing_systems = use_changing_systems)
    end

    if forecast_or_decomp_type == :full

        if auto_add_procs
            @show "Loading workers"
            ENV["frbnyjuliamemory"] = compute_shockdecs ? "20G" : "16G"
            myprocs = addprocs_frbny(n_workers)
            @everywhere using DSGE, OrderedCollections
            DSGE.sendto(workers(), USER=USER)
            @everywhere include("/data/dsge_data_dir/dsgejl/$USER/proc/includeall.jl")
        end

        @show "Running this"

        usual_forecast(m, :full, cond_type, output_vars; df = df,
                       forecast_string = forecast_string,
                       density_bands = [0.5, 0.6, 0.68, 0.7, 0.8, 0.9],
                       run_forecast = true,
                       run_meansbands = true, mb_matrix = true, check_empty_columns = false,
                       catch_smoother_lapack = true, skipnan = true,
                       params = para, pseudo2data = Dict{Symbol, Symbol}(:PseudoGDP => :obs_gdp, :PseudoCorePCE => :obs_corepce),
                       n_back = 0, back_shocks = [:AIT_sh], use_changing_systems = use_changing_systems,
                       rerun_smoother = true)
        if auto_add_procs
            rmprocs(myprocs)
        end

    end
end
