# RTWALK104: copy of work/rtvint104/scripts/common.jl. Changes: W, FROZEN_DF (read from rtvint104, read-only), header.
# vintage 261013. Private dataroot (copy of input_data_pristine) and saveroot under W; nothing
# written to dsgejl/ or SystemwideDSGE/.
using DSGE, ModelConstructors, Dates, CSV, DataFrames, OrderedCollections, Distributed
using Random, Printf, Statistics, LinearAlgebra
const W = "/data/dsge_data_dir/estimation-AI/work/rtwalk104"   # private dataroot copy input_data_pristine (read-only), saveroot
const HELP = "/data/dsge_data_dir/estimation-AI/reference/golden-step4/helpers/proc"  # frozen copy, = elena/proc for these 2 files
include(joinpath(HELP, "database/util.jl"))      # implied_fomc_meeting_dates
include(joinpath(HELP, "policy/drivers.jl"))     # usual_settings!
const DATA_VINT = "261013"
const LAST_Q    = Date(2019, 12, 31)
const FCAST19   = Date(2020, 3, 31)
const FROZEN_DF = "/data/dsge_data_dir/estimation-AI/work/rtvint104/data_ss104_vint=261013_fcast2026Q3_full.csv"  # revised 261013 frame (read-only)

function private_dataroot(tag)
    dr = joinpath(W, "input_data_$tag")
    isdir(dr) && rm(dr; recursive = true)
    cp(joinpath(W, "input_data_pristine"), dr)
    return dr * "/"
end

const COND_FULL = [:obs_gdp, :obs_corepce, :obs_spread, :obs_longrate, :obs_nominalrate,
                   :obs_avgshortinflation, [Symbol("obs_exp_nominalrate$i") for i in 1:6]...]
const COND_SEMI = [:obs_spread, :obs_longrate]

# ss104 exactly as estimate_ss104_scratch.jl (lines 44-69), with fcast date and add_avgshortinfl as arguments
# antmu = 2: Elena's script (anticipated mark-up shocks on in 2025Q2/Q3). antmu = 0: Marco's variant
# "ss104noSPFnoAntMU" (with avg = false): no λ_f_tl states / λ_f_ant_sh shocks / σ_λ_f1, σ_λ_f2.
function build_ss104(fcast_date::Date; avg::Bool = true, antmu::Int = 2, dataroot = joinpath(W, "input_data_pristine") * "/",
                     saveroot = joinpath(W, "saveroot") * "/")
    cs = [Setting(:date_forecast_start, fcast_date),
          Setting(:date_conditional_end, fcast_date),
          Setting(:add_ant_markup_shocks_ind, antmu),
          Setting(:add_avgshortinfl, avg)]
    antmu > 0 && push!(cs, Setting(:fix_ant_markup_shocks_0, false))
    m = Model1002("ss104", custom_settings = cs)
    usual_settings!(m, DATA_VINT, cdvt = DATA_VINT, fcast_date = fcast_date,
                    cond_full_names = avg ? COND_FULL : setdiff(COND_FULL, [:obs_avgshortinflation]),
                    cond_semi_names = COND_SEMI)
    m <= Setting(:dataroot, dataroot)
    m <= Setting(:saveroot, saveroot)
    m <= Setting(:exp_ffr_data_start_date, Date(2022, 3, 31))
    m <= Setting(:date_conditional_end, fcast_date)
    m <= Setting(:fcast_date, "$(year(fcast_date))Q$(DSGE.datetoquarter(fcast_date))", true, "fcastdate", "")
    for k in [:σ_r_m, :σ_r_m1, :σ_r_m2, :σ_r_m3, :σ_r_m4, :σ_r_m5, :σ_r_m6]
        m[k].regimes[:fixed] = OrderedDict(1 => false, 2 => true)
    end
    return m
end

function build_ss10(fcast_date::Date)
    m = Model1002("ss10", custom_settings = [Setting(:date_forecast_start, fcast_date),
                                             Setting(:date_conditional_end, fcast_date)])
    usual_settings!(m, DATA_VINT, cdvt = DATA_VINT, fcast_date = fcast_date)
    m <= Setting(:dataroot, joinpath(W, "input_data_pristine") * "/")
    m <= Setting(:saveroot, joinpath(W, "saveroot") * "/")
    return m
end

# ss207: the subspec plus the model custom settings of the ss207 seed spec
# (SystemwideDSGE/Estimation/System_2024_Q2/ss207/estimation_files/estim1002_ss207_2025Q1_noFFRmeaserr_antcovid_filterregimes_scratch.jl:213-244, 559-570)
function build_ss207(fcast_date::Date)
    cs = [Setting(:date_forecast_start, fcast_date),
          Setting(:forecast_horizons, 40),
          Setting(:add_pseudo_gdp, true), Setting(:add_pseudo_corepce, true),
          Setting(:add_iid_cond_obs_gdp_meas_err, false),
          Setting(:add_iid_cond_obs_corepce_meas_err, false),
          Setting(:standard_shocks_mode_adjust, 1, false, "modeadj", ""),
          Setting(:standard_shocks_spread_adjust, 2, false, "spreadadj", ""),
          Setting(:add_altpolicy_pgap, true), Setting(:add_altpolicy_ygap, true),
          Setting(:add_shadow_taylor, true), Setting(:add_κ_covid, true), Setting(:add_κ_pce, false),
          Setting(:expected_ffr, [1, 2, 3, 4, 5, 6]), Setting(:add_taylor_rm, true)]
    m = Model1002("ss207", custom_settings = cs)
    usual_settings!(m, DATA_VINT, cdvt = DATA_VINT, fcast_date = fcast_date)
    m <= Setting(:dataroot, joinpath(W, "input_data_pristine") * "/")
    m <= Setting(:saveroot, joinpath(W, "saveroot") * "/")
    m2p = get_setting(m, :model2para_regime)
    for p in [:σ_r_m, :σ_r_m1, :σ_r_m2, :σ_r_m3, :σ_r_m4, :σ_r_m5, :σ_r_m6]
        m2p[p] = Dict(i => 1 for i in 1:get_setting(m, :n_regimes))
    end
    m <= Setting(:model2para_regime, m2p)
    m <= Setting(:covid_ant_equal_contemp, true)
    return m
end

# data matrix (n_obs x T) for model m from a df, presample start .. lastq, NaN for absent columns
function data_matrix(m, df::DataFrame, lastq::Date = LAST_Q)
    dts = Date.(df.date)
    sel = findall((dts .>= date_presample_start(m)) .& (dts .<= lastq))
    sel = sel[sortperm(dts[sel])]
    d = fill(NaN, n_observables(m), length(sel))
    missingcols = Symbol[]
    for (k, i) in m.observables
        if string(k) in names(df)
            d[i, :] = Float64.(coalesce.(df[sel, string(k)], NaN))
        else
            push!(missingcols, k)
        end
    end
    isempty(missingcols) || println("  data_matrix: columns absent from df (NaN): ", missingcols)
    return d
end

# likelihood as SMC's _my_likelihood_dsge (update!, clear the memo, likelihood(catch_errors = true))
function lik!(m, vals, data)
    ModelConstructors.toggle_regime!(m.parameters, 1)
    DSGE.update!(m, vals)
    m <= Setting(:preprocessed_transitions, Dict())
    return likelihood(m, data; sampler = false, catch_errors = true, verbose = :none)
end
