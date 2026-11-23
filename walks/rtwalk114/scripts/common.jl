# RTWALK114: copy of work/rtwalk104/scripts/common.jl. Changes (RTWALK114.md §2):
#   - W -> work/rtwalk114 (private dataroot copy input_data_pristine, saveroot);
#   - build_walk_model(fcast_date, ss; ...) = rtwalk104's build_ss104 with the subspec as an argument ("ss114" for the
#     walk; "ss104" gives rtwalk104's build exactly, used by the tests and the seed reweight). Every setting is the same
#     as build_ss104's: custom settings date_forecast_start / date_conditional_end / add_ant_markup_shocks_ind /
#     add_avgshortinfl, usual_settings!(261013) with the same cond lists, exp_ffr_data_start_date 2022-03-31,
#     σ_r_m..σ_r_m6 regime 2 fixed. (This is also exactly work/ss114/scripts/common.jl build_model with avg = false,
#     antmu = 0: same custom settings, COND_VINT = DATA_VINT = 261013, same COND_FULL without obs_avgshortinflation.)
#   - build_ss10 / build_ss207 / lik! dropped (unused by the walk).
using DSGE, ModelConstructors, Dates, CSV, DataFrames, OrderedCollections, Distributed
using Random, Printf, Statistics, LinearAlgebra
const W = "/data/dsge_data_dir/estimation-AI/work/rtwalk114"   # private dataroot copy input_data_pristine, saveroot
const HELP = "/data/dsge_data_dir/estimation-AI/reference/golden-step4/helpers/proc"  # frozen copy (= work/ss114/helpers/proc for these 2 files, byte-identical)
include(joinpath(HELP, "database/util.jl"))      # implied_fomc_meeting_dates
include(joinpath(HELP, "policy/drivers.jl"))     # usual_settings!
const DATA_VINT = "261013"
const FROZEN_DF = "/data/dsge_data_dir/estimation-AI/work/rtvint104/data_ss104_vint=261013_fcast2026Q3_full.csv"  # revised 261013 frame (read-only; = rtwalk104's)

const COND_FULL = [:obs_gdp, :obs_corepce, :obs_spread, :obs_longrate, :obs_nominalrate,
                   :obs_avgshortinflation, [Symbol("obs_exp_nominalrate$i") for i in 1:6]...]
const COND_SEMI = [:obs_spread, :obs_longrate]

# = rtwalk104 build_ss104 (= estimate_ss104_scratch.jl lines 44-69) with the subspec `ss` as an argument
function build_walk_model(fcast_date::Date, ss::AbstractString = "ss114"; avg::Bool = false, antmu::Int = 0,
                          dataroot = joinpath(W, "input_data_pristine") * "/", saveroot = joinpath(W, "saveroot") * "/")
    @assert ss in ("ss114", "ss104")
    cs = [Setting(:date_forecast_start, fcast_date),
          Setting(:date_conditional_end, fcast_date),
          Setting(:add_ant_markup_shocks_ind, antmu),
          Setting(:add_avgshortinfl, avg)]
    antmu > 0 && push!(cs, Setting(:fix_ant_markup_shocks_0, false))
    m = Model1002(ss, custom_settings = cs)
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

# data matrix (n_obs x T) for model m from a df, presample start .. lastq, NaN for absent columns (= rtwalk104)
function data_matrix(m, df::DataFrame, lastq::Date)
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
