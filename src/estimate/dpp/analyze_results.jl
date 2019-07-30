using DSGE, DSGEModels, FileIO, CSV, StatsBase, Plots, Dates, Random, MAT, Distributions, SharedArrays, JLD2
include("plot_fnct.jl")
include("compute_with_posterior.jl")

# Get estimation result
# estim_fn = "right805current904"
# save_fn = "right805_current904"
estim_fn = "wrongorigmatlab"
save_fn = "wrong805_orig904"

# Make poolmodel object
filepath = dirname(@__FILE__)
savepath = "$(filepath)/../../../test/estimate/dpp/save/"
datapath = "$(filepath)/../../../test/estimate/dpp/save/input_data/"
vint = "990110"
iter = 1
prev = 980110
est = 2
m1 = Model805()
m2 = Model904()
for model in [m1, m2]
    model <= Setting(:sampling_method, :SMC)
    model <= Setting(:saveroot, savepath)
    model <= Setting(:dataroot, datapath)
    model <= Setting(:data_vintage, vint, true, "vint", "")
    model <= Setting(:prev, prev, true, "prev", "")
    model <= Setting(:est, est, true, "est", "")
end

# Read in data for models
y1 = CSV.read(get_setting(m1, :dataroot) * "realtime_spec=m805_hp=true_vint=170410.csv")
datevec = y1.date[y1.date .>= Date("1992-03-31")]
y1 = Matrix{Float64}(Matrix(y1[y1.date .>= Date("1992-03-31"),:])[:,2:end]') # subset for desired data
y2 = CSV.read(get_setting(m2, :dataroot) * "realtime_spec=m904_hp=true_vint=170410.csv")
y2 = Matrix{Float64}(Matrix(y2[y2.date .>= Date("1992-03-31"),:])[:,2:end]') # subset for desired data
datevec = datevec[1:78]

# Load cond predictive densities
matdata = matread("save/input_data/pred_dens_$(save_fn).mat")
preddens1_1 = vec(matdata["p805"])
preddens2_1 = vec(matdata["p904"])

periods = 4
pm = PoolModel(Dict(:Model805 => y1[:,1:78], :Model904 => y2[:,1:78]), periods,
               Dict(:Model805 => preddens1_1, :Model904 => preddens2_1), [m2, m1]; static = false)
pm <= Setting(:data_vintage, "190726", true, "vint", "")

# Construct save path
pm <= Setting(:saveroot, "$(filepath)/save")
T = get_periods(pm)
h = get_forecast_horizon(pm)

# analyze smc output
pm <= Setting(:n_particles, 10)
λvec = Dict{Int64, Vector{Float64}}()
λ_t_evol = Matrix{Float64}(undef,get_setting(pm,:n_particles),T)
λhat_t = Vector{Float64}(undef,T) # E[λ_t|I_t^P, P]
λhat_tplush = Vector{Float64}(undef,T) # E[λ_{t+h}|I_t^P, P]
Random.seed!(1793)
for t in 1:T
    # for each theta particle, draw a random lambda particle's path
    # over 1:t from the likelihood particle filter
    # and choose only the time t set of theta particles
    θmat = load_draws(pm, :full; filestring_addl = ["period=$(t)", "preddens=$(estim_fn)"],
                      verbose = :none) # since resample from posterior, these particles should have equal weight
    println("Loading draws for period $(t)")

    # given posterior distribution of θ, sample λ
    # λ_t_evol[:,t] = sample_λ(pm, θmat, t)
    update_λ!(λ_t_evol, pm, θmat, t)
    if sum(λ_t_evol[:,t]) == 0
        break
    end
    λhat_tplush[t], λhat_t[t] = compute_Eλ(pm, λ_t_evol[:,t])
end
@assert false
dpp_preddens = λhat_tplush .* get_cond_pred_dens(pm, :Model904) .+
    (1 .- λhat_tplush) .* get_cond_pred_dens(pm, :Model805)
θmat_T = load_draws(pm, :full; filestring_addl = ["period=$(T)", "preddens=$(estim_fn)"])
datemat = Matrix{Date}(undef,get_setting(pm,:n_particles),T)
for t in 1:T
    datemat[:,t] .= datevec[t]
end

# Plot!
gr()
plot_datevec = Vector{Date}(datevec)
plot1 = plot(plot_datevec, [log.(get_cond_pred_dens(pm, :Model805)),
                       log.(get_cond_pred_dens(pm, :Model904))],
             xlabel = "Date",
             ylabel = "Log predictive densities",
             plot_title = "Log score comparison for SWFF vs. SWπ",
             label = ["SWπ", "SWFF"],
             legend = :bottomleft)
plot3 = plot(plot_datevec, [log.(dpp_preddens), log.(get_cond_pred_dens(pm, :Model805)),
                       log.(get_cond_pred_dens(pm, :Model904))],
             xlabel = "Date",
             ylabel = "Log predictive densities",
             plot_title = "Log score comparison for DP, SWFF, and SWπ",
             label = ["DP", "SW Inflation", "SWFF"],
             legend = :bottomleft)
plot4 = heatmap_posterior_λ_evolution(pm, plot_datevec, λ_t_evol)
plot5 = histogram(vec(θmat_T[:,1]))

# save analysis
analysis_savefile = saveroot(pm)
restofsave = "/output_data/poolmodel/ss0/estimate/work/"
JLD2.jldopen(analysis_savefile * restofsave * "analyze_realtime_estimation_preddens=$(estim_fn).jld2", true, true, true, IOStream) do file
    file["lambda_t_evol"] = λ_t_evol
    file["datemat"] = datemat
    file["datevec"] = datevec
    file["plot_datevec"] = plot_datevec
    file["dpp_preddens"] = dpp_preddens
    file["lambdahat_t"] = λhat_t
    file["lambdahat_tplush"] = λhat_tplush
    file["logscores_nodpp"] = plot1
    file["logscores_dpp"] = plot3
    file["heatmap"] = plot4
    file["posterior_rho"] = plot5
end

# save individual plots as pngs
png(plot1, analysis_savefile * restofsave * "log_pred_dens_swpi_swff=$(estim_fn).png")
png(plot3, analysis_savefile * restofsave * "log_pred_dens_compare_preddens=$(estim_fn).png")
png(plot4, analysis_savefile * restofsave * "heatmap_posterior_lambda_t_preddens=$(estim_fn).png")
png(plot5, analysis_savefile * restofsave * "posterior_rho_preddens=$(estim_fn).png")
