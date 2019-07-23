using DSGE, DSGEModels, FileIO, CSV, StatsBase, Plots
# This script estimates in real time

filepath = dirname(@__FILE__)
savepath = "$filepath/../../../test/estimate/dpp/save/"
datapath = "$filepath/../../../test/estimate/dpp/save/input_data/"
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
datevec = y1.date[y1.date .>= Date("1991-12-31")]
y1 = Matrix{Float64}(Matrix(y1[y1.date .>= Date("1991-12-31"),:])[:,2:end]') # subset for desired data
y2 = CSV.read(get_setting(m2, :dataroot) * "realtime_spec=m904_hp=true_vint=170410.csv")
y2 = Matrix{Float64}(Matrix(y2[y2.date .>= Date("1991-12-31"),:])[:,2:end]') # subset for desired data
datevec = datevec[1:78]

# Load loglhs here, second number is the data type, 1 -> no conditional on rate exp,
# 4 -> conditional on rate exp
# Based on the online appendix, it appears we should not condition on rate expectations
# file_log1_1 = "m805_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
# file_log2_1 = "m904_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
# loglhs1_1 = load(get_setting(m1, :dataroot) * file_log1_1)["logscores"]
# loglhs2_1 = load(get_setting(m2, :dataroot) * file_log2_1)["logscores"]
# loglhs1_1 = vec(mean(loglhs1_1, dims = 1))
# loglhs2_1 = vec(mean(loglhs2_1, dims = 1))
matdata = matread(datapath * "pred_dens_wrong.mat")
preddens1_1 = vec(matdata["p805"])
preddens2_1 = vec(matdata["p904"])

periods = 4
pm = PoolModel(Dict(:Model805 => y1[:,1:78], :Model904 => y2[:,1:78]), periods,
               Dict(:Model805 => preddens1_1, :Model904 => preddens2_1), [m2, m1]; static = false)

# Construct real time estimation of lambda (evolution over time)
h = get_forecast_horizon(pm)
T = get_periods(pm)
λmat = Dict{Int64, Vector{Float64}}()
λhat_t = Vector{Float64}(undef,T) # E[λ_t|I_t^P, P]
λhat_tplush = Vector{Float64}(undef,T) # E[λ_{t+h}|I_t^P, P]
λ_plots = Dict{Int64,Any}()
dpp_loglhs = Vector{Float64}(undef,T)
Random.seed!(1793)
for t in 1:T
    # run smc estimation
    estimate(pm, data[:,1:t])

    # for each theta particle, draw a random lambda particle's path
    # over 1:t from the likelihood particle filter
    # and choose only the time t set of theta particles
    θmat = load_draws(pm) # since resample from posterior, these particles should have equal weight

    # given posterior distribution of θ, sample λ
    λmat[t] = sample_λ(pm, θmat, t)
    λhat_tplush[t], λhat_t[t] = compute_Eλ(pm, λmat)
end

# Plot!
# hist_post_λ_evol = plot_posterior_λ_evolution(pm, λmat[t])
# pc = load(get_setting(pm, :saveroot) * "smc.jld2")
# plot_post_θ = plot_posterior_hyperparameter(pm, pc)
