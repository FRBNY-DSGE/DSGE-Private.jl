using DSGE, DSGEModels, FileIO, CSV, StatsBase, Plots, Dates, StateSpaceRoutines, Random, MAT, Distributions
# This script estimates in real time

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

# Load loglhs here, second number is the data type, 1 -> no conditional on rate exp,
# 4 -> conditional on rate exp
# Based on the online appendix, it appears we should not condition on rate expectations
# file_log1_1 = "m805_preddens/logscores_T0=1992-03-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
# file_log2_1 = "m904_preddens/logscores_T0=1992-03-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
# loglhs1_1 = load(get_setting(m1, :dataroot) * file_log1_1)["logscores"]
# loglhs2_1 = load(get_setting(m2, :dataroot) * file_log2_1)["logscores"]
# loglhs1_1 = vec(mean(loglhs1_1, dims = 1))
# loglhs2_1 = vec(mean(loglhs2_1, dims = 1))
matdata = matread("save/input_data/pred_dens_right805_orig904.mat")
preddens1_1 = vec(matdata["p805"])
preddens2_1 = vec(matdata["p904"])

periods = 4
pm = PoolModel(Dict(:Model805 => y1[:,1:78], :Model904 => y2[:,1:78]), periods,
               Dict(:Model805 => preddens1_1, :Model904 => preddens2_1), [m2, m1]; static = false)

# Edit estimation to implement MH
tuning = get_setting(pm, :tuning)
tuning[:parallel] = true
pm <= Setting(:tuning, tuning)
pm <= Setting(:sampling_method, :MH)
pm <= Setting(:hessian_path, "$(filepath)/save/input_data/mh_hessian.h5")
pm <= Setting(:saveroot, "$(filepath)/save")

# Construct real time estimation of lambda (evolution over time)
h = get_forecast_horizon(pm)
T = get_periods(pm)
data = zeros(1,T)
print("Starting to run MH\n")
Random.seed!(1793)
for t in 1:T
    # run MH estimation
    DSGE.estimate(pm, data[:,1:t]; filestring_addl = ["period=$(t)", "preddens=right805orig904"],
                  proposal_covariance = [1 0 0; 0 0 0; 0 0 0])
end
