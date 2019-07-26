using DSGE, DSGEModels, FileIO, CSV, StatsBase, Plots, StateSpaceRoutines, Dates, Random, MAT, Distributions
# This script fixes ρ = 0.9 and estimates the time path of λ using Metropolis-Hastings

print("Starting to run this file\n")
filepath = dirname(@__FILE__)
saveroot = "$filepath/../../../test/estimate/dpp/save/"
dataroot = "$filepath/../../../test/estimate/dpp/save/input_data/"
vint = "990110"
iter = 1
prev = 980110
est = 2
m1 = Model805()
m2 = Model904()
for model in [m1, m2]
    model <= Setting(:sampling_method, :SMC)
    model <= Setting(:saveroot, saveroot)
    model <= Setting(:dataroot, dataroot)
    model <= Setting(:data_vintage, vint, true, "vint", "")
    model <= Setting(:prev, prev, true, "prev", "")
    model <= Setting(:est, est, true, "est", "")
end

# Read in data for models
y1 = CSV.read(get_setting(m1, :dataroot) * "realtime_spec=m805_hp=true_vint=170410.csv")
datevec = y1.date[y1.date .>= Date("1991-06-30")]
y1 = Matrix{Float64}(Matrix(y1[y1.date .>= Date("1991-06-30"),:])[:,2:end]') # subset for desired data
y2 = CSV.read(get_setting(m2, :dataroot) * "realtime_spec=m904_hp=true_vint=170410.csv")
y2 = Matrix{Float64}(Matrix(y2[y2.date .>= Date("1991-06-30"),:])[:,2:end]') # subset for desired data
datevec = datevec[1:78]

# Load loglhs here, second number is the data type, 1 -> no conditional on rate exp,
# 4 -> conditional on rate exp
# Based on the online appendix, it appears we should not condition on rate expectations
# file_log1_1 = "m805_preddens/logscores_T0=1991-06-30_T=2016-06-30_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
# file_log2_1 = "m904_preddens/logscores_T0=1991-06-30_T=2016-06-30_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
# loglhs1_1 = load(get_setting(m1, :dataroot) * file_log1_1)["logscores"]
# loglhs2_1 = load(get_setting(m2, :dataroot) * file_log2_1)["logscores"]
# loglhs1_1 = vec(mean(loglhs1_1, dims = 1))
# loglhs2_1 = vec(mean(loglhs2_1, dims = 1))
matdata = matread(dataroot * "pred_dens_wrong.mat")
preddens1_1 = vec(matdata["p805"])
preddens2_1 = vec(matdata["p904"])

periods = 4
pm = PoolModel(Dict(:Model805 => y1[:,1:78], :Model904 => y2[:,1:78]), periods,
               Dict(:Model805 => preddens1_1, :Model904 => preddens2_1), [m2, m1]; static = false)
pm <= parameter(:ρ, 0.8, (1e-20,1-1e-7), (1e-20,1-1e-7), DSGE.SquareRoot(), Distributions.Uniform(0.,1.), fixed = false,
                description="ρ: persistence of AR processing underlying λ.",
                tex_label="\\rho")
pm <= Setting(:mh_cc, 0.15^2)
pm <= Setting(:mh_cc0, 0.15^2)
tuning = get_setting(pm, :tuning)
tuning[:parallel] = true
pm <= Setting(:tuning, tuning)
pm <= Setting(:sampling_method, :MH)
pm <= Setting(:calculate_hessian, false)
pm <= Setting(:reoptimize, false)
pm <= Setting(:n_mh_simulations, 1000)
# pm <= Setting(:n_mh_simulations, 100)
pm <= Setting(:n_mh_blocks, 1)
pm <= Setting(:mh_thin, 1)
pm <= Setting(:n_mh_burn, 0)
pm <= Setting(:hessian_path, "$filepath/mh_hessian.h5")
pm <= Setting(:saveroot, "$filepath/save")
print("Number of periods: " * string(get_periods(pm)) * "\n")
out = DSGE.estimate(pm, zeros(1, get_periods(pm)); proposal_covariance = [1 0 0; 0 0 0; 0 0 0])
