using DSGE, DSGEModels, FileIO, CSV, StatsBase, Plots, StateSpaceRoutines, Dates, Random, MAT
# This script fixes ρ = 0.9 and estimates the time path of λ

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

# Load loglhs here, second number is the data type, 1 -> no conditional on rate exp,
# 4 -> conditional on rate exp
# Based on the online appendix, it appears we should not condition on rate expectations
file_log1_1 = "m805_preddens/logscores_T0=1991-06-30_T=2016-06-30_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
file_log2_1 = "m904_preddens/logscores_T0=1991-06-30_T=2016-06-30_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
loglhs1_1 = load(get_setting(m1, :dataroot) * file_log1_1)["logscores"]
loglhs2_1 = load(get_setting(m2, :dataroot) * file_log2_1)["logscores"]
loglhs1_1 = vec(mean(loglhs1_1, dims = 1))
loglhs2_1 = vec(mean(loglhs2_1, dims = 1))

periods = 4
pm = PoolModel(Dict(:Model805 => y1, :Model904 => y2), periods,
               Dict(:Model805 => exp.(loglhs1_1), :Model904 => exp.(loglhs2_1)), [m1, m2]; static = true)
saveroot = "$filepath/../../../test/estimate/dpp/save/"
jld_data = load("$filepath/../../../test/reference/tpf_poolmodel.jld2")
tpf_out = jld_data["tpf_out"]
tpf_out_noinit = jld_data["tpf_out_noinit"]
tuning = jld_data["tuning"]
data = jld_data["data"]
s_init = jld_data["s_init"]
tuning[:get_t_particle_dist] = true
tuning[:allout] = true
pm <= Setting(:tuning, tuning, "tuning parameters for TPF")
pm <= Setting(:sampling_method, :SMC)
n_particles = get_setting(pm, :tuning)[:n_particles]
pm[:ρ].value = 0.9

T = length(datevec)
λhat_t = zeros(T)
λhat_tplush = zeros(T)
Random.seed!(1793)
~, ~, ~, λ_particle_dist, λ_weights = DSGE.filter(pm; tuning = tuning)
for t in 1:T
    λhat_t[t] = mean(vec(λ_particle_dist[t][1,:]) .* λ_weights[:,t]) # get ̂λ_{t|t} before mutating particle distribution in period t
    for j in 1:get_forecast_horizon(pm)
        ϵ = rand(get_F_ϵ(pm), n_particles)
        for i in 1:n_particles
            λ_particle_dist[t][:,i] = get_Φ(pm)(λ_particle_dist[t][:,i], [ϵ[i]])
        end
    end
    λhat_tplush[t] = mean(vec(λ_particle_dist[t][1,:]) .* λ_weights[:,t])
end
dpp_loglhs = log.(λhat_tplush .* exp.(loglhs1_1) + (1 .- λhat_tplush) .* exp.(loglhs2_1))

gr()
plot1 = plot(datevec, λhat_t)
plot2 = plot(datevec, λhat_tplush)
plot3 = plot(datevec, [dpp_loglhs, loglhs1_1, loglhs2_1])
