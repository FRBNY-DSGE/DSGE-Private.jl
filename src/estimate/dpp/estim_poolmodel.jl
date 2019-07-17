using DSGE, DSGEModels, FileIO, CSV, StatsBase, Plots
# This script estimates in real time

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
datevec = y1.date[y1.date .>= Date("1991-12-31")]
y1 = Matrix{Float64}(Matrix(y1[y1.date .>= Date("1991-12-31"),:])[:,2:end]') # subset for desired data
y2 = CSV.read(get_setting(m2, :dataroot) * "realtime_spec=m904_hp=true_vint=170410.csv")
y2 = Matrix{Float64}(Matrix(y2[y2.date .>= Date("1991-12-31"),:])[:,2:end]') # subset for desired data

# Load loglhs here, second number is the data type, 1 -> no conditional on rate exp,
# 4 -> conditional on rate exp
# Based on the online appendix, it appears we should not condition on rate expectations
file_log1_1 = "m805_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
file_log2_1 = "m904_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"

loglhs1_1 = load(get_setting(m1, :dataroot) * file_log1_1)["logscores"]
loglhs2_1 = load(get_setting(m2, :dataroot) * file_log2_1)["logscores"]
loglhs1_1 = vec(mean(loglhs1_1, dims = 1))
loglhs2_1 = vec(mean(loglhs2_1, dims = 1))
periods = 4
pm = PoolModel(Dict(:Model805 => y1, :Model904 => y2), periods,
               Dict(:Model805 => exp.(loglhs1_1), :Model904 => exp.(loglhs2_1)), [m1, m2])
saveroot = "$filepath/../../../test/estimate/dpp/save/"
jld_data = load("$filepath/../../../test/reference/tpf_poolmodel.jld2")
tpf_out = jld_data["tpf_out"]
tpf_out_noinit = jld_data["tpf_out_noinit"]
tuning = jld_data["tuning"]
data = jld_data["data"]
s_init = jld_data["s_init"]
pm <= Setting(:tuning, tuning, "tuning parameters for TPF")
pm <= Setting(:sampling_method, :SMC)
n_particles = get_setting(pm, tuning)[:n_particles]

# Construct real time estimation of lambda (evolution over time)
h = get_forecast_horizon(pm)
T = get_periods(pm)
Eλ_tplush = zeros(get_periods(pm)) # equation 25 in the period, tplush = t + h
λhat_t = Dict{Int64,Vector{Float64}}()
λhat_tplush = Dict{Int64,Vector{Float64}}()
dpp_loglhs = Vector{Float64}(undef,T)
Random.seed!(1793)
for t in 1:T
    # run smc estimation
    estimate(pm, data[:,1:t])

    # for each theta particle, draw a random lambda particle's path
    # over 1:t from the likelihood particle filter
    # and choose only the time t set of theta particles

    # given posterior distribution of lambda, iterate forward h periods
    λ_tplush = zeros(n_particles)
    for θ for 1:θ_vec
        DSGE.update!(pm, θ)
        @sync @distributed for particle in 1:n_particles
            for j = 1:h
                ϵ_j = rand(get_F_ϵ(pm), 1)
                λ_tplush[particle] = get_Φ(pm)(λ_tplush[particle], [ϵ_j])
            end
            λ_tplush[particle] *= weights[particle] # weights for approxing integral
        end
        Eλ_tplush[θ] = mean(λ_tplush)
    end
    # save these lambda draws and plot over time
end


# gr()
# fit(Histogram, λ_draws, bins=:fd, weights =
# plot(datevec
