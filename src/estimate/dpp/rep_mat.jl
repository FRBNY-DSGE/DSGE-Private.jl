using DSGE, DSGEModels, FileIO, CSV, StatsBase, Plots, StateSpaceRoutines, Dates, Random, MAT
# This script attempts to replicate line by line what happens in the
# particle filtering used by the Matlab code

# matlab values
mat_n_particles = 5000
mat_ρ = 0.5
mat_μ = 0.
mat_σ = 1.
mat_seed = 1793 # this probably doesn't work cuz they have different rng probably
mat_T = 5

filepath = dirname(@__FILE__)
savefile = "$filepath/save/julia_rep_mat.jld2"
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
datevec = Vector{Date}(y1.date[y1.date .>= Date("1991-06-30")])
y1 = Matrix{Float64}(Matrix(y1[y1.date .>= Date("1991-06-30"),:])[:,2:end]') # subset for desired data
y2 = CSV.read(get_setting(m2, :dataroot) * "realtime_spec=m904_hp=true_vint=170410.csv")
y2 = Matrix{Float64}(Matrix(y2[y2.date .>= Date("1991-06-30"),:])[:,2:end]') # subset for desired data
datevec = datevec[1:78]
matdata = matread(dataroot * "pred_dens_wrong.mat")
preddens1_1 = vec(matdata["p805"])
preddens2_1 = vec(matdata["p904"])
periods = 4
pm = PoolModel(Dict(:Model805 => y1[:,1:78], :Model904 => y2[:,1:78]), periods,
               Dict(:Model805 => preddens1_1, :Model904 => preddens2_1), [m2, m1]; static = false)
saveroot = "$filepath/../../../test/estimate/dpp/save/"
jld_data = load("$filepath/../../../test/reference/tpf_poolmodel.jld2")
tuning = jld_data["tuning"]
tuning[:get_t_particle_dist] = true
tuning[:allout] = true
tuning[:parallel] = true
tuning[:n_particles] = mat_n_particles
pm <= Setting(:tuning, tuning, "tuning parameters for TPF")
pm <= Setting(:sampling_method, :MH)
n_particles = get_setting(pm, :tuning)[:n_particles]
pm[:ρ].value = mat_ρ
pm[:μ].value = mat_μ
pm[:σ].value = mat_σ

T = length(datevec)
λhat_t = zeros(mat_T)
λhat_tplush = zeros(mat_T)
Random.seed!(mat_seed)
~, ~, ~, λ_particle_dist, λ_weights = DSGE.filter(pm, zeros(1,mat_T); tuning = tuning)
for t in mat_T:mat_T
    λhat_t[t] = mean(vec(λ_particle_dist[t][1,:]) .* λ_weights[:,t]) # get ̂λ_{t|t} before mutating particle distribution in period t
    for j in 1:get_forecast_horizon(pm)
        ϵ = rand(get_F_ϵ(pm), n_particles)
        for i in 1:n_particles
            λ_particle_dist[t][:,i] = get_Φ(pm)(λ_particle_dist[t][:,i], [ϵ[i]])
        end
    end
    λhat_tplush[t] = mean(vec(λ_particle_dist[t][1,:]) .* λ_weights[:,t])
end
# dpp_preddens = λhat_tplush .* preddens2_1 + (1 .- λhat_tplush) .* preddens1_1

gr()
# plot1 = plot(datevec, λhat_t)
# plot2 = plot(datevec, λhat_tplush)
# plot3 = plot(datevec, [log.(dpp_preddens), log.(preddens1_1), log.(preddens2_1)],
#              xlabel = "Date",
#              ylabel = "Log predictive densities",
#              plot_title = "Log score comparison for SWFF vs. SWπ",
#              label = ["DP", "SW Inflation", "SWFF"],
#              legend = :bottomleft)

λ_t_evol = zeros(n_particles, mat_T)
datemat = Matrix{Date}(undef,n_particles,mat_T)
for t in 1:mat_T
    λ_t_evol[:,t] = vec(λ_particle_dist[t][1,:])
    datemat[:,t] .= datevec[t]
end

i = 0
i += 1; histogram(λ_particle_dist[i][1,:]; weights = λ_weights[:,i], nbins = 15)

# save(savefile, "particle_dist", λ_particle_dist, "weights", λ_weights, "preddens805", preddens1_1, "preddens904", preddens2_1, "lambda_evol", λ_t_evol, "datemat", datemat, "datevec", datevec)

# plot4 = heatmap_posterior_λ_evolution(pm, datevec, λ_t_evol, λ_weights)
# plot5, λedges, dates, λwts, λmodes = surface_posterior_λ_evolution(pm, datevec, λ_t_evol, λ_weights)
