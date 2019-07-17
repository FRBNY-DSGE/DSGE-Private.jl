using DSGEModels

# Set up underlying models
# filepath = pwd()
filepath = dirname(@__FILE__)
saveroot = filepath * "/dpp/save/"
dataroot = filepath * "/dpp/save/input_data/"
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
y1 = Matrix{Float64}(Matrix(y1[y1.date .>= Date("1991-12-31"),:])[:,2:end]') # subset for desired data
y2 = CSV.read(get_setting(m2, :dataroot) * "realtime_spec=m904_hp=true_vint=170410.csv")
y2 = Matrix{Float64}(Matrix(y2[y2.date .>= Date("1991-12-31"),:])[:,2:end]') # subset for desired data

# Load pred_dens here, second number is the data type, 1 -> no conditional on rate exp,
# 4 -> conditional on rate exp
# Based on the online appendix, it appears we should not condition on rate expectations
file_log1_1 = "m805_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
file_log2_1 = "m904_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"

pred_dens1_1 = exp.(load(get_setting(m1, :dataroot) * file_log1_1)["logscores"])
pred_dens2_1 = exp.(load(get_setting(m2, :dataroot) * file_log2_1)["logscores"])
pred_dens1_1 = vec(mean(pred_dens1_1, dims = 1))
pred_dens2_1 = vec(mean(pred_dens2_1, dims = 1))
periods = 4
pm = PoolModel(Dict(:Model805 => y1, :Model904 => y2), periods,
               Dict(:Model805 => pred_dens1_1, :Model904 => pred_dens2_1), [m1, m2])

# This commented code produces the saved output
tuning = Dict(:r_star => 2., :c_init => 0.3, :target_accept_rate => 0.4,
              :resampling_method => :systematic, :n_mh_steps => 1,
              :n_particles => 1000, :n_presample_periods => 0,
              :allout => true)
pm <= Setting(:tuning, tuning, "tuning parameters for TPF")
data = zeros(1, get_periods(pm))
Random.seed!(1793)
s_init = reshape(rand(get_F_λ(pm), tuning[:n_particles]), 1, 1000)
s_init = [s_init; 1 .- s_init] # this tpf output should be saved later
tpf_out, ~, ~ = tempered_particle_filter(data, get_Φ(pm), get_Ψ(pm), get_F_ϵ(pm), get_F_u(pm),
                                   s_init; tuning..., verbose = :none,
                                   fixed_sched = [1.], parallel = false,
                                   dynamic_measurement = true, poolmodel = true)
# jld_data = load("$filepath/../reference/tpf_poolmodel.jld2")
# tpf_out = jld_data["tpf_out"]
# tpf_out_noinit = jld_data["tpf_out_noinit"]
# tuning = jld_data["tuning"]
# data = jld_data["data"]
# s_init = jld_data["s_init"]
# pm <= Setting(:tuning, tuning, "tuning parameters for TPF")


Random.seed!(1793)
filt_tpf_out, ~, ~ = DSGE.filter(pm, data; tuning = get_setting(pm, :tuning))
Random.seed!(1793)
filt_nodata_tpf_out, ~, ~ = DSGE.filter(pm; tuning = get_setting(pm, :tuning))
Random.seed!(1793)
filt_lik_tpf_out = sum(DSGE.filter_likelihood(pm, data; tuning = get_setting(pm, :tuning)))

@testset "Check call to tempered particle filter without providing initial states" begin
    @test tpf_out == filt_tpf_out
    @test tpf_out == filt_nodata_tpf_out
    @test tpf_out == filt_lik_tpf_out
end

Random.seed!(1793)
filt_tpf_out, ~, ~ = DSGE.filter(pm, data, s_init; tuning = get_setting(pm, :tuning))
Random.seed!(1793)
filt_lik_tpf_out = sum(DSGE.filter_likelihood(pm, data, s_init; tuning = get_setting(pm, :tuning)))

@testset "Check call to tempered particle filter when providing initial states" begin
    @test tpf_out_noinit == filt_tpf_out
    @test tpf_out_noinit == filt_lik_tpf_out
end


nothing
