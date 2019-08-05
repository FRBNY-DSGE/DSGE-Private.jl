# Note that this test assumes TPF properly works
pm = PoolModel("ss0")
filepath = dirname(@__FILE__)
pm <= Setting(:dataroot, "$(filepath)/../reference/")
df = CSV.read(dataroot(pm) * "wrongorigmatlab.csv")
data = Matrix{Float64}(Matrix{Float64}(df[[:p904, :p805]])')

# This commented code produces the saved output
tuning = Dict(:r_star => 2., :c_init => 0.3, :target_accept_rate => 0.4,
              :resampling_method => :systematic, :n_mh_steps => 1,
              :n_particles => 1000, :n_presample_periods => 0,
              :allout => true)
pm <= Setting(:tuning, tuning, "tuning parameters for TPF")
pm[:ρ].value = 0.5
pm[:μ].value = 0.
pm[:σ].value = 1.
Random.seed!(1793)
Φfilt, Ψfilt, F_ϵfilt, F_ufilt, F_λfilt = compute_system(pm)
s_init = reshape(rand(F_λfilt, tuning[:n_particles]), 1, 1000)
s_init = [s_init; 1 .- s_init]
tpf_out, ~, ~ = tempered_particle_filter(data, Φfilt, Ψfilt, F_ϵfilt, F_ufilt,
                                   s_init; tuning..., verbose = :none,
                                   fixed_sched = [1.], parallel = false, poolmodel = true)

Random.seed!(1793)
filt_tpf_out, ~, ~ = DSGE.filter(pm, data; tuning = get_setting(pm, :tuning))
Random.seed!(1793)
filt_lik_tpf_out = sum(DSGE.filter_likelihood(pm, data; tuning = get_setting(pm, :tuning)))

@testset "Check call to tempered particle filter without providing initial states" begin
    @test tpf_out == filt_tpf_out
    @test tpf_out == filt_lik_tpf_out
end

Random.seed!(1793)
s_init = reshape(rand(F_λfilt, tuning[:n_particles]), 1, 1000)
s_init = [s_init; 1 .- s_init] # this tpf output should be saved later
filt_tpf_out, ~, ~ = DSGE.filter(pm, data, s_init; tuning = get_setting(pm, :tuning))
Random.seed!(1793)
s_init = reshape(rand(F_λfilt, tuning[:n_particles]), 1, 1000)
s_init = [s_init; 1 .- s_init] # this tpf output should be saved later
filt_lik_tpf_out = sum(DSGE.filter_likelihood(pm, data, s_init; tuning = get_setting(pm, :tuning)))

@testset "Check call to tempered particle filter when providing initial states" begin
    @test tpf_out == filt_tpf_out
    @test tpf_out == filt_lik_tpf_out
end


nothing
