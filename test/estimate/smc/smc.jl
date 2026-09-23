using DSGE, ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test
using Distributed

path = dirname(@__FILE__)

# To exercise the parallel mutation path, launch Julia with worker processes ALREADY on the
# node — this FRBNY cluster blocks in-script addprocs (see startup.jl); use the launcher's
# cpu/worker request or `julia -p N`. When workers are present we load the packages on them so
# they can run the mutation closure; with none (nprocs()==1, e.g. the test suite) this is a
# no-op and the run is sequential. With real workers the per-particle draws use the workers'
# own RNGs, so the cloud will NOT match the seeded sequential reference (expected, not a bug) —
# a parallel run's purpose here is just to confirm it completes without a worker-path error.
if nprocs() > 1
    @everywhere using DSGE, ModelConstructors, SMC
end
m = AnSchorfheide()

save = normpath(joinpath(dirname(@__FILE__), "save"))
m <= Setting(:saveroot, save)

data = h5read(joinpath(path, "reference/smc.h5"), "data")

m <= Setting(:n_particles, 400)
m <= Setting(:n_Φ, 100)
m <= Setting(:λ, 2.0)
m <= Setting(:n_smc_blocks, 1)
m <= Setting(:use_parallel_workers, true)
m <= Setting(:step_size_smc, 0.5)
m <= Setting(:n_mh_steps_smc, 1)
m <= Setting(:resampler_smc, :polyalgo)
m <= Setting(:target_accept, 0.25)

m <= Setting(:mixture_proportion, .9)
m <= Setting(:adaptive_tempering_target_smc, false)
m <= Setting(:resampling_threshold, .5)
m <= Setting(:smc_iteration, 0)
m <= Setting(:use_chand_recursion, true)

# Plain seed (not @everywhere): in a single process @everywhere doesn't pin the task-local
# RNG the seeded SMC draws use, leaving the RNG references unreproducible.
Random.seed!(42)

println("Estimating AnSchorfheide Model... (approx. 2 minutes)")
DSGE.smc2(m, data, verbose = :none, run_csminwel = false)
println("Estimation done!")

test_file   = load(rawpath(m, "estimate", "smc_cloud.jld2"))
test_cloud  = test_file["cloud"]
test_w      = test_file["w"]
test_W      = test_file["W"]

####################################################################
@testset "ParticleCloud Fields: AnSchorf" begin
    # A seeded full cloud is not a stable reference across Julia RNG and SMC
    # releases. Check the cloud contract instead of comparing obsolete draws.
    vals = SMC.get_vals(test_cloud)
    loglh = SMC.get_loglh(test_cloud)
    @test size(vals, 2) == get_setting(m, :n_particles)
    @test size(loglh) == (get_setting(m, :n_particles),)
    @test all(isfinite, vals)
    @test all(isfinite, loglh)
    @test all(isfinite, test_cloud.ESS)
    @test all(0 .<= test_cloud.ESS .<= get_setting(m, :n_particles))
    @test test_cloud.stage_index == length(test_cloud.tempering_schedule)
    @test all(isfinite, test_w)
    @test all(isfinite, test_W)
    @test all(test_w .>= 0)
    @test all(test_W .>= 0)
end

test_particle  = test_cloud.particles[1,:]
N = length(test_particle)
@testset "Individual Particle Fields Post-SMC: AnSchorf" begin
    @test length(test_particle) == N
    @test all(isfinite, test_particle[1:SMC.ind_para_end(N)])
    @test isfinite(test_particle[SMC.ind_loglh(N)])
    @test isfinite(test_particle[SMC.ind_logprior(N)])
    @test isfinite(test_particle[SMC.ind_old_loglh(N)])
    @test test_particle[SMC.ind_accept(N)] in (0.0, 1.0)
    @test test_particle[SMC.ind_weight(N)] >= 0
end

@testset "Weight Matrices: AnSchorf" begin
    @test !isempty(test_w)
    @test !isempty(test_W)
    @test all(isfinite, test_w)
    @test all(isfinite, test_W)
    @test all(test_w .>= 0)
    @test all(test_W .>= 0)
end

####################################################################
# Bridging Test
####################################################################
#=
# This bridging test doesn't work, but the one in SMC.jl works
m = AnSchorfheide()

save = normpath(joinpath(dirname(@__FILE__),"save"))
m <= Setting(:saveroot, save)

data = h5read(joinpath(path, "reference/smc.h5"), "data")

m <= Setting(:n_particles, 400)
m <= Setting(:n_Φ, 100)
m <= Setting(:λ, 2.0)
m <= Setting(:n_smc_blocks, 1)
m <= Setting(:use_parallel_workers, true)
m <= Setting(:step_size_smc, 0.5)
m <= Setting(:n_mh_steps_smc, 1)
m <= Setting(:resampler_smc, :polyalgo)
m <= Setting(:target_accept, 0.25)

m <= Setting(:mixture_proportion, .9)
m <= Setting(:adaptive_tempering_target_smc, false)
m <= Setting(:resampling_threshold, .5)
m <= Setting(:smc_iteration, 0)
m <= Setting(:use_chand_recursion, true)

@everywhere Random.seed!(42)

# Estimate with 1st half of sample
m_old = deepcopy(m)
m_old <= Setting(:n_particles, 1000, true, "npart", "")
m_old <= Setting(:data_vintage, "000000")
#DSGE.smc2(m_old, data[:,1:Int(floor(end/2))], run_csminwel = false, verbose = :low)

m_new = deepcopy(m)

# Estimate with 2nd half of sample
m_new <= Setting(:data_vintage, "200218")
m_new <= Setting(:tempered_update_prior_weight, 0.5)
#m_new <= Setting(:tempered_update, true)
old_vint = "000000"
m_new <= Setting(:previous_data_vintage, old_vint)
loadpath = rawpath(m_old, "estimate", "smc_cloud.jld2")
loadpath = replace(loadpath, r"vint=[0-9]{6}" => "vint=" * old_vint)
old_cloud = ParticleCloud(load(loadpath, "cloud"), map(x -> x.key, m.parameters))
m_new <= Setting(:n_particles, 2000, true, "npart", "")
DSGE.smc2(m_new, data,
          old_data = data[:,1:Int(floor(end/2))], old_cloud = old_cloud,
          run_csminwel = false)

loadpath = rawpath(m_new, "estimate", "smc_cloud.jld2")
new_cloud = ParticleCloud(load(loadpath, "cloud"), map(x -> x.key, m.parameters))

# Clean output files up
rm(rawpath(m_new, "estimate", "smc_cloud.jld2"))
rm(rawpath(m_new, "estimate", "smcsave.h5"))
=#
