using DSGE, ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test
using Distributed

path = dirname(@__FILE__)

n_workers = 0   # no workers needed for estimate=false (load clouds + compute MDD only); set >0 to estimate
myprocs   = Int[]
if n_workers > 0
    ENV["frbnyjuliamemory"] = "6G"
    myprocs = addprocs_frbny(n_workers)
end
if nprocs() > 1
    @everywhere using DSGE, ModelConstructors, SMC, OrderedCollections
end

if VERSION < v"1.5"
    ver = "111"
elseif VERSION < v"1.6"
    ver = "150"
elseif VERSION < v"1.7"
    ver = "160"
else
    ver = "1126"
end

# set to true to (re)generate the clouds; flip back to false to run the test against saved ones.
# The full/old smc2 calls are isfile-guarded so they REUSE existing clouds (those don't depend on the
# bridge / prior_weight); only the online (m_new) run needs regenerating if the bridge config changes.
estimate = false


# instantiate model
m = AnSchorfheide()

save = normpath(joinpath(dirname(@__FILE__), "save"))
m <= Setting(:saveroot, save)


data = h5read(joinpath(path, "reference/smc.h5"), "data")

# model settings

m <= Setting(:n_particles, 400)
m <= Setting(:n_Φ, 100)
m <= Setting(:λ, 2.0)
m <= Setting(:n_smc_blocks, 1)
m <= Setting(:use_parallel_workers, nprocs() > 1)
m <= Setting(:step_size_smc, 0.5)
m <= Setting(:n_mh_steps_smc, 5)
m <= Setting(:resampler_smc, :polyalgo)
m <= Setting(:target_accept, 0.25)

m <= Setting(:mixture_proportion, 0.9)
m <= Setting(:adaptive_tempering_target_smc, false)
m <= Setting(:resampling_threshold, 0.5)
m <= Setting(:smc_iteration, 0)
m <= Setting(:use_chand_recursion, true)


verbose = :low
use_chand_recursion = true

# Estimate with full sample
m = deepcopy(m)
m <= Setting(:n_particles, 10000, true, "npart", "")
m <= Setting(:data_vintage, "210714")

# Plain seed (not @everywhere): in a single process @everywhere doesn't pin the task-local
# RNG the seeded SMC draws use, leaving the RNG references unreproducible.
Random.seed!(42)



savepath_full = rawpath(m, "estimate", "smc_cloud.jld2")

if estimate == true && !isfile(rawpath(m, "estimate", "smc_cloud.jld2"))
    DSGE.smc2(m, data; verbose = verbose, run_csminwel = false)   # reuse existing full cloud if present
end

# load in saved cloud and weights for full estimation
full_file   = load(rawpath(m, "estimate", "smc_cloud.jld2"))
full_cloud  = full_file["cloud"]
full_w      = full_file["w"]

# get marginal data density of full estimation
mdd_full = marginal_data_density(m, data)

# Estimate with 1st half of sample
m_old = deepcopy(m)
m_old <= Setting(:n_particles, 10000, true, "npart", "")
m_old <= Setting(:data_vintage, "000000")

savepath_old = rawpath(m_old, "estimate", "smc_cloud.jld2")
loadpath_old = rawpath(m_old, "estimate", "smc_cloud.jld2")



if estimate == true && !isfile(rawpath(m_old, "estimate", "smc_cloud.jld2"))
    println("Estimating Initial AnSchorfheide Model...")
    DSGE.smc2(m_old, data[:, 1:Int(floor(end/2))]; verbose = verbose, run_csminwel = false)
    println("Initial estimation done!")
end

old_file   = load(rawpath(m_old, "estimate", "smc_cloud.jld2"))
old_cloud  = old_file["cloud"]

# marginal data density of the first-half (old) estimation; passed to the online run below
mdd_old = marginal_data_density(m_old, data[:, 1:Int(floor(end/2))])

m_new = deepcopy(m)

# Estimate with 2nd half of sample
m_new <= Setting(:data_vintage, "200218")
# Exercise the prior/posterior bridge mixture. The SMC correction weights now
# include the normalized old posterior factor, so the saved incremental weights
# recover the full-data MDD without a separate bridge_vec correction.
m_new <= Setting(:tempered_update_prior_weight, 0.25)
m_new <= Setting(:tempered_update, true)
old_vint = "000000"
new_vint = "200218"

savepath_new = rawpath(m_new, "estimate", "smc_cloud.jld2")

m_new <= Setting(:previous_data_vintage, old_vint)

if estimate == true

    println("Beginning online estimation")
    DSGE.smc2(m_new, data; verbose = verbose, old_data = data[:,1:Int(floor(end/2))],old_cloud = old_cloud,
           old_model = m_old, log_prob_old_data = mdd_old, run_csminwel = false)
    println("Finished online estimation")
end

# (sequential run — no worker processes to remove)

# load in online cloud and weights
loadpath_new = replace(loadpath_old, r"vint=[0-9]{6}" => "vint=" * new_vint)
online_cloud = load(loadpath_new, "cloud")
online_w     = load(loadpath_new, "w")

# The bridge's saved weights include the old-data normalization, so this is
# already on the full-data MDD scale.
mdd_new = marginal_data_density(m_new, data)

@testset "Online Estimation: AnSchorf" begin
    @test abs(mdd_new - mdd_full) < 3
end

# Tear down only the workers this script self-spawned (leave any launched via `julia -p N`).
if !isempty(myprocs)
    rmprocs(myprocs)
end
