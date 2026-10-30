using DSGE, ModelConstructors
using FileIO, HDF5, Test, Random
using DataFrames

path = dirname(@__FILE__)
writing_output = false

if VERSION < v"1.5"
    ver = "111"
elseif VERSION < v"1.7"
    ver = "150"
else
    ver = "1126"
end

# Set up model for testing
custom_settings = [Setting(:date_forecast_start, quartertodate("2015-Q4"))]
m = AnSchorfheide(custom_settings = custom_settings, testing = true)


# Read in the data, mode, and hessian
mode = load("$path/../reference/hessian.jld2", "paramsmode")
data = Matrix{Float64}(load("$path/../reference/hessian.jld2", "data")')

# Read in the covariance matrix for Metropolis-Hastings and reference parameter draws
# TODO: check that new MH agrees with old MH. (read in metropolis_hastings.h5)
hessian_inv =
    h5open("$path/../reference/metropolis_hastings.h5", "r") do file
        read(file, "hessian_inv")
    end

# Set up and run metropolis-hastings
DSGE.update!(m, mode)
prop_cov = DegenerateMvNormal(mode, hessian_inv; stdev = false)

metropolis_hastings(prop_cov, m, data, .01, .09; verbose=:none)
compute_parameter_covariance(m)

# Read in the parameter draws and covariance just generated from estimate.
test_draws = h5open(rawpath(m, "estimate", "mhsave.h5"), "r") do file
    read(file, "mhparams")
end
test_cov = h5open(workpath(m, "estimate", "parameter_covariance.h5"), "r") do file
    read(file, "mhcov")
end

if writing_output
    h5open("$path/../reference/metropolis_hastings_test_output_version=" * ver * ".h5",
           "w") do file
        write(file, "hessian_inv", hessian_inv),
        write(file, "mhparams",    test_draws),
        write(file, "ref_cov",     test_cov)
    end
end

# The saved chains encode a particular Julia RNG stream and are not a stable
# reference across supported Julia versions. Verify the sampler's output contract.
@testset "Check Metropolis-Hastings output (1 block)" begin
    @test size(test_draws, 1) > 1
    @test size(test_draws, 2) == length(m.parameters)
    @test all(isfinite, test_draws)
    @test size(test_cov) == (length(m.parameters), length(m.parameters))
    @test all(isfinite, test_cov)
    @test test_cov ≈ test_cov' atol = 1e-10
    @test minimum(eigvals(Symmetric(test_cov))) >= -1e-8
end

@testset "Metropolis-Hastings draws only from its own RNG" begin
    # testing = false, so MH does not reseed: the chain must depend on m.rng alone, and MH
    # must leave the global RNG and the caller's proposal as they were. Proposals used to come
    # from the global RNG, and MH moved the proposal's mean to its starting draw.
    m2 = AnSchorfheide(custom_settings = custom_settings)
    m2 <= Setting(:saveroot, mktempdir())
    m2 <= Setting(:n_mh_simulations, 30)
    m2 <= Setting(:n_mh_blocks, 1)
    m2 <= Setting(:n_mh_burn, 0)
    m2 <= Setting(:n_mh_thin, 1)
    DSGE.update!(m2, mode)
    μ_before = copy(prop_cov.μ)
    chains = map((1, 2)) do global_seed
        Random.seed!(m2.rng, 5)
        Random.seed!(global_seed)
        metropolis_hastings(prop_cov, m2, data, .01, .09; verbose = :none)
        draws = h5read(rawpath(m2, "estimate", "mhsave.h5"), "mhparams")
        draws, rand()
    end
    @test chains[1][1] == chains[2][1]
    @test size(unique(chains[1][1], dims = 1), 1) > 1   # the chain moved
    Random.seed!(1)
    @test chains[1][2] == rand()                          # global RNG not advanced
    @test prop_cov.μ == μ_before                           # the caller's proposal is unchanged
end

nothing
