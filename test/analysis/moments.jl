using DSGE, Test, FileIO, Random, ModelConstructors

write_test_output = false
if VERSION < v"1.7"
    outver = ""
else
    # Julia 1.7 switched the default RNG from MersenneTwister to per-Task Xoshiro256++,
    # so sample_λ draws differ and need their own reference fixture.
    outver = "_version=1126"
end

m = AnSchorfheide()
@testset "Test moments" begin
    @test DSGE.moments(m.parameters[findfirst(x -> x.key==:τ, m.parameters)]) == (2.0, 0.5)
    @test DSGE.moments(m.parameters[findfirst(x -> x.key==:e_y, m.parameters)]) == (0.1159846, 0.0)
    @test DSGE.moments(m.parameters[findfirst(x -> x.key==:σ_R, m.parameters)]) == (0.4, 4.0)
end

# This script currently just tests sample_λ, compute_Eλ

fp = dirname(@__FILE__)
save_λmat_noparallel = load("$(fp)/../reference/moments_poolmodel_outputs$(outver).jld2", "lammat_noparallel")
save_λmat_parallel = load("$(fp)/../reference/moments_poolmodel_outputs$(outver).jld2", "lammat_parallel")
Random.seed!(1793)
m = PoolModel("ss1")
θs = load("$(fp)/../reference/moments_poolmodel_inputs.jld2", "thetas")[1:2,:]
pred_dens = load("$(fp)/../reference/moments_poolmodel_inputs.jld2", "pred_dens")
λmat_noparallel = sample_λ(m, pred_dens, θs, 1)
λmat_parallel   = sample_λ(m, pred_dens, θs, 1; parallel = true)

if !write_test_output
@testset "Check sample_λ works correctly" begin
    @test @test_matrix_approx_eq save_λmat_noparallel λmat_noparallel
    @test @test_matrix_approx_eq save_λmat_parallel λmat_parallel
end
end

λvec = 0.5 * ones(2)
θchange = [.8 0. 1.] .* ones(2)
λhat_plush, λhat_t = compute_Eλ(m, 4, λvec, θchange)
λhat_plush_parallel, λhat_t_parallel = compute_Eλ(m, 4, λvec, θchange; parallel = true)
if !write_test_output
@testset "Check compute_Eλ works correctly" begin
    @test λhat_plush == 0.5
    @test λhat_plush_parallel == 0.5
    @test λhat_t == 0.5
    @test λhat_t_parallel == 0.5
end
end

Random.seed!(1793)
sm = PoolModel("ss1"; weight_type = :static)
sm <= Setting(:saveroot, "$(fp)/../reference/")
sm <= Setting(:hessian_path, "$(fp)/../reference/mh_hessian_poolmodel.h5")
sm <= Setting(:n_mh_simulations, 1)
save_sλmat_noparallel = load("$(fp)/../reference/moments_poolmodel_outputs$(outver).jld2", "slammat_noparallel_one")
save_sλmat_parallel = load("$(fp)/../reference/moments_poolmodel_outputs$(outver).jld2", "slammat_parallel_one")
sλmat_noparallel = sample_λ(sm, pred_dens, 1)
sλmat_parallel   = sample_λ(sm, pred_dens, 1; parallel = true)
rm(joinpath(saveroot(sm), "output_data/poolmodel"); recursive = true)

if write_test_output
    save("$(fp)/../reference/moments_poolmodel_outputs$(outver).jld2",
         Dict("lammat_noparallel" => λmat_noparallel, "lammat_parallel" => λmat_parallel,
              "slammat_noparallel_one" => sλmat_noparallel, "slammat_parallel_one" => sλmat_parallel))
end

if !write_test_output
@testset "Check sample_λ works for a static pool" begin
    @test sλmat_noparallel == save_sλmat_noparallel
    @test sλmat_parallel == save_sλmat_parallel
end
end
