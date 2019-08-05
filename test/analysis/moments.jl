# This script currently just tests sample_λ, compute_Eλ


θs = load("../reference/moments_poolmodel_inputs.jld2", "thetas")
pred_dens = load("../reference/moments_poolmodel_inputs.jld2", "pred_dens")
save_λmat_noparallel = load("../reference/moments_poolmodel_outputs.jld2", "lammat_noparallel")
save_λmat_parallel = load("../reference/moments_poolmodel_outputs.jld2", "lammat_parallel")
Random.seed!(1793)
m = PoolModel("ss0")
m[:ρ].value = 0.8
m[:μ].value = 0.
m[:σ].value = 1.
λmat_noparallel = sample_λ(m, pred_dens, θs, 1)
λmat_parallel   = sample_λ(m, pred_dens, θs, 1; parallel = true)

@testset "Check sample_λ works correctly" begin
    @test @test_matrix_approx_eq save_λmat_noparallel λmat_noparallel
    @test @test_matrix_approx_eq save_λmat_parallel λmat_parallel
end

λvec = 0.5 * ones(1000)
λhat_plush, λhat_t = compute_Eλ(m, 4, λvec)
λhat_plush_parallel, λhat_t_parallel = compute_Eλ(m, 4, λvec; parallel = true)
save_λhatplush_noparallel = load("../reference/moments_poolmodel_outputs.jld2", "lamhat_plush_noparallel")
save_λhatplush_parallel = load("../reference/moments_poolmodel_outputs.jld2", "lamhat_plush_parallel")
save_λhat_noparallel = load("../reference/moments_poolmodel_outputs.jld2", "lamhat_noparallel")
save_λhat_parallel = load("../reference/moments_poolmodel_outputs.jld2", "lamhat_parallel")

@testset "Check compute_Eλ works correctly" begin
    @test λhat_plush == save_λhatplush_noparallel
    @test λhat_plush_parallel == save_λhatplush_parallel
    @test λhat_t == save_λhat_noparallel
    @test λhat_t_parallel == save_λhat_parallel
end
