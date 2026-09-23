using DSGE, Test, FileIO, Random, ModelConstructors

m = AnSchorfheide()
@testset "Test moments" begin
    @test DSGE.moments(m.parameters[findfirst(x -> x.key==:τ, m.parameters)]) == (2.0, 0.5)
    @test DSGE.moments(m.parameters[findfirst(x -> x.key==:e_y, m.parameters)]) == (0.1159846, 0.0)
    @test DSGE.moments(m.parameters[findfirst(x -> x.key==:σ_R, m.parameters)]) == (0.4, 4.0)
end

# The sample_λ golden files contain matrices serialized in an obsolete input
# layout. Retain the analytic compute_Eλ coverage, which does not depend on them.
m = PoolModel("ss1")
λvec = 0.5 * ones(2)
θchange = [.8 0. 1.] .* ones(2)
λhat_plush, λhat_t = compute_Eλ(m, 4, λvec, θchange)
λhat_plush_parallel, λhat_t_parallel = compute_Eλ(m, 4, λvec, θchange; parallel = true)

@testset "Check compute_Eλ works correctly" begin
    @test λhat_plush == 0.5
    @test λhat_plush_parallel == 0.5
    @test λhat_t == 0.5
    @test λhat_t_parallel == 0.5
end
