using DSGE, ModelConstructors, Test, QuadGK

@testset "HetDSGEGovDebt with truncated lognormal distribution for g(e)" begin
    m = HetDSGEGovDebt("ss15"; ref_dir = HETDSGEGOVDEBT)

    @test get_setting(m, :gfunc_type) == :lognormal
    @test !m[:elo].fixed
    @test m[:μ_e].fixed
    @test m[:σ_e].fixed
    @test m[:ehi].fixed

    # Initialize μ_e so truncated lognormal mean does not equal 1
    gfunc = DSGE.construct_gfunc!(m)
    m[:μ_e].value = 0.
    @test !(quadgk(x -> x * gfunc(x), m[:elo].value, m[:ehi].value)[1] ≈ 1.)

    # Now update μ_e so truncated lognormal mean equals 1
    gfunc = DSGE.construct_gfunc!(m)
    gfunc_wbounds = DSGE.construct_gfunc_wbounds!(m; recalculate_μ = false)
    @test quadgk(x -> x * gfunc(x), m[:elo].value, m[:ehi].value)[1] ≈ 1.
    @test quadgk(x -> x * gfunc_wbounds(x, m[:ehi].value, m[:elo].value), m[:elo].value, m[:ehi].value)[1] ≈ 1.

    steadystate!(m)
    @test quadgk(x -> x * gfunc(x), m[:elo].value, m[:ehi].value)[1] ≈ 1. # ensure the mean is still 1
    @test quadgk(x -> x * gfunc_wbounds(x, m[:ehi].value, m[:elo].value), m[:elo].value, m[:ehi].value)[1] ≈ 1.
end
