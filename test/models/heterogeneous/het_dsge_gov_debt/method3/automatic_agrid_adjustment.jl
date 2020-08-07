using DSGE, Test, ModelConstructors

@testset "Automatic expansion of the agrid" begin
    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    m <= Setting(:ahi_incs, [1.])
    @test_throws DSGE.CashOnHandError DSGE.method3_steadystate!(m)

    m = HetDSGEGovDebt(; ref_dir =
                       joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
    m <= Setting(:ahi_incs, [1., 20., 30.])
    @test isnothing(DSGE.method3_steadystate!(m))
end
