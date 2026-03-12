using Test, DSGE, Random, OrderedCollections

@testset "Macros for retrieving and converting steady-state deviations" begin
    Random.seed!(1793)
    x = rand(3)
    xss = rand(3)
    nt = (a = xss[1:2], b = xss[3])
    id = OrderedDict{Symbol, UnitRange{Int}}(:a => 1:2, :b => 3:3, :a′ => 1:2, :b′ => 3:3)

    @test isa(DSGE.variable2index2value(x, id, Val(:a)), SubArray)
    @test DSGE.variable2index2value(x, id, Val(:a)) == @view x[id[:a]]
    @test isa(DSGE.variable2index2value(x, id, Val(:b)), Number)
    @test !isa(DSGE.variable2index2value(x, id, Val(:b)), SubArray)
    @test DSGE.variable2index2value(x, id, Val(:b)) == x[id[:b][1]]

    DSGE.@variables2indices2values a, b = x, id
    @test a == @view x[id[:a]]
    @test b == x[id[:b][1]]

    @test isa(DSGE.sslogdeviation2level(x, id, nt, Val(:a)), Array)
    @test DSGE.sslogdeviation2level(x, id, nt, Val(:a)) == exp.((@view x[id[:a]]) + nt[:a])
    @test isa(DSGE.sslogdeviation2level(x, id, nt, Val(:b)), Number)
    @test !isa(DSGE.sslogdeviation2level(x, id, nt, Val(:b)), Array)
    @test DSGE.sslogdeviation2level(x, id, nt, Val(:b)) == exp((x[id[:b][1]]) + nt[:b])

    DSGE.@sslogdeviations2levels a, b = x, id, nt
    @test a == exp.((@view x[id[:a]]) + nt[:a])
    @test b == exp((x[id[:b][1]]) + nt[:b])

    @test isa(DSGE.sslogdeviation2level_unprimekeys(x, id, nt, Val(:a′)), Array)
    @test DSGE.sslogdeviation2level_unprimekeys(x, id, nt, Val(:a′)) == exp.((@view x[id[:a′]]) + nt[:a])
    @test isa(DSGE.sslogdeviation2level_unprimekeys(x, id, nt, Val(:b′)), Number)
    @test !isa(DSGE.sslogdeviation2level_unprimekeys(x, id, nt, Val(:b′)), Array)
    @test DSGE.sslogdeviation2level_unprimekeys(x, id, nt, Val(:b′)) == exp((x[id[:b′][1]]) + nt[:b])

    DSGE.@sslogdeviations2levels_unprimekeys a′, b′ = x, id, nt
    @test a′ == exp.((@view x[id[:a′]]) + nt[:a])
    @test b′ == exp((x[id[:b′][1]]) + nt[:b])

    @test isa(DSGE.sslogdeviation2log(x, id, nt, Val(:a)), Array)
    @test DSGE.sslogdeviation2log(x, id, nt, Val(:a)) == ((@view x[id[:a]]) + nt[:a])
    @test isa(DSGE.sslogdeviation2log(x, id, nt, Val(:b)), Number)
    @test !isa(DSGE.sslogdeviation2log(x, id, nt, Val(:b)), Array)
    @test DSGE.sslogdeviation2log(x, id, nt, Val(:b)) == ((x[id[:b][1]]) + nt[:b])

    DSGE.@sslogdeviations2logs a, b = x, id, nt
    @test a == ((@view x[id[:a]]) + nt[:a])
    @test b == ((x[id[:b][1]]) + nt[:b])

    @test isa(DSGE.ssdeviation2level(x, id, nt, Val(:a)), Array)
    @test DSGE.ssdeviation2level(x, id, nt, Val(:a)) == ((@view x[id[:a]]) + nt[:a])
    @test isa(DSGE.ssdeviation2level(x, id, nt, Val(:b)), Number)
    @test !isa(DSGE.ssdeviation2level(x, id, nt, Val(:b)), Array)
    @test DSGE.ssdeviation2level(x, id, nt, Val(:b)) == ((x[id[:b][1]]) + nt[:b])

    DSGE.@ssdeviations2levels a, b = x, id, nt
    @test a == ((@view x[id[:a]]) + nt[:a])
    @test b == ((x[id[:b][1]]) + nt[:b])

    @test isa(DSGE.get_deviation(x, id, Val(:a)), SubArray)
    @test DSGE.get_deviation(x, id, Val(:a)) == (@view x[id[:a]])
    @test isa(DSGE.get_deviation(x, id, Val(:b)), Number)
    @test !isa(DSGE.get_deviation(x, id, Val(:b)), Array)
    @test DSGE.get_deviation(x, id, Val(:b)) == (x[id[:b][1]])

    DSGE.@get_deviations a, b = x, id
    @test a == (@view x[id[:a]])
    @test b == (x[id[:b][1]])

    id = OrderedDict{Symbol, Int}(:a => 1, :b => 3, :a′ => 1, :b′ => 3)
    nt = (a = xss[1], b = xss[3])

    @test DSGE.variable2index2value(x, id, Val(:a)) == x[id[:a]]
    @test isa(DSGE.variable2index2value(x, id, Val(:b)), Number)
    @test DSGE.variable2index2value(x, id, Val(:b)) == x[id[:b]]

    DSGE.@variables2indices2values a, b = x, id
    @test a == x[id[:a]]
    @test b == x[id[:b]]

    @test DSGE.sslogdeviation2level(x, id, nt, Val(:a)) == exp((x[id[:a]]) + nt[:a])
    @test isa(DSGE.sslogdeviation2level(x, id, nt, Val(:b)), Number)
    @test DSGE.sslogdeviation2level(x, id, nt, Val(:b)) == exp((x[id[:b]]) + nt[:b])

    DSGE.@sslogdeviations2levels a, b = x, id, nt
    @test a == exp.((x[id[:a]]) + nt[:a])
    @test b == exp((x[id[:b][1]]) + nt[:b])

    @test DSGE.sslogdeviation2level_unprimekeys(x, id, nt, Val(:a′)) == exp((x[id[:a′]]) + nt[:a])
    @test isa(DSGE.sslogdeviation2level_unprimekeys(x, id, nt, Val(:b′)), Number)
    @test DSGE.sslogdeviation2level_unprimekeys(x, id, nt, Val(:b′)) == exp((x[id[:b′][1]]) + nt[:b])

    DSGE.@sslogdeviations2levels_unprimekeys a′, b′ = x, id, nt
    @test a′ == exp.((x[id[:a′]]) + nt[:a])
    @test b′ == exp((x[id[:b′][1]]) + nt[:b])

    @test DSGE.sslogdeviation2log(x, id, nt, Val(:a)) == ((x[id[:a]]) + nt[:a])
    @test isa(DSGE.sslogdeviation2log(x, id, nt, Val(:b)), Number)
    @test DSGE.sslogdeviation2log(x, id, nt, Val(:b)) == ((x[id[:b]]) + nt[:b])

    DSGE.@sslogdeviations2logs a, b = x, id, nt
    @test a == ((x[id[:a]]) + nt[:a])
    @test b == ((x[id[:b][1]]) + nt[:b])

    @test DSGE.ssdeviation2level(x, id, nt, Val(:a)) == ((x[id[:a]]) + nt[:a])
    @test isa(DSGE.ssdeviation2level(x, id, nt, Val(:b)), Number)
    @test DSGE.ssdeviation2level(x, id, nt, Val(:b)) == ((x[id[:b]]) + nt[:b])

    DSGE.@ssdeviations2levels a, b = x, id, nt
    @test a == ((x[id[:a]]) + nt[:a])
    @test b == ((x[id[:b][1]]) + nt[:b])

    @test DSGE.get_deviation(x, id, Val(:a)) == (x[id[:a]])
    @test isa(DSGE.get_deviation(x, id, Val(:b)), Number)
    @test DSGE.get_deviation(x, id, Val(:b)) == (x[id[:b]])

    DSGE.@get_deviations a, b = x, id
    @test a == (x[id[:a]])
    @test b == (x[id[:b]])
end

@testset "Macros for computing Jacobians of equilibrium conditions" begin
    d = Dict{Symbol, UnitRange{Int64}}(:a => 1:1, :b => 2:2)
    DSGE.@unpack_and_first a, b = d
    @test a == 1
    @test b == 2
end
