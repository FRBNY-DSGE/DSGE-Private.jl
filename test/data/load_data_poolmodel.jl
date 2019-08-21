using DSGE
using Test, DataFrames, HDF5, JLD2

path = dirname(@__FILE__)

@testset "Test various forms of data loading" begin
    m = PoolModel("ss1")
    fp = dirname(@__FILE__)
    m <= Setting(:dataroot, "$(fp)/../reference/")
    observables = OrderedDict{Symbol,Observable}()
    observables[:Model904] = Observable(:Model904, [:wrongorigmatlab, :p904],
                                        x -> x, x -> x,
                                        "Model 904 predictive density",
                                        "Model 904 conditional predictive density scores")
    observables[:Model805] = Observable(:Model805, [:wrongorigmatlab, :p805],
                                        x -> x, x -> x,
                                        "Model 805 predictive density",
                                        "Model 805 conditional predictive density scores")
    m.observable_mappings = observables
    df1 = load_data(m)
    df2 = CSV.read(joinpath(dataroot(m), "raw/wrongorigmatlab.csv"))
    df2[:date] = Vector{Dates.Date}(df2[:date])
    df2[:p904] = Vector{Float64}(df2[:p904])
    df2[:p805] = Vector{Float64}(df2[:p805])

    @test df1[:Model904] == df2[:p904]
    @test df1[:Model805] == df2[:p805]
end

nothing
