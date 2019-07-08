path = dirname(@__FILE__)
using DSGE, DSGEModels, CSV, FileIO, HDF5, Random, Dates, Statistics

###########################################################################
# Set up for testing PoolModel instantiation
###########################################################################
include(String(path) * "/spec/poolspec.jl")
Random.seed!(42)
m1 = Model805()
poolspec!(m1)
m2 = Model904()
poolspec!(m2)

# save = normpath(joinpath(dirname(@__FILE__),"save"))

# Load true data here
# tbd . . .

# Load prediction data here
# y1 = load(replace(get_setting(m1, :dataroot) * "smc.h5", ".h5" => "jld2")
y1 = CSV.read(get_setting(m1, :dataroot) * "realtime_spec=m805_hp=true_vint=170410.csv")
y1 = Matrix{Float64}(Matrix(y1[y1.date .>= Date("1991-12-31"),:])[:,2:end]') # subset for desired data
y2 = CSV.read(get_setting(m2, :dataroot) * "realtime_spec=m904_hp=true_vint=170410.csv")
y2 = Matrix{Float64}(Matrix(y2[y2.date .>= Date("1991-12-31"),:])[:,2:end]') # subset for desired data

# Load loglhs here, second number is the data type, 1 -> no conditional on rate exp,
# 4 -> conditional on rate exp
file_log1_1 = "m805_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
file_log1_4 = "m805_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
file_log2_1 = "m904_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
file_log2_4 = "m904_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"

loglhs1_1 = load(get_setting(m1, :dataroot) * file_log1_1)["logscores"]
loglhs1_4 = load(get_setting(m1, :dataroot) * file_log1_4)["logscores"]
loglhs2_1 = load(get_setting(m2, :dataroot) * file_log2_1)["logscores"]
loglhs2_4 = load(get_setting(m2, :dataroot) * file_log2_4)["logscores"]

# Process loglhs (compute means across time periods
loglhs1_1 = vec(mean(loglhs1_1, dims = 1))
loglhs1_4 = vec(mean(loglhs1_4, dims = 1))
loglhs2_1 = vec(mean(loglhs2_1, dims = 1))
loglhs2_4 = vec(mean(loglhs2_4, dims = 1))

# Test outer constructors
periods = 4
pm1 = PoolModel(Dict(:Model805 => y1, :Model904 => y2), periods,
                Dict(:Model805 => loglhs1_1, :Model904 => loglhs2_1), [m1, m2])
pm4 = PoolModel(Dict(:Model805 => y1, :Model904 => y2), periods,
                Dict(:Model805 => loglhs1_4, :Model904 => loglhs2_4), [m1, m2])

@assert false
@testset "Check outer constructors" begin
    @test typeof(PoolModel(data = y1, h = h, [m1, m1]; testing = true)) == PoolModel
    @test typeof(PoolModel(datas = Dict(:AnSchorfheide => y1, :SmetsWouters => y2),
                             h = h, models = [m1, m2]; testing = true)) == PoolModel
    m = PoolModel(datas = Dict(:AnSchorfheide => y1, :SmetsWouters => y2),
                  h = h, models = [m1, m2]; testing = true)
    mstatic = PoolModel(datas = Dict(:AnSchorfheide => y1, :SmetsWouters => y2),
                        h = h, models = [m1, m2]; static = true, testing = true)
    @test mstatic[:ρ] == 1 && mstatic[:ρ].fixed

    # Test access and update functions
    @test get_models(m) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
    @test get_models(m, :AnScorfheide) == OrderedDict(:AnScorfheide => m1)
    @test get_models(m, [:AnSchorfheide, :SmetsWouters]) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
    @test get_datas(m) == OrderedDict(:AnSchorfheide => y1, :SmetsWouters => y2)
    @test get_datas(m, :AnScorfheide) == OrderedDict(:AnScorfheide => y1)
    @test get_datas(m, [:AnSchorfheide, :SmetsWouters]) == OrderedDict(:AnSchorfheide => y1, :SmetsWouters => y2)
    # @test get_particles(m) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
    # @test get_particles(m, :AnScorfheide) == OrderedDict(:AnScorfheide => m1)
    # @test get_particles(m, [:AnSchorfheide, :SmetsWouters]) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
    @test typeof(get_cond_loglhs(m)) == OrderedDict
    @test length(get_cond_loglhs(m, :AnScorfheide)) == 1
    @test length(get_cond_loglhs(m, [:AnSchorfheide, :SmetsWouters])) == 2
    @test haskey(get_system(m), :statespace) && haskey(get_system(m), :distributions)
    @test haskey(get_statespace(m), :Φ) && haskey(get_statespace(m), :Ψ)
    @test typeof(get_statespace(m; :Φ)) == Function
    @test typeof(get_statespace(m; :Ψ)) == Function
    @test haskey(get_distributions(m), :F_ϵ)) && haskey(get_distributions(m), :F_u)
    @test typeof(get_distributions(m; :F_ϵ)) == Distribution
    @test typeof(get_distributions(m; :F_u)) == Distribution
    @test get_Φ(m) == get_statespace(m, :Φ)
    @test get_Ψ(m) == get_statespace(m, :Ψ)
    @test get_F_ϵ(m) == get_distributions(m, :F_ϵ)
    @test get_F_u(m) == get_distributions(m, :F_u)
    oldm = deepcopy(m)
    update_models!(m, m1, m2; populate = false)
    @test m == oldm
    update_models!(m, [m1, m2]; populate = false)
    @test m == oldm
    update_datas!(m, Dict(:AnSchorfheide => y1))
    @test m == oldm
    # update_particles!(m, Dict(:AnSchorfheide => pc1))
    # @test m == oldm
    update_cond_loglhs!(m, Dict(:AnSchorfheide => get_loglhs(m, :AnSchorfheide)))
    @test m == oldm
    update_Φ!(m, get_Φ(m))
    @test m == oldm
    update_Ψ!(m, get_Ψ(m))
    @test m == oldm
    update_F_ϵ!(m, get_F_ϵ(m))
    @test m == oldm
    update_F_u!(m, get_F_u(m))
    @test m == oldm
end


# Check predictive densities are correct
# recreate PoolModel object with AnSchorfheide and SWFF?
poolspec!(m2)
# pc1_filepath = get_setting(m1, :saveroot) * "output_data/an_schorfheide/ss0/estimate/raw/"
# pc2_filepath = get_setting(m2, :saveroot) * "output_data/an_schorfheide/ss1/estimate/raw/"
# pc1 = load(pc1_filepath * pc1_file)
# pc2 = load(pc2_filepath * pc2_file)
y2 = h5read(get_setting(m2, :dataroot) * "sw_orig_smc.h5", "data")

# cond_loglhs_mat2 = ...
@testset "Check prediction densities are correctly computed" begin
    @test @test_matrix_approx_eq get_cond_loglhs(m, :SWFF) ≈ cond_loglhs_mat2
end

# Check ParticleClouds
# @testset "Check ParticleClouds are correctly added" begin
#     @test get_particles(m, :SWFF) == pc1
#     @test get_particles(m, :SWπ) == pc2
# end

# Check solve and statespace functions apply to PoolModel
Phi, Psi, F_eps, F_uu = compute_system_function(m)
@testset "Check solve and statespace functions apply to PoolModel" begin
    @test typeof(solve(m)) == Nothing
    @test typeof(compute_system(m)) == Nothing
    @test typeof(Phi) == Function
    @test typeof(Psi) == Function
    @test typeof(F_eps) == Distribution
    @test typeof(F_uu) == Distribution
end

nothing
