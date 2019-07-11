using DSGEModels, CSV

###########################################################################
# Set up for testing PoolModel instantiation
###########################################################################
# filepath = pwd()
filepath = dirname(@__FILE__)
saveroot = filepath * "/save/"
dataroot = filepath * "/save/input_data/"
vint = "990110"
iter = 1
prev = 980110
est = 2
m1 = Model805()
m2 = Model904()
for model in [m1, m2]
    model <= Setting(:sampling_method, :SMC)
    model <= Setting(:saveroot, saveroot)
    model <= Setting(:dataroot, dataroot)
    model <= Setting(:data_vintage, vint, true, "vint", "")
    model <= Setting(:prev, prev, true, "prev", "")
    model <= Setting(:est, est, true, "est", "")
end

# Load prediction data here
y1 = CSV.read(get_setting(m1, :dataroot) * "realtime_spec=m805_hp=true_vint=170410.csv")
y1 = Matrix{Float64}(Matrix(y1[y1.date .>= Date("1991-12-31"),:])[:,2:end]') # subset for desired data
y2 = CSV.read(get_setting(m2, :dataroot) * "realtime_spec=m904_hp=true_vint=170410.csv")
y2 = Matrix{Float64}(Matrix(y2[y2.date .>= Date("1991-12-31"),:])[:,2:end]') # subset for desired data

# Load loglhs here, second number is the data type, 1 -> no conditional on rate exp,
# 4 -> conditional on rate exp
# Based on the online appendix, it appears we should not condition on rate expectations
file_log1_1 = "m805_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
# file_log1_4 = "m805_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
file_log2_1 = "m904_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"
# file_log2_4 = "m904_preddens/logscores_T0=1991-12-31_T=2016-12-31_cond=semi_data=1_est=2_hor=4_samp=SMC.jld2"

loglhs1_1 = load(get_setting(m1, :dataroot) * file_log1_1)["logscores"]
# loglhs1_4 = load(get_setting(m1, :dataroot) * file_log1_4)["logscores"]
loglhs2_1 = load(get_setting(m2, :dataroot) * file_log2_1)["logscores"]
# loglhs2_4 = load(get_setting(m2, :dataroot) * file_log2_4)["logscores"]

# Process loglhs (compute means across time periods
loglhs1_1 = vec(mean(loglhs1_1, dims = 1))
# loglhs1_4 = vec(mean(loglhs1_4, dims = 1))
loglhs2_1 = vec(mean(loglhs2_1, dims = 1))
# loglhs2_4 = vec(mean(loglhs2_4, dims = 1))

# Test outer constructors
periods = 4
pm1 = PoolModel(Dict(:Model805 => y1, :Model904 => y2), periods,
                Dict(:Model805 => loglhs1_1, :Model904 => loglhs2_1), [m1, m2])
# pm4 = PoolModel(Dict(:Model805 => y1, :Model904 => y2), periods,
#                 Dict(:Model805 => loglhs1_4, :Model904 => loglhs2_4), [m1, m2])

@testset "Check outer constructors" begin
    @test typeof(PoolModel(y1, periods, Dict(:Model805 => loglhs1_1, :Model904 => loglhs2_1), [m1, m2]; testing = true)) == PoolModel{Float64}
    @test typeof(pm1) == PoolModel{Float64}
    mstatic = PoolModel(Dict(:Model805 => y1, :Model904 => y2), periods,
                Dict(:Model805 => loglhs1_1, :Model904 => loglhs2_1), [m1, m2]; static = true)
    @test mstatic[:ρ].value == 1 && mstatic[:ρ].fixed

    # Test access and update functions
    @test get_models(pm1) == OrderedDict(:Model805 => m1, :Model904 => m2)
    @test get_models(pm1, :Model805) == OrderedDict(:Model805 => m1)
    @test get_models(pm1, [:Model805, :Model904]) == OrderedDict(:Model805 => m1, :Model904 => m2)
    @test get_datas(pm1) == OrderedDict(:Model805 => y1, :Model904 => y2)
    @test get_datas(pm1, :Model805) == OrderedDict(:Model805 => y1)
    @test get_datas(pm1, [:Model805, :Model904]) == OrderedDict(:Model805 => y1, :Model904 => y2)
    @test get_periods(pm1) == 101
    @test get_forecast_horizon(pm1) == 4
    # @test get_particles(m) == OrderedDict(:Model805 => m1, :Model904 => m2)
    # @test get_particles(m, :AnScorfheide) == OrderedDict(:AnScorfheide => m1)
    # @test get_particles(m, [:Model805, :Model904]) == OrderedDict(:Model805 => m1, :Model904 => m2)
    @test typeof(get_cond_loglhs(pm1)) == OrderedDict{Symbol,Vector{Float64}}
    @test typeof(get_cond_loglhs(pm1, :Model805)) == Vector{Float64}
    @test length(get_cond_loglhs(pm1, [:Model805, :Model904])) == 2
    @test haskey(get_system(pm1), :statespace) && haskey(get_system(pm1), :distributions)
    @test haskey(get_statespace(pm1), :Φ) && haskey(get_statespace(pm1), :Ψ)
    @test get_statespace(pm1, :Φ) == get_statespace(pm1)[:Φ]
    @test get_statespace(pm1, :Ψ) == get_statespace(pm1)[:Ψ]
    @test haskey(get_distributions(pm1),:F_ϵ) && haskey(get_distributions(pm1), :F_u)
    @test haskey(get_distributions(pm1),:F_λ)
    @test get_distributions(pm1, :F_ϵ) == get_distributions(pm1)[:F_ϵ]
    @test get_distributions(pm1, :F_u) == get_distributions(pm1)[:F_u]
    @test get_distributions(pm1, :F_λ) == get_distributions(pm1)[:F_λ]
    @test get_Φ(pm1) == get_statespace(pm1, :Φ)
    @test get_Ψ(pm1) == get_statespace(pm1, :Ψ)
    @test get_F_ϵ(pm1) == get_distributions(pm1, :F_ϵ)
    @test get_F_u(pm1) == get_distributions(pm1, :F_u)
    @test get_F_λ(pm1) == get_distributions(pm1, :F_λ)
    oldm = deepcopy(pm1)
    update_models!(pm1, m1, m2; populate = false)
    @test keys(pm1.models) == keys(oldm.models)
    update_models!(pm1, [m1, m2]; populate = false)
    @test keys(pm1.models) == keys(oldm.models)
    update_datas!(pm1, Dict(:Model805 => y1))
    @test pm1.datas == oldm.datas
    # update_particles!(pm1, Dict(:Model805 => pc1))
    # @test pm1 == oldm
    update_cond_loglhs!(pm1, Dict(:Model805 => get_cond_loglhs(pm1, :Model805)))
    @test pm1.cond_loglhs == oldm.cond_loglhs
    # update_Φ!(pm1, get_Φ(pm1))
    # @test get_statespace(pm1, :Φ) == get_statespace(oldm, :Φ)
    # update_Ψ!(pm1, get_Ψ(pm1))
    # @test get_Ψ(pm1, :Ψ) == get_statespace(oldm, :Ψ)
    update_F_ϵ!(pm1, get_F_ϵ(pm1))
    @test get_distributions(pm1, :F_ϵ) == get_distributions(oldm, :F_ϵ)
    update_F_u!(pm1, get_F_u(pm1))
    @test get_distributions(pm1, :F_u) == get_distributions(oldm, :F_u)
    update_F_λ!(pm1, get_F_λ(pm1))
    @test get_distributions(pm1, :F_λ) == get_distributions(oldm, :F_λ)
end

# Check predictive densities are correct
# cond_loglhs_mat2 = ...
# @testset "Check prediction densities are correctly computed" begin
#     @test @test_matrix_approx_eq get_cond_loglhs(m, :SWFF) ≈ cond_loglhs_mat2
# end

# Check ParticleClouds
# @testset "Check ParticleClouds are correctly added" begin
#     @test get_particles(m, :SWFF) == pc1
#     @test get_particles(m, :SWπ) == pc2
# end

# Check solve and statespace functions apply to PoolModel
@testset "Check solve and statespace functions apply to PoolModel" begin
    Phi1, Psi1, F_eps1, F_uu1 = compute_system_function(pm1)
    @test typeof(solve(pm1)) == Nothing
    @test typeof(compute_system(pm1)) == Nothing
end

nothing
