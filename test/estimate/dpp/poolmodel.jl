path = dirname(@__FILE__)
using FileIO, HDF5

###########################################################################
# Set up for testing PoolModel instantiation
###########################################################################
m1 = AnSchorfheide()
pc1_file = poolspec!(m1)
m2 = SmetsWouters()
pc2_file = poolspec!(m2)

# save = normpath(joinpath(dirname(@__FILE__),"save"))

# Load true data here, name it data
# tbd . . .

# Load prediction data here
y1 = h5read(get_setting(m1, :dataroot) * "smc.h5")
y2 = h5read(get_setting(m1, :dataroot) * "sw_orig_smc.h5")

# Load particle clouds here
pc1_filepath = get_setting(m1, :saveroot) * "output_data/an_schorfheide/ss0/estimate/raw/"
pc2_filepath = get_setting(m2, :saveroot) * "output_data/an_schorfheide/ss1/estimate/raw/"
pc1 = load(pc1_filepath * pc1_file)
pc2 = load(pc2_filepath * pc2_file)

# Test outer constructors
h = 4
@testset "Check outer constructors" begin
    @test typeof(PoolModel(data = y1, h = h, [m1, m1])) == PoolModel
    @test typeof(PoolModel(datas = Dict(:AnSchorfheide => y1, :SmetsWouters => y2),
                             h = h, models = [m1, m2])) == PoolModel
    m = PoolModel(datas = Dict(:AnSchorfheide => y1, :SmetsWouters => y2),
                  h = h, models = [m1, m2])
    mstatic = PoolModel(datas = Dict(:AnSchorfheide => y1, :SmetsWouters => y2),
                        h = h, models = [m1, m2]; static = true)
    @test mstatic[:ρ] == 1 && mstatic[:ρ].fixed

    # Test access and update functions
    @test get_models(m) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
    @test get_models(m, :AnScorfheide) == OrderedDict(:AnScorfheide => m1)
    @test get_models(m, [:AnSchorfheide, :SmetsWouters]) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
    @test get_datas(m) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
    @test get_datas(m, :AnScorfheide) == OrderedDict(:AnScorfheide => m1)
    @test get_datas(m, [:AnSchorfheide, :SmetsWouters]) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
    @test get_particles(m) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
    @test get_particles(m, :AnScorfheide) == OrderedDict(:AnScorfheide => m1)
    @test get_particles(m, [:AnSchorfheide, :SmetsWouters]) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
    @test get_cond_loglhs(m) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
    @test get_cond_loglhs(m, :AnScorfheide) == OrderedDict(:AnScorfheide => m1)
    @test get_cond_loglhs(m, [:AnSchorfheide, :SmetsWouters]) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
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
    update_particles!(m, Dict(:AnSchorfheide => pc1))
    @test m == oldm
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
# recreate PoolModel object with SWFF and SWπ
pc1_file = poolspec!(m1)
pc2_file = poolspec!(m2)
pc1_filepath = get_setting(m1, :saveroot) * "output_data/an_schorfheide/ss0/estimate/raw/"
pc2_filepath = get_setting(m2, :saveroot) * "output_data/an_schorfheide/ss1/estimate/raw/"
pc1 = load(pc1_filepath * pc1_file)
pc2 = load(pc2_filepath * pc2_file)
y1 = h5read(get_setting(m1, :dataroot) * "smc.h5")
y2 = h5read(get_setting(m1, :dataroot) * "sw_orig_smc.h5")

# cond_loglhs_mat1 = ...
# cond_loglhs_mat2 = ...
@testset "Check prediction densities are correctly computed" begin
    @test @test_matrix_approx_eq get_cond_loglhs(m, :SWFF) ≈ cond_loglhs_mat1
    @test @test_matrix_appro_eq get_cond_loglhs(m, :SWπ) ≈ cond_loglhs_mat2
end

# Check ParticleClouds
@testset "Check ParticleClouds are correctly added" begin
    @test get_particles(m, :SWFF) == pc1
    @test get_particles(m, :SWπ) == pc2
end

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
