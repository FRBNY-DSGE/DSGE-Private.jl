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
@assert typeof(PoolModel(data = y1, h = h, [m1, m1])) == PoolModel
@assert typeof(PoolModel(datas = Dict(:AnSchorfheide => y1, :SmetsWouters => y2),
                         h = h, models = [m1, m2])) == PoolModel
m = PoolModel(datas = Dict(:AnSchorfheide => y1, :SmetsWouters => y2),
                         h = h, models = [m1, m2])

# Test access and update functions
@assert get_models(m) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
@assert get_models(m, :AnScorfheide) == OrderedDict(:AnScorfheide => m1)
@assert get_models(m, [:AnSchorfheide, :SmetsWouters]) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
@assert get_datas(m) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
@assert get_datas(m, :AnScorfheide) == OrderedDict(:AnScorfheide => m1)
@assert get_datas(m, [:AnSchorfheide, :SmetsWouters]) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
@assert get_particles(m) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
@assert get_particles(m, :AnScorfheide) == OrderedDict(:AnScorfheide => m1)
@assert get_particles(m, [:AnSchorfheide, :SmetsWouters]) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
@assert get_cond_loglhs(m) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
@assert get_cond_loglhs(m, :AnScorfheide) == OrderedDict(:AnScorfheide => m1)
@assert get_cond_loglhs(m, [:AnSchorfheide, :SmetsWouters]) == OrderedDict(:AnSchorfheide => m1, :SmetsWouters => m2)
@assert haskey(get_system(m), :statespace) && haskey(get_system(m), :distributions)
@assert haskey(get_statespace(m), :Φ) && haskey(get_statespace(m), :Ψ)
@assert typeof(get_statespace(m; :Φ)) == Function
@assert typeof(get_statespace(m; :Ψ)) == Function
@assert haskey(get_distributions(m), :F_ϵ)) && haskey(get_distributions(m), :F_u)
@assert typeof(get_distributions(m; :F_ϵ)) == Distribution
@assert typeof(get_distributions(m; :F_u)) == Distribution
@assert get_Φ(m) == get_statespace(m, :Φ)
@assert get_Ψ(m) == get_statespace(m, :Ψ)
@assert get_F_ϵ(m) == get_distributions(m, :F_ϵ)
@assert get_F_u(m) == get_distributions(m, :F_u)
oldm = deepcopy(m)
update_models!(m, m1, m2; populate = false)
@assert m == oldm
update_models!(m, [m1, m2]; populate = false)
@assert m == oldm
update_datas!(m, Dict(:AnSchorfheide => y1))
@assert m == oldm
update_particles!(m, Dict(:AnSchorfheide => pc1))
@assert m == oldm
update_cond_loglhs!(m, Dict(:AnSchorfheide => get_loglhs(m, :AnSchorfheide)))
@assert m == oldm
update_Φ!(m, get_Φ(m))
@assert m == oldm
update_Ψ!(m, get_Ψ(m))
@assert m == oldm
update_F_ϵ!(m, get_F_ϵ(m))
@assert m == oldm
update_F_u!(m, get_F_u(m))
@assert m == oldm

# Check predictive densities are correct
# recreate PoolModel object with SWFF and SWπ
# cond_loglhs_mat1 = ...
# cond_loglhs_mat2 = ...
@assert get_cond_loglhs(m, :SWFF) ≈ cond_loglhs_mat1
@assert get_cond_loglhs(m, :SWΠ) ≈ cond_loglhs_mat2
