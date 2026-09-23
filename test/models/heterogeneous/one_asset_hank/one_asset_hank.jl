using DSGE, JLD2
import Test: @test, @testset
import DSGE: @test_matrix_approx_eq
import DataStructures: OrderedDict

### Model
m = OneAssetHANK()
endo = m.endogenous_states
### Model indices
@testset "Model indices" begin
    # Endogenous states
    @test n_states(m) == length(endo[:value_function]) + length(endo[:inflation]) + length(endo[:distribution]) + length(endo[:monetary_policy]) + 5
    @test get_setting(m, :n_jump_vars) == length(endo[:value_function]) + length(endo[:inflation])
    @test get_setting(m, :n_state_vars) == length(endo[:distribution]) - get_setting(m, :n_state_vars_unreduce) + length(endo[:monetary_policy])

    # Exogenous shocks
    @test n_shocks_exogenous(m) == 1

    # Expectation shocks
    @test n_shocks_expectational(m) == get_setting(m, :n_jump_vars)
end

### Steady state
#=
# Load test values. Only using params for now, hard-coded other values,
# but we list the other sets of parameters here if user wants to
# edit this test file.
params = load("../../../test_outputs/one_asset_hank/saved_params.jld2", "params")
# grids = load("../../../test_outputs/one_asset_hank/saved_params.jld2", "grids")
# init_params = load("../../../test_outputs/one_asset_hank/saved_params.jld2", "init_params")
# approx_params = load("../../../test_outputs/one_asset_hank/saved_params.jld2", "approx_params")
# red_params = load("../../../test_outputs/one_asset_hank/saved_params.jld2", "red_params")
vars_ss_mat = load("../../../test_outputs/one_asset_hank/vars_ss.jld2")

# Update parameters
m[:coefrra] = params[:coefrra]
m[:frisch] = params[:frisch]
m[:meanlabeff] = params[:meanlabeff]
m[:maxhours] = params[:maxhours]
m[:ceselast] = params[:ceselast]
m[:labtax] = params[:labtax]
m[:govbondtarget] = params[:govbondtarget]
m[:labdisutil] = params[:labdisutil]
m[:lumptransferpc] = params[:lumptransferpc]
m[:priceadjust] = params[:priceadjust]
m[:taylor_inflation] = params[:taylor_inflation]
m[:taylor_outputgap] = params[:taylor_outputgap]
m[:govbcrule_fixnomB] = params[:govbcrule_fixnomB]
m[:σ_MP] = params[:ssigma_MP]
m[:θ_MP] = params[:ttheta_MP]
=#
# Update settings
for setting in (
    Setting(:r0, .005), Setting(:rmax, .08), Setting(:rmin, .001),
    Setting(:ρ0, .02), Setting(:ρmax, .05), Setting(:ρmin, .005),
    Setting(:iterate_r, false), Setting(:iterate_ρ, true),
    Setting(:I, 100), Setting(:amin, 0.), Setting(:amax, 40.),
    Setting(:agridparam, 1),
)
    m <= setting
end
#update!(m.settings[:a], Setting(:a, construct_asset_grid(get_setting(m, :I), get_setting(m, :agridparam), get_setting(m, :amin), get_setting(m, :amax))))
m <= Setting(:J, 2)
m <= Setting(:ygrid_combined, [0.2, 1.])
m <= Setting(:ymarkov_combined, [-.5 .5; .0376 -.0376])
#update!(m.settings[:g_z] , Setting(:g_z, compute_stationary_income_distribution(get_setting(m, :ymarkov_combined), get_setting(m, :J))))
#update!(m.settings[:zz], Setting(:zz, construct_labor_income_grid(get_setting(m, :ygrid_combined), get_setting(m, :g_z), m[:meanlabeff].value, get_setting(m, :I))))
m <= Setting(:z, get_setting(m, :zz)[1, :])
m <= Setting(:n_jump_vars, 201)
m <= Setting(:n_state_vars, 200)
m <= Setting(:n_state_vars_unreduce, 0)
m <= Setting(:maxit_HJB, 500)
m <= Setting(:tol_HJB, 1e-8)
m <= Setting(:Δ_HJB, 1e6)
m <= Setting(:Ir, 100)
m <= Setting(:crit_S, 1e-5)
m <= Setting(:maxit_kfe, 1000)
m <= Setting(:tol_kfe, 1e-12)
m <= Setting(:Δ_kfe, 1e6)
m <= Setting(:niter_hours, 10)
m <= Setting(:reduce_state_vars, true)
m <= Setting(:reduce_v, true)
m <= Setting(:krylov_dim, 20)
m <= Setting(:n_knots, 12)
m <= Setting(:c_power, 1)
knots = collect(range(get_setting(m, :amin), stop=get_setting(m, :amax), length=(get_setting(m, :n_knots) - 1)))
knots = (get_setting(m, :amax) .- get_setting(m, :amin))/(2^get_setting(m, :c_power) - 1) * ((knots .- get_setting(m, :amin)) / (get_setting(m, :amax) .- get_setting(m, :amin)) .+ 1).^get_setting(m, :c_power) .+ get_setting(m, :amin) .- (get_setting(m, :amax) - get_setting(m, :amin))/(2^get_setting(m, :c_power) - 1)
m <= Setting(:knots_dict, Dict(1 => knots))
m <= Setting(:spline_grid, get_setting(m, :a))
m <= Setting(:n_prior, 1)
m <= Setting(:n_post, 2)

# compare steady state values
steadystate!(m)
#=
@testset "Steady State" begin
    @test @test_matrix_approx_eq vec(m[:V_ss].value) vec(vars_ss_mat["V_SS"])
    @test @test_matrix_approx_eq vec(m[:g_ss].value) vec(vars_ss_mat["g_SS"])
    @test @test_matrix_approx_eq vec(m[:u_ss].value) vec(vars_ss_mat["u_SS"])
    @test @test_matrix_approx_eq vec(m[:c_ss].value) vec(vars_ss_mat["c_SS"])
    @test @test_matrix_approx_eq vec(m[:h_ss].value) vec(vars_ss_mat["h_SS"])
    @test @test_matrix_approx_eq vec(m[:s_ss].value) vec(vars_ss_mat["s_SS"])
    @test m[:B_ss].value ≈ vars_ss_mat["B_SS"] # approx to allow machine error
    @test m[:N_ss].value ≈ vars_ss_mat["N_SS"]
    @test m[:Y_ss].value ≈ vars_ss_mat["Y_SS"]
    @test m[:G_ss].value ≈ vars_ss_mat["G_SS"]
    @test m[:T_ss].value ≈ vars_ss_mat["T_SS"]
end
=#
### Equilibrium conditions
Γ0, Γ1, Ψ, Π, C = eqcond(m)

# Transition and measurement equations
TTT, RRR, CCC = solve(m)
# measurement here

# Matrices have expected dimensions
@testset "Canonical and State Transition Matrices" begin
    n_vars = n_states(m)
    @test size(Γ0) == (n_vars, n_vars)
    @test size(Γ1) == (n_vars, n_vars)
    @test size(C) == (n_vars,)
    @test size(Ψ) == (n_vars, 1)
    @test size(Π) == (n_vars, n_shocks_expectational(m))
    if get_setting(m, :reduce_v)
        @test size(TTT) == (get_setting(m, :n_state_vars_red) + get_setting(m, :n_splined), get_setting(m, :n_state_vars_red) + get_setting(m, :n_splined))
        @test size(RRR) == (get_setting(m, :n_state_vars_red) + get_setting(m, :n_splined), 1)
        @test size(vec(CCC)) == (get_setting(m, :n_state_vars_red) + get_setting(m, :n_splined),)
    elseif get_setting(m, :reduce_state_vars)
        @test size(TTT) == (get_setting(m, :n_jump_vars) + get_setting(m, :n_state_vars_red), get_setting(m, :n_state_vars_red))
        @test size(RRR) == (get_setting(m, :n_jump_vars) + get_setting(m, :n_state_vars_red), 1)
        @test size(vec(CCC)) == (get_setting(m, :n_jump_vars) + get_setting(m, :n_state_vars_red),)
    else
        @test size(TTT) == (n_vars, n_vars)
        @test size(RRR) == (n_vars, 1)
        @test size(vec(CCC)) == (n_vars,)
    end
end
