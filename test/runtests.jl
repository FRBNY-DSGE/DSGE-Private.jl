using ModelConstructors, Nullables, SMC, Test, Distributed, Distributions
using Dates, DataFrames, OrderedCollections, FileIO, DataStructures, LinearAlgebra, SparseArrays
using StatsBase, Random, CSV, StateSpaceRoutines, HDF5, JLD2, MAT, Plots
import ModelConstructors: @test_matrix_approx_eq, @test_matrix_approx_eq_eps
@everywhere using DSGE, JLD2, Printf, LinearAlgebra, ModelConstructors, SMC
HETDSGEGOVDEBT = normpath(joinpath(@__DIR__, "..", "src", "models", "heterogeneous",
                                   "het_dsge_gov_debt", "reference"))

# Shared JLD2-proxy converter: reference MvNormals serialized under older PDMats/Distributions come
# back as JLD2-reconstructed proxies on the current stack; rebuild a real MvNormal from μ/Σ. Defined
# here (not per-test) so it's available no matter which test files run. Mirrors _jld2_to_df in
# test/data/transform_data.jl.
_jld2_to_mvnormal(x::Distribution) = x
_jld2_to_mvnormal(x) = MvNormal(collect(Float64, x.μ),
                                Matrix{Float64}(x.Σ isa AbstractMatrix ? x.Σ : x.Σ.mat))

# Keep FRED-dependent data-loader tests deterministic. The cached vintage is
# the same fixture the tests compare against, so PR jobs never need live FRED.
function install_fred_test_cache!(m)
    dataroot = mktempdir()
    rawroot = joinpath(dataroot, "raw")
    mkpath(rawroot)
    fixture_rawroot = joinpath(@__DIR__, "reference", "input_data", "raw")
    for name in readdir(fixture_rawroot)
        cp(joinpath(fixture_rawroot, name), joinpath(rawroot, name))
    end
    fixture_condroot = joinpath(@__DIR__, "reference", "input_data", "cond")
    condroot = joinpath(dataroot, "cond")
    mkpath(condroot)
    for name in readdir(fixture_condroot)
        cp(joinpath(fixture_condroot, name), joinpath(condroot, name))
    end
    cp(joinpath(@__DIR__, "reference", "fred_160812.csv"),
       joinpath(rawroot, "fred_160812.csv"))
    m <= Setting(:dataroot, dataroot)
    return m
end

# Authoritative automated-test manifest. CI and `Pkg.test()` both run this list.
my_tests = [
            "abstractdsgemodel",
            "abstractvarmodel",
            "altpolicy/ait",
            "altpolicy/alt_inflation",
            "altpolicy/altpolicy",
            "altpolicy/default_policy",
            "altpolicy/flexible_ait",
            "altpolicy/ngdp_target",
            "altpolicy/rw",
            "altpolicy/rw_zero_rate",
            "altpolicy/smooth_ait_gdp",
            "altpolicy/smooth_ait_gdp_alt",
            "altpolicy/taylor93",
            "altpolicy/taylor99",
            "altpolicy/taylor_rule",
            "altpolicy/zero_rate",
            "altpolicy/zlb_rule",
            "analysis/compute_meansbands",
            "analysis/create_q4q4_mb",
            "analysis/df_to_table",
            "analysis/io",
            "analysis/meansbands",
            "analysis/meansbands_to_matrix",
            "analysis/moments",
            "analysis/util",
            "data/fred_data",
            "data/load_data",
            "data/load_data_poolmodel",
            "data/manual_data_adjustments",
            "data/misc",
            "data/reverse_transform",
            "data/simulate_data",
            "data/transformations",
            "data/util",
            "decomp/decompose_forecast",
            "decomp/decomposition_periods",
            "decomp/io",
            "defaults",
            "estimate/backwards_compatibility",
            "estimate/cat",
            "estimate/combined_optimizer",
            "estimate/csminwel",
            "estimate/ct_filters/block_kalman_filter",
            "estimate/ct_filters/ct_block_kalman_filter",
            "estimate/ct_filters/ct_kalman_filter",
            "estimate/ct_filters/ct_kalman_simple",
            "estimate/estimate",
            "estimate/estimate_bma",
            "estimate/filter",
            "estimate/filter_hank",
            "estimate/filter_poolmodel",
            "estimate/hessian",
            "estimate/hessizero_prior",
            "estimate/kalman",
            "estimate/lbfgs",
            "estimate/marginal_data_density",
            "estimate/metropolis_hastings",
            "estimate/nearest_spd",
            "estimate/nelder_mead",
            "estimate/optimize",
            "estimate/optimize_dssw",
            "estimate/poolmodel_tpf",
            "estimate/posterior",
            "estimate/posterior_poolmodel",
            "estimate/regime_switching_mh",
            "estimate/resample",
            "estimate/simulated_annealing",
            "estimate/smc/helpers",
            "estimate/smc/initialization",
            "estimate/smc/mutation",
            "estimate/smc/particle",
            "estimate/smc/regime_switching_smc",
            "estimate/smc/resample",
            "estimate/smc/smc",
            "estimate/smc/util",
            "estimate/transform_transition_matrices",
            "estimate/util",
            "estimate/var/dsgevar_likelihood",
            "estimate/var/dsgevecm_likelihood",
            "forecast/automatic_tempalt_zlb",
            "forecast/drivers",
            "forecast/forecast",
            "forecast/forecast_one",
            "forecast/forecast_regime_switching",
            "forecast/impulse_responses",
            "forecast/io",
            "forecast/m1002_ss62_forecast_test",
            "forecast/multiple_altpol_imperfect_awareness",
            "forecast/shock_decompositions",
            "forecast/smooth",
            "forecast/time_varying_credibility",
            "forecast/util",
            "forecast/var/dsgevar/impulse_responses",
            "forecast/var/dsgevecm/impulse_responses",
            "forecast/var/impulse_responses",
            "forecast/wrappers_impulse_responses/dsgevar_lambda_impulse_responses",
            "forecast/wrappers_impulse_responses/observables_identified_dsge_impulse_responses",
            "forecast/wrappers_impulse_responses/var_approx_dsge_impulse_responses",
            "grids",
            "models/financial_frictions",
            "models/heterogeneous/bond_labor/bond_labor",
            "models/heterogeneous/het_dsge/het_dsge",
            "models/heterogeneous/het_dsge_gov_debt/het_dsge_gov_debt",
            "models/heterogeneous/het_dsge_gov_debt/het_dsge_gov_debt_reduce_ell",
            "models/heterogeneous/krusell_smith/krusell_smith",
            "models/heterogeneous/krusell_smith_ct/krusell_smith_ct",
            "models/heterogeneous/one_asset_hank/one_asset_hank",
            "models/heterogeneous/real_bond/real_bond",
            "models/heterogeneous/real_bond_mkup/real_bond_mkup",
            "models/poolmodel/poolmodel",
            "models/representative/an_schorfheide/an_schorfheide",
            "models/representative/m1002/m1002",
            "models/representative/m1010/m1010",
            "models/representative/m805/m805",
            "models/representative/m904/m904",
            "models/representative/m990/m990",
            "models/representative/rep_dsge_gov_debt/rep_dsge_gov_debt",
            "models/representative/smets_wouters/augment_states",
            "models/representative/smets_wouters/observables",
            "models/representative/smets_wouters_orig/smets_wouters_orig",
            "models/representative/smets_wouters/smets_wouters",
            "models/representative/smets_wouters/subspecs",
            "models/var/dsgevar/dsgevar",
            "models/var/dsgevar/measurement_error",
            "models/var/dsgevar/subspecs",
            "models/var/dsgevecm/dsgevecm",
            "models/var/dsgevecm/measurement_error",
            "models/var/dsgevecm/subspecs",
            "models/var/util",
            "packet/packet",
            "parameters",
            "plot/plot",
            "plot/util",
            "scenarios/drivers",
            "scenarios/forecast",
            "scenarios/scenario",
            "scenarios/switching",
            "workflows/model_filter_forecast",
            "solve/gensys",
            "solve/gensys2",
            "solve/gensys2_uncertain_altpol_test1",
            "solve/gensys2_uncertain_altpol_test2",
            "solve/gensys_uncertain_altpol",
            "solve/klein",
            "solve/solve",
            "solve/solve_poolmodel",
            "statespace/statespace",
            "util",
            ]

# These files are deliberately not run as standalone tests. Give each
# exclusion a reason so the inventory check cannot silently lose new tests.
const nonstandalone_test_files = Set([
    "ci_tester.jl",                          # compatibility wrapper; delegates to this runner
    "run_all_tests.jl",                      # legacy partial runner
    "runtests.jl",                           # this runner
    "jld2_compat.jl",                        # support code included by legacy JLD2 tests
    "forecast/tvcred_parameterize.jl",        # test support included by time-varying credibility tests
    "estimate/optimize_ss10.jl",              # loads a personal includeall.jl outside the checkout
    "estimate/optimize_ss10_pso.jl",           # loads a personal includeall.jl outside the checkout
    "estimate/optimize2.jl",                  # legacy script depends on globals from another script and has no assertions
    "estimate/optimize_dssw_pso.jl",          # requests unsupported :pso_meta and has no assertions
    "estimate/hessizero_large.jl",            # diagnostic/stress script has no assertions
    "estimate/hessizero_rosenbrock.jl",        # assertions are commented out; diagnostic script only
    "estimate/smc/online.jl",                 # expects generated 10,000-particle clouds absent from a clean checkout
    "estimate/smc/Brookings_test/regime_switching_brookings.jl", # expects Brookings-specific saved estimation output
    "estimate/smc/Brookings_test/util_brookings.jl", # helper included by the Brookings test
    "models/heterogeneous/one_asset_hank/interns.jl", # exploratory script with assignment instead of equality assertions
    "solve/gensys_ct.jl",                     # requires generated test_outputs/ files absent from the checkout
    "solve/reduction.jl",                     # requires generated test_outputs/ files absent from the checkout
    "solve/solve_ct.jl",                      # requires generated test_outputs/ files absent from the checkout
])

# Make test discovery auditable: every Julia file under test/ must either be
# listed above or have an explicit nonstandalone reason.
const listed_test_files = Set(string(test, ".jl") for test in my_tests)
const discovered_test_files = Set(replace(relpath(joinpath(root, file), @__DIR__), '\\' => '/')
    for (root, _, files) in walkdir(@__DIR__) for file in files if endswith(file, ".jl"))
const unaccounted_test_files = setdiff(discovered_test_files, union(listed_test_files, nonstandalone_test_files))
const stale_test_entries = setdiff(listed_test_files, discovered_test_files)
isempty(unaccounted_test_files) || error("Unaccounted test files (add to my_tests or document in nonstandalone_test_files): $(sort(collect(unaccounted_test_files)))")
isempty(stale_test_entries) || error("Test manifest entries do not exist: $(sort(collect(stale_test_entries)))")

const test_filter = get(ENV, "DSGE_TEST_FILTER", "")
const tests_to_run = isempty(test_filter) ? my_tests : Base.filter(test -> startswith(test, test_filter), my_tests)
isempty(tests_to_run) && error("DSGE_TEST_FILTER='$(test_filter)' matched no test manifest entries")

failures = Tuple{String, Any}[]
for test in tests_to_run
    test_file = string("$test.jl")
    @printf " * %s\n" test_file
    try
        include(test_file)
    catch err
        push!(failures, (test_file, err))
        @error "Test file failed" test_file exception = (err, catch_backtrace())
    end
end

println("\n", "="^70)
@printf "TEST SUMMARY: %d of %d test files passed\n" (length(tests_to_run) - length(failures)) length(tests_to_run)
if !isempty(failures)
    println("Failed test files:")
    for (test_file, _) in failures
        println("  ✗ ", test_file)
    end
    error("$(length(failures)) test file(s) failed")
else
    println("All test files passed.")
end
