using DSGE
using ModelConstructors
using StateSpaceRoutines
using HDF5
using Random
using CSV
using Dates
using DataFrames
using Distributions
using Printf
using LinearAlgebra

path = dirname(@__FILE__)

#=
# Test `hessizero` in context of Rosenbrock function
function rosenbrock(x::Vector)
    a = 100
    b = 1
    return a*(x[2]-x[1]^2.0)^2.0 + b*(1-x[1])^2.0
end

function rosenbrock_hessian(x::Vector)
    H = zeros(eltype(x), 2, 2)
    H[1,1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2.0
    H[1,2] = -400.0 * x[1]
    H[2,1] = -400.0 * x[1]
    H[2,2] = 200.0
    return H
end
=#


function analytical_prior_hessian_diag(param::AbstractParameter{T}) where T<:AbstractFloat
    """
    Compute the second derivative of -log(prior) with respect to the parameter value.
    For the negative log prior, we want d²/dx²[-log(p(x))]
    """
    # Handle fixed parameters - return 0 for fixed parameters
    if param.fixed || param.prior == 0 || isnothing(param.prior)
        return 0.0
    end

    # Extract distribution from Nullable wrapper if needed
    prior_dist = try
        get(param.prior)
    catch
        param.prior
    end
    x = param.value

    if isa(prior_dist, Distributions.Normal)
        # Normal(μ, σ): -log p(x) = 0.5*log(2π) + log(σ) + 0.5*((x-μ)/σ)²
        # d²/dx²[-log p(x)] = 1/σ²
        σ = prior_dist.σ
        return 1.0 / (σ^2)

    elseif isa(prior_dist, Distributions.Gamma)
        # Gamma(α, θ) with scale θ: -log p(x) = -[(α-1)*log(x) - x/θ - α*log(θ) - log(Γ(α))]
        # d²/dx²[-log p(x)] = (α-1)/x²
        α = prior_dist.α
        return (α - 1.0) / (x^2)

    elseif isa(prior_dist, Distributions.Beta)
        # Beta(α, β): -log p(x) = -[(α-1)*log(x) + (β-1)*log(1-x) - log(B(α,β))]
        # d²/dx²[-log p(x)] = (α-1)/x² + (β-1)/(1-x)²
        α = prior_dist.α
        β = prior_dist.β
        return (α - 1.0) / (x^2) + (β - 1.0) / ((1.0 - x)^2)

    elseif isa(prior_dist, Distributions.Uniform)
        # Uniform: constant, second derivative is 0
        return 0.0

    elseif typeof(prior_dist).name.name == :RootInverseGamma
        # RootInverseGamma(ν, τ): from logpdf in distributions_ext.jl:
        # log p(x) = log(2) - log(Γ(ν/2)) + (ν/2)*log(ν*τ²/2) - ((ν+1)/2)*log(x²) - ν*τ²/(2x²)
        #          = log(2) - log(Γ(ν/2)) + (ν/2)*log(ν*τ²/2) - (ν+1)*log(x) - ν*τ²/(2x²)
        # -log p(x) = -log(2) + log(Γ(ν/2)) - (ν/2)*log(ν*τ²/2) + (ν+1)*log(x) + ν*τ²/(2x²)
        # d/dx[-log p(x)] = (ν+1)/x - ν*τ²/x³
        # d²/dx²[-log p(x)] = -(ν+1)/x² + 3*ν*τ²/x⁴
        ν = prior_dist.ν
        τ = prior_dist.τ
        return -(ν + 1.0) / (x^2) + 3.0 * ν * τ^2 / (x^4)

    else
        # Unknown distribution, use numerical approximation
        return NaN
    end
end

function hessian_prior(m::Union{AbstractDSGEModel,AbstractVARModel},
                  x::Vector{T}, data::AbstractArray; diag_order_error = diag_order_error, offdiag_order_error = offdiag_order_error, offdiag_method = offdiag_method, check_neg_diag::Bool = true,
                  toggle::Bool = true, verbose::Symbol = :none) where T<:AbstractFloat

    regime_switching = haskey(DSGE.get_settings(m), :regime_switching) &&
        get_setting(m, :regime_switching)

    DSGE.update!(m, x)

    # Index of free parameters
    para_free_inds = ModelConstructors.get_free_para_inds(DSGE.get_parameters(m);
                                                          regime_switching = regime_switching, toggle = toggle)

    # Compute hessian only for free parameters with indices less than max. Useful for
    # testing purposes.
    max_free_ind = n_hessian_test_params(m)
    if max_free_ind < maximum(para_free_inds)
        para_free_inds = para_free_inds[1:max_free_ind]
    end

    n_para = length(x)
    hessian  = zeros(n_para, n_para)

    # x_hessian is the vector of free params
    # x_model is the vector of all params
    x_model = copy(x)
    x_hessian = x_model[para_free_inds]

    function f_hessian(x_hessian)
        x_model[para_free_inds] = x_hessian
        DSGE.update!(m, x_model)
        return -DSGE.prior(m)
    end

    #TESTING purposes, change later
    #distr = use_parallel_workers(m)
    distr = false

    #get bounds
    n_free_params = length(para_free_inds)
    lb = zeros(n_free_params)
    ub = zeros(n_free_params)

    params = DSGE.get_parameters(m)
    for (i, idx) in enumerate(para_free_inds)
        lb[i] = params[idx].valuebounds[1]
        ub[i] = params[idx].valuebounds[2]
    end



    hessian_free, has_errors = hessizero(f_hessian, x_hessian, lb, ub; diag_order_error = diag_order_error, offdiag_order_error = offdiag_order_error, offdiag_method = offdiag_method,
        check_neg_diag = false, verbose = verbose, distr = distr)

    # Fill in rows/cols of zeros corresponding to location of fixed parameters
    # For each row corresponding to a free parameter, fill in columns corresponding to free
    # parameters. Everything else is 0.
    for (row_free, row_full) in enumerate(para_free_inds)
        hessian[row_full, para_free_inds] = hessian_free[row_free, :]
    end

    return hessian, has_errors
end


#m = DSSW()
m = Model1002("ss10")
values = Float64[]
for i = 1:length(m.parameters)
    push!(values, m.parameters[i].value)
end

# Compute analytical expected Hessian for priors
# Get free parameter indices
regime_switching = haskey(DSGE.get_settings(m), :regime_switching) &&
    get_setting(m, :regime_switching)
para_free_inds = ModelConstructors.get_free_para_inds(DSGE.get_parameters(m);
                                                      regime_switching = regime_switching, toggle = true)

# Compute hessian only for free parameters with indices less than max
max_free_ind = n_hessian_test_params(m)
if max_free_ind < maximum(para_free_inds)
    para_free_inds = para_free_inds[1:max_free_ind]
end

n_free_params = length(para_free_inds)
hessian_expected = zeros(length(m.parameters), length(m.parameters))

# Fill diagonal with analytical values for free parameters
params = DSGE.get_parameters(m)
for (i, idx) in enumerate(para_free_inds)
    analytical_diag = analytical_prior_hessian_diag(params[idx])
    if !isnan(analytical_diag)
        hessian_expected[idx, idx] = analytical_diag
    else
        println("Warning: Cannot compute analytical Hessian for parameter $(params[idx].key)")
    end
end
                


file = "$path/../reference/vecm2007_data_log.csv"
file2 = "$path/../reference/vecm2007_cointdata_log.csv"


# Read in matlab dataset
function construct_data()
     # Load in US dataset #3
     df = DataFrame(CSV.File(file))
     df_coint = DataFrame(CSV.File(file2))

     # Add dates
     dates_vec = []
     start_date = Date(1954, 9, 30)
     n_obs = size(df, 1)
     push!(dates_vec, start_date)
     for i = 2:n_obs
         push!(dates_vec, DSGE.iterate_quarters(start_date, i-1))
     end
     dates_col = DataFrame(dates = dates_vec)

     # Return df_to_matrix output
     df_mat = Matrix(df[1:end-1, :])
     df_coint = Matrix(df_coint)

     data = hcat(df_mat, df_coint)

     return data'
end

data = construct_data()


#hessian, _ = hessian_prior(m, values, data; toggle = true, verbose = :low)


diag_order_ls = [2,4]
offdiag_order_ls = [1,2,4]
o4_method = [:richardson, :sixteenstencil]

# Storage for results
results = []

for diag_order in diag_order_ls
    for offdiag_order in offdiag_order_ls
        for method in o4_method
            if offdiag_order != 4 && method == :sixteenstencil
                continue
            else
                start = time()

                hessian, _ = hessian_prior(m, values, data; diag_order_error = diag_order, offdiag_order_error = offdiag_order, offdiag_method = method, toggle = true, verbose = :low)
                endt = time()-start

                # Extract only free parameter submatrices for comparison
                hessian_free_only = hessian[para_free_inds, para_free_inds]
                hessian_expected_free_only = hessian_expected[para_free_inds, para_free_inds]

                inf_norm = norm(hessian_expected_free_only - hessian_free_only, Inf)
                l2_norm = norm(hessian_expected_free_only - hessian_free_only, 2)

                # Store results
                push!(results, (diag_order=diag_order, offdiag_order=offdiag_order,
                               method=method, inf_norm=inf_norm, l2_norm=l2_norm, seconds=endt))
            end
        end
    end
end

# Print all results at the end
println("\n" * "="^100)
println("SUMMARY OF RESULTS")
println("="^100)
println(rpad("Diag Order", 12) * rpad("Offdiag Order", 15) * rpad("Method", 20) *
        rpad("Inf Norm", 15) * rpad("L2 Norm", 15) * "Time (s)")
println("-"^100)

for result in results
    method_str = result.offdiag_order == 4 ? string(result.method) : "N/A"
    println(rpad("O(h^$(result.diag_order))", 12) *
            rpad("O(h^$(result.offdiag_order))", 15) *
            rpad(method_str, 20) *
            rpad(@sprintf("%.6e", result.inf_norm), 15) *
            rpad(@sprintf("%.6e", result.l2_norm), 15) *
            @sprintf("%.4f", result.seconds))
end
println("="^100)

#=
@testset "Check valid hessian at the minimum" begin
    @test hessian_expected ≈ hessian atol=0.01
    #@test_matrix_approx_eq hessian_expected hessian
end

# Not at the min (indeed, we are at max), ensure throws error
x1 = [1.0, 1.0]
rosenbrock_neg(x) = -rosenbrock(x)
@testset "Throw error for hessian calculation not at the min" begin
    @test_throws Exception DSGE.hessizero(rosenbrock_neg, x1; check_neg_diag=true)
end

# Not at the min, check matches closed form
x1 = Vector[[0.5, 1.5], [-1.0, -1.0]]
@testset "Check closed-form of the hessian" begin
    for x in x1
        local hessian_expected = rosenbrock_hessian(x)
        local hessian, = DSGE.hessizero(rosenbrock, x)
        @test @test_matrix_approx_eq hessian_expected hessian
    end
end
=#
