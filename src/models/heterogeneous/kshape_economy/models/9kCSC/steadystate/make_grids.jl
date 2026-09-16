using LinearAlgebra
using SpecialFunctions: erf

include(joinpath(@__DIR__, "..", "..", "helpers", "set_field!.jl"))

_normal_cdf(x) = 0.5 * (1 + erf(x / sqrt(2.0)))

function _tauchen_lognormal(rho::Float64, sigma_e::Float64, n::Int, m::Float64=3.0)
    sigma_y = sigma_e / sqrt(1 - rho^2)
    zmax = m * sigma_y
    zgrid = collect(range(-zmax, zmax, length=n))
    step = zgrid[2] - zgrid[1]

    P = zeros(Float64, n, n)
    for j in 1:n
        P[j, 1] = _normal_cdf((zgrid[1] - rho * zgrid[j] + step / 2) / sigma_e)
        for k in 2:(n - 1)
            upper = (zgrid[k] - rho * zgrid[j] + step / 2) / sigma_e
            lower = (zgrid[k] - rho * zgrid[j] - step / 2) / sigma_e
            P[j, k] = _normal_cdf(upper) - _normal_cdf(lower)
        end
        P[j, n] = 1 - _normal_cdf((zgrid[n] - rho * zgrid[j] - step / 2) / sigma_e)
    end
    P ./= sum(P, dims=2)

    pi = fill(1.0 / n, n)
    for _ in 1:10_000
        pi = P' * pi
    end

    s_vals = exp.(zgrid)
    return s_vals ./ dot(pi, s_vals), P, pi
end

function make_grids(param::Dict{String,Any})
    grid = Dict{String, Any}()

    grid["nb"] = 40
    grid["na"] = 40
    grid["ns_1"] = 6
    grid["ns_2"] = 5
    grid["ns"] = grid["ns_1"] + grid["ns_2"]
    grid["nse"] = 2 * grid["ns"] + 1

    a_min = 0.0
    a_max = 1000.0
    grid["a"] = exp.(range(0.0, stop=log(a_max - a_min + 1), length=grid["na"])) .- 1 .+ a_min

    b_min = param["b_min"]
    b_max = 1000.0
    b_core = exp.(exp.(range(0.0, stop=log(log(b_max - b_min + 1) + 1), length=grid["nb"] - 1)) .- 1) .- 1 .+ b_min
    grid["b"] = sort(vcat(b_core, 0.0))

    sgrid, P_SS, _ = _tauchen_lognormal(param["rhoS"], param["sigmaS"], Int(grid["ns"]), 3.0)
    grid["s"] = collect(sgrid)
    grid["s_dist"] = fill(1.0 / grid["ns"], Int(grid["ns"]))
    for _ in 1:10_000
        grid["s_dist"] = P_SS' * grid["s_dist"]
    end

    set_field!(param, :P_SS, P_SS)

    grid["s_bar"] = dot(grid["s"], grid["s_dist"])
    grid["s2_bar"] = dot(grid["s"].^param["b_aux"], grid["s_dist"])

    benefit_states = ((1 + 1 / param["xi"]) / (1 / param["xi"] + param["tau_P"]) / param["GHH"]) .* min.(param["b_ratio"] .* grid["s"], grid["s_bar"])
    grid["se"] = vcat(grid["s"], benefit_states, 1.0)

    P_SS2 = kron(Matrix(1.0I, 2, 2), P_SS)
    P_SS2 = P_SS2 * (1 - param["in"])
    P_SS2 = hcat(P_SS2, fill(param["in"], 2 * Int(grid["ns"]), 1))
    lastrow = vcat(zeros(Int(grid["ns"])), fill(param["out"] / grid["ns"], Int(grid["ns"])), [1 - param["out"]])'
    P_SS2 = vcat(P_SS2, lastrow)

    grid["s2_dist"] = fill(1.0 / grid["nse"], Int(grid["nse"]))
    for _ in 1:10_000
        grid["s2_dist"] = P_SS2' * grid["s2_dist"]
    end

    set_field!(param, :P_SS2, P_SS2)
    set_field!(param, :Eshare, grid["s2_dist"][end])

    return grid
end
