using LinearAlgebra

function _csc_build_transition(f_1::Float64, f_2::Float64, grid::Dict{String,Any}, param::Dict{String,Any})
    ns_1 = Int(grid["ns_1"])
    ns_2 = Int(grid["ns_2"])
    λ_1 = param["lambda_1"]
    λ_2 = param["lambda_2"]

    I_1 = Matrix(1.0I, ns_1, ns_1)
    I_2 = Matrix(1.0I, ns_2, ns_2)
    z_12 = zeros(ns_1, ns_2)
    z_21 = zeros(ns_2, ns_1)
    z_11 = zeros(ns_1, 1)
    z_21c = zeros(ns_2, 1)

    row_1 = hcat((1 - λ_1 + λ_1 * f_1) * I_1, z_12, λ_1 * (1 - f_1) * I_1, z_12, z_11)
    row_2 = hcat(z_21, (1 - λ_2 + λ_2 * f_2) * I_2, z_21, λ_2 * (1 - f_2) * I_2, z_21c)
    row_3 = hcat(f_1 * I_1, z_12, (1 - f_1) * I_1, z_12, z_11)
    row_4 = hcat(z_21, f_2 * I_2, z_21, (1 - f_2) * I_2, z_21c)
    row_5 = reshape(vcat(zeros(2 * (ns_1 + ns_2)), [1.0]), 1, :)

    return vcat(row_1, row_2, row_3, row_4, row_5)
end

function _csc_stationary_distribution(P::Matrix{Float64}, n::Int, iterations::Int)
    dist = fill(1.0 / n, n)
    for _ in 1:iterations
        dist = P' * dist
    end
    return vec(dist)
end

function _csc_compute_labor_blocks(f_1::Float64, f_2::Float64, param::Dict{String,Any}, grid::Dict{String,Any})
    ns = Int(grid["ns"])
    ns_1 = Int(grid["ns_1"])
    ns_2 = Int(grid["ns_2"])
    nse = Int(grid["nse"])

    P_SS3 = _csc_build_transition(f_1, f_2, grid, param)
    P_SE = param["P_SS2"] * P_SS3
    se_dist = _csc_stationary_distribution(P_SE, nse, 100_000)

    s = vec(grid["s"])
    se_bar_1 = dot(s[1:ns_1], se_dist[1:ns_1]) / sum(se_dist[1:ns_1])
    se_bar_2 = dot(s[(ns_1 + 1):ns], se_dist[(ns_1 + 1):ns]) / sum(se_dist[(ns_1 + 1):ns])

    N_tilde_1 = sum(se_dist[1:ns_1])
    N_tilde_2 = sum(se_dist[(ns_1 + 1):ns])
    U_tilde_1 = sum(se_dist[(ns + 1):(ns + ns_1)])
    U_tilde_2 = sum(se_dist[(ns + ns_1 + 1):(ns + ns)])

    P_SS4 = _csc_build_transition(0.0, 0.0, grid, param)
    P_SE2 = param["P_SS2"] * P_SS4
    se_dist_2 = vec(P_SE2' * se_dist)

    P_SE3 = P_SS3 * param["P_SS2"]
    se_dist_3 = _csc_stationary_distribution(P_SE3, 2 * ns + 1, 10_000)

    N_1 = sum(se_dist_2[1:ns_1])
    N_2 = sum(se_dist_2[(ns_1 + 1):ns])
    U_1 = sum(se_dist_2[(ns + 1):(ns + ns_1)])
    U_2 = sum(se_dist_2[(ns + ns_1 + 1):(ns + ns)])

    M_1 = N_tilde_1 - N_1
    M_2 = N_tilde_2 - N_2

    return (
        P_SS3=P_SS3,
        P_SE=P_SE,
        se_dist=se_dist,
        se_bar_1=se_bar_1,
        se_bar_2=se_bar_2,
        N_tilde_1=N_tilde_1,
        N_tilde_2=N_tilde_2,
        U_tilde_1=U_tilde_1,
        U_tilde_2=U_tilde_2,
        P_SS4=P_SS4,
        P_SE2=P_SE2,
        se_dist_2=se_dist_2,
        P_SE3=P_SE3,
        se_dist_3=se_dist_3,
        N_1=N_1,
        N_2=N_2,
        U_1=U_1,
        U_2=U_2,
        M_1=M_1,
        M_2=M_2,
    )
end

function find_fs(x::Vector{Float64}, param::Dict{String,Any}, grid::Dict{String,Any})
    f_1 = exp(x[1])
    f_2 = exp(x[2])

    stats = _csc_compute_labor_blocks(f_1, f_2, param, grid)

    u_1 = stats.U_tilde_1 / (stats.N_tilde_1 + stats.U_tilde_1)
    u_2 = stats.U_tilde_2 / (stats.N_tilde_2 + stats.U_tilde_2)

    return [param["u_1"] - u_1, param["u_2"] - u_2]
end
