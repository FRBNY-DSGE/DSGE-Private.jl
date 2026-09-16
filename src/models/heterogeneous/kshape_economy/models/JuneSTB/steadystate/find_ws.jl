using LinearAlgebra

function _csc_job_value_stats(r_l_1::Float64, r_l_2::Float64, w_bar_1::Float64, w_bar_2::Float64,
                              fix_L_1::Float64, fix_L_2::Float64,
                              param::Dict{String,Any}, grid::Dict{String,Any})
    ns = Int(grid["ns"])
    ns_1 = Int(grid["ns_1"])
    ns_2 = Int(grid["ns_2"])

    s = vec(grid["s"])
    se_dist = vec(grid["se_dist"])

    I_ns = Matrix(1.0I, ns, ns)
    λ_aux = [
        (1 - param["lambda_1"]) * Matrix(1.0I, ns_1, ns_1) zeros(ns_1, ns_2);
        zeros(ns_2, ns_1) (1 - param["lambda_2"]) * Matrix(1.0I, ns_2, ns_2)
    ]

    J_inst_aux = vcat(
        (r_l_1 - fix_L_1 - w_bar_1) * param["n_1"] .* s[1:ns_1],
        (r_l_2 - fix_L_2 - w_bar_2) * param["n_2"] .* s[(ns_1 + 1):ns],
    )

    A = I_ns - param["beta"] * (1 - param["death_rate"]) * (1 - param["in"]) * param["P_SS"] * λ_aux
    J_aux = A \ J_inst_aux

    J_bar_1 = dot(J_aux[1:ns_1], se_dist[1:ns_1]) / sum(se_dist[1:ns_1])
    J_bar_2 = dot(J_aux[(ns_1 + 1):ns], se_dist[(ns_1 + 1):ns]) / sum(se_dist[(ns_1 + 1):ns])

    return J_aux, J_bar_1, J_bar_2
end

function find_ws(x::Vector{Float64}, r_l_1::Float64, r_l_2::Float64,
                 V_1::Float64, V_2::Float64, M_1::Float64, M_2::Float64,
                 param::Dict{String,Any}, grid::Dict{String,Any})
    w_bar_1 = x[1]
    fix_L_2 = x[2]
    fix_L_1 = 0.0
    w_bar_2 = w_bar_1 * param["earning_ratio"] / (grid["se_bar_2"] / grid["se_bar_1"])

    _, J_bar_1, J_bar_2 = _csc_job_value_stats(r_l_1, r_l_2, w_bar_1, w_bar_2, fix_L_1, fix_L_2, param, grid)

    return [
        param["iota_1"] - M_1 / V_1 * J_bar_1,
        param["iota_2"] - M_2 / V_2 * J_bar_2,
    ]
end
