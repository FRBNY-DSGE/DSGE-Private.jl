function q_cons(c_a_guess::Array{Float64,3}, c_n_guess::Array{Float64,3},
               w_bar_1::Float64, w_bar_2::Float64, n_1::Float64, n_2::Float64,
               tau_L::Float64, tau_P::Float64, AProb::Array{Float64,3}, mu_dist_tilde::Array{Float64,3},
               meshes::Dict{String,Any}, param::Dict{String,Any}, grid::Dict{String,Any})

    ns_1 = Int(grid["ns_1"])
    ns   = Int(grid["ns"])
    xi   = param["xi"]

    X_agg = sum(vec(mu_dist_tilde) .* vec(AProb) .* vec(c_a_guess) .+
                vec(mu_dist_tilde) .* (1 .- vec(AProb)) .* vec(c_n_guess))

    NW_aux = w_bar_2 * n_2 .* meshes["se"]
    NW_aux[:, :, 1:ns_1] .= w_bar_1 * n_1 .* meshes["se"][:, :, 1:ns_1]

    if param["GHH"] == 1
        C_agg = X_agg + sum((1 - tau_L) * (1 - tau_P) / (1 + 1 / xi) .*
                            NW_aux[:, :, 1:ns].^(1 - tau_P) .* mu_dist_tilde[:, :, 1:ns])
    else
        C_agg = X_agg
    end

    return C_agg
end