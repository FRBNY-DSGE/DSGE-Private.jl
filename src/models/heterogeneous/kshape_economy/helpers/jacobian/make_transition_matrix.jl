# TO-DO: add types

function make_transition_matrix(Dist_b, idb, Dist_a, ida, P_transition_dist, nb, na, nse)
    N = nb * na * nse
    Nab = nb * na

    # repeat ids across future exogenous states
    idb_r = repeat(vec(idb), 1, nse)
    ida_r = repeat(vec(ida), 1, nse)

    idse = repeat(1:nse, inner=Nab)

    # destination columns
    index11 = sub2ind3(nb, na, vec(idb_r),     vec(ida_r),     idse)
    index12 = sub2ind3(nb, na, vec(idb_r),     vec(ida_r) .+ 1, idse)
    index21 = sub2ind3(nb, na, vec(idb_r) .+ 1, vec(ida_r),     idse)
    index22 = sub2ind3(nb, na, vec(idb_r) .+ 1, vec(ida_r) .+ 1, idse)

    weight11 = Array{Float64}(undef, Nab, nse, nse)
    weight12 = Array{Float64}(undef, Nab, nse, nse)
    weight21 = Array{Float64}(undef, Nab, nse, nse)
    weight22 = Array{Float64}(undef, Nab, nse, nse)

    for hh in 1:nse
        db = Dist_b[:, :, hh]
        da = Dist_a[:, :, hh]
        Ph = transpose(P_transition_dist[hh, :])

        w11 = vec((1 .- db) .* (1 .- da))
        w12 = vec((1 .- db) .* da)
        w21 = vec(db .* (1 .- da))
        w22 = vec(db .* da)

        weight11[:, :, hh] = w11 * Ph
        weight12[:, :, hh] = w12 * Ph
        weight21[:, :, hh] = w21 * Ph
        weight22[:, :, hh] = w22 * Ph
    end

    weight11 = permutedims(weight11, (1, 3, 2))
    weight12 = permutedims(weight12, (1, 3, 2))
    weight21 = permutedims(weight21, (1, 3, 2))
    weight22 = permutedims(weight22, (1, 3, 2))

    rows = repeat(1:N, 4 * nse)
    cols = [vec(index11); vec(index21); vec(index12); vec(index22)]
    vals = [vec(weight11); vec(weight21); vec(weight12); vec(weight22)]

    return sparse(rows, cols, vals, N, N)
end
