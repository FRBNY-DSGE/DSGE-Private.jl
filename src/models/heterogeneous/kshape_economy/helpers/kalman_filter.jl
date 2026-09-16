using Dates

# Regime construction for `StateSpaceRoutines.kalman_filter(regime_indices, y, Ts, ...)`.
#
# Two conventions used in this codebase / DSGE.jl:
#
# 1) **DSGE.jl calendar ZLB** — `src/estimate/kalman.jl`, `zlb_regime_indices(m, data, start_date)`
#    - Requires `n_mon_anticipated_shocks(m) > 0`.
#    - Splits sample indices `1:T` at ONE break: the first quarter of the ZLB window
#      measured from `start_date` to `date_zlb_start(m)` via `subtract_quarters`.
#    - Returns at most two contiguous `UnitRange`s: pre-ZLB and post-ZLB.
#    - Matrices per regime come from `zlb_regime_matrices` (e.g. `QQ` differs pre/post);
#      `TTT`, `RRR`, … are usually identical across those two blocks.
#
# 2) **Kshape / OccBin (MATLAB `post_density_covid_Kshape_MCMC_v2.m`)**
#    - ZLB is **endogenous** per period: `ZLB_indicator(t)` from the policy rate.
#    - OccBin depth varies with `ZLB_duration_1` / `unique_EZLB_duration_1`, so `(T,R,C)`
#      can change whenever `(in_ZLB, occbin_slice_k)` changes — partition into contiguous
#      runs for `kalman_filter(regime_indices, ...)`.
#
# This file implements (2) for the kshape tests and exposes (1) as a small helper
# mirroring the DSGE **index** logic (matrix choice remains the caller’s job).

"""
    dsgestyle_zlb_regime_indices(start_date, zlb_start_date, T)

Return `Vector{UnitRange{Int}}` partitioning `1:T` the same way as
`DSGE.zlb_regime_indices` when there is a single ZLB onset inside the sample:
`1:n_nozlb` then `n_nozlb+1:T`, with `n_nozlb = subtract_quarters(zlb_start_date, start_date)`.

If the ZLB start is not strictly inside `(0, T)` in quarter units, returns `[1:T]`.
"""
function dsgestyle_zlb_regime_indices(start_date::Dates.Date, zlb_start_date::Dates.Date, T::Int)
    n_nozlb = DSGE.subtract_quarters(zlb_start_date, start_date)
    if n_nozlb <= 0 || n_nozlb >= T
        return UnitRange{Int}[1:T]
    end
    return UnitRange{Int}[1:n_nozlb, (n_nozlb + 1):T]
end

"""Pad or truncate `ZLB_indicator` to length `Nt` (calendar quarters)."""
function zlb_indicator_quarters(ZLB_indicator, Nt::Int)
    indv = Bool.(vec(ZLB_indicator))
    return vcat(indv, fill(false, max(0, Nt - length(indv))))[1:Nt]
end

"""
    zlb_duration_per_period(zdur_raw, zi, Nt)

`zdur_raw` is `vec(ZLB_duration_1)` from MAT. If it has one value per ZLB quarter
(MATLAB `ZLB_duration_1(ZLB_num_1)`), it is shorter than `Nt`; advance an index
only when `zi[t]` is true. If `length(zdur_raw) ≥ Nt`, treat it as per-calendar-`t`.
"""
function zlb_duration_per_period(zdur_raw::AbstractVector{<:Real}, zi::AbstractVector{Bool}, Nt::Int)
    zdur = zeros(Float64, Nt)
    lz = length(zdur_raw)
    if lz >= Nt
        zdur .= Float64.(zdur_raw[1:Nt])
    else
        n_zlb = 0
        for t in 1:Nt
            if zi[t]
                n_zlb += 1
                zdur[t] = Float64(zdur_raw[min(n_zlb, lz)])
            end
        end
    end
    return zdur
end

"""OccBin slice index `k` with `Ps_aux[:,:,k]` etc.; same rule as MATLAB `sum(unique .<= Tmax)`."""
function occbin_slice_index(udur::AbstractVector{<:Real}, Tmax::Real, in_zlb::Bool)
    in_zlb || return 0
    return max(1, searchsortedlast(udur, Float64(Tmax)))
end

"""
    occbin_regime_partition(zlb, kk)

Contiguous `UnitRange`s partitioning `1:length(zlb)` where `zlb` is constant and,
when in ZLB (`zlb==true`), `kk` matches the OccBin duration slice index.

Returns `(regime_indices, regime_keys)` with `regime_keys[i]::Tuple{Bool,Int}` as `(in_ZLB, k_slice_or_0)`.
"""
function occbin_regime_partition(zlb::AbstractVector{Bool}, kk::AbstractVector{<:Integer})
    Nt = length(zlb)
    @assert length(kk) == Nt
    ranges = UnitRange{Int}[]
    regime_keys = Tuple{Bool,Int}[]
    start = 1
    for t in 2:Nt
        same = (zlb[t] == zlb[t - 1]) && (!zlb[t] || kk[t] == kk[t - 1])
        if !same
            push!(ranges, start:(t - 1))
            push!(regime_keys, (zlb[t - 1], zlb[t - 1] ? Int(kk[t - 1]) : 0))
            start = t
        end
    end
    push!(ranges, start:Nt)
    push!(regime_keys, (zlb[Nt], zlb[Nt] ? Int(kk[Nt]) : 0))
    return ranges, regime_keys
end

"""Merge contiguous identical `zlb` only (two physical regimes, one `kk` per ZLB spell ignored)."""
function indicator_only_regime_partition(zlb::AbstractVector{Bool})
    Nt = length(zlb)
    kk = zeros(Int, Nt)
    return occbin_regime_partition(zlb, kk)
end


function prepare_PQ_regimes(m, df, H_aux, P_ref, Q_ref, HQ, SIGMA_full, SIGMA, ZLB_indicator, unique_EZLB_duration_1, Ps_aux, Ds_aux, Es_aux, D_ZLB)


    Nt_data, Ny_data = size(df)
    if Ny_data != length(m.grids[:obs])
        error("Number of rows in data_est does not match observable_names")
    end
    if Ny_data != size(H_aux, 1)
        error("Number of rows in data_est does not match number of rows in H_aux")
    end
    if ncol(df) != length(m.grids[:obs])
        error("Unexpected number of columns in df")
    end
    
    Ny = Ny_data
    Nt = min(Nt_data, nrow(df))
    if Nt < 1
        error("No data to process (Nt < 1)")
    end
    
    y = zeros(Float64, Ny, Nt)
    for (row, sym) in enumerate(m.grids[:obs])
        y[row, :] .= Float64.(df[1:Nt, sym])
    end
    
    Ns = size(P_ref, 1)
    EE = Matrix(Diagonal(fill(1e-12, Ny)))
    QQ = Float64.(SIGMA_full)
    ZZ = Float64.(H_aux)
    if size(ZZ, 2) != Ns
        error("Mismatch in size(ZZ, 2) and Ns")
    end
    if size(Q_ref, 2) != size(QQ, 1)
        error("Mismatch in size(Q_ref, 2) and size(QQ, 1)")
    end
    
    zi = DSGE.zlb_indicator_quarters(ZLB_indicator, Nt)
    
    udur = Float64.(unique_EZLB_duration_1)
    if !issorted(udur)
        error("udur is not sorted")
    end
    
    ZLB_duration_1 = m.dicts[:ZLB_duration_1]
    zdur = DSGE.zlb_duration_per_period(vec(ZLB_duration_1), zi, Nt)
    
    kk = zeros(Int, Nt)
    for t in 1:Nt
        kk[t] = DSGE.occbin_slice_index(udur, zdur[t], zi[t])
    end
    
    regime_inds, regime_keys = DSGE.occbin_regime_partition(zi, kk)
    
    Ts = Matrix{Float64}[]
    Rs = Matrix{Float64}[]
    Cs = Vector{Float64}[]
    Qs = Matrix{Float64}[]
    Zs = Matrix{Float64}[]
    Ds = Vector{Float64}[]
    Es = Matrix{Float64}[]
        for (inzlb, kslice) in regime_keys
        if !inzlb
            push!(Ts, Float64.(P_ref))
            push!(Rs, Float64.(Q_ref))
            push!(Cs, zeros(Float64, Ns))
        else
            if !(1 <= kslice <= size(Ps_aux, 3))
                error("kslice $kslice out of bounds for Ps_aux")
            end
            push!(Ts, Float64.(Ps_aux[:, :, kslice]))
            push!(Rs, Float64.(Es_aux[:, :, kslice]))
            push!(Cs, Float64.(Ds_aux[:, kslice]))
            if size(Rs[end], 2) != size(QQ, 1)
                error("Mismatch in size(Rs[end], 2) and size(QQ, 1)")
            end
        end
        push!(Qs, QQ)
        push!(Zs, ZZ)
        push!(Ds, zeros(Float64, Ny))
        push!(Es, EE)
    end

    return regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es
end

function prepare_PQ_regimes(m, df, H_aux, P_ref, Q_ref, HQ, SIGMA_full, SIGMA)
    df = df[:, 2:end]
    Nt_data, Ny_data = size(df)
    if Ny_data != length(m.grids[:obs])
        error("Number of rows in data_est does not match observable_names")
    end
    if Ny_data != size(H_aux, 1)
        error("Number of rows in data_est does not match number of rows in H_aux")
    end
    if ncol(df) != length(m.grids[:obs])
        error("Unexpected number of columns in df")
    end

    Ny = Ny_data
    Nt = min(Nt_data, nrow(df))
    if Nt < 1
        error("No data to process (Nt < 1)")
    end

    y = zeros(Float64, Ny, Nt)
    for (row, sym) in enumerate(m.grids[:obs])
        y[row, :] .= Float64.(df[1:Nt, sym])
    end

    Ns = size(P_ref, 1)
    EE = Matrix(Diagonal(fill(1e-12, Ny)))
    QQ = Float64.(SIGMA_full)
    ZZ = Float64.(H_aux)
    if size(ZZ, 2) != Ns
        error("Mismatch in size(ZZ, 2) and Ns")
    end
    if size(Q_ref, 2) != size(QQ, 1)
        error("Mismatch in size(Q_ref, 2) and size(QQ, 1)")
    end

    regime_inds = UnitRange{Int}[1:Nt]
    Ts = Matrix{Float64}[Float64.(P_ref)]
    Rs = Matrix{Float64}[Float64.(Q_ref)]
    Cs = Vector{Float64}[zeros(Float64, Ns)]
    Qs = Matrix{Float64}[QQ]
    Zs = Matrix{Float64}[ZZ]
    Ds = Vector{Float64}[zeros(Float64, Ny)]
    Es = Matrix{Float64}[EE]

    return regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es
end