using Revise
using Test
using JLD2
using DSGE
using MAT
using OrderedCollections: OrderedDict
using CSV
using DataFrames
using Dates
using LinearAlgebra
using StateSpaceRoutines: kalman_filter

include(joinpath(@__DIR__, "../reference/kalman_filter.jl"))

mat_contents = matread("../data/pq_in.mat")
hx            = mat_contents["hx"]
gx            = mat_contents["gx"]
F1_aux        = mat_contents["F1_aux"]
F2_aux        = mat_contents["F2_aux"]
F3_aux        = mat_contents["F3_aux"]
F4_aux        = mat_contents["F4_aux"]
param         = mat_contents["param"]
indicator_1   = mat_contents["indicator_1"]
F1_ZLB_aux    = mat_contents["F1_ZLB_aux"]
F2_ZLB_aux    = mat_contents["F2_ZLB_aux"]
F3_ZLB_aux    = mat_contents["F3_ZLB_aux"]
F4_ZLB_aux    = mat_contents["F4_ZLB_aux"]

mat_contents = matread("../data/pq_in2.mat")
data_est       = mat_contents["data_est"]
H_aux          = mat_contents["H_aux"]
ZLB_duration_1 = mat_contents["ZLB_duration_1"]
Fss_ZLB        = mat_contents["Fss_ZLB"]


jld2file = "../data/XssYss.jld2"
@load jld2file StateSS ControlSS SS_stats grid param

m = mBBQ()
m.dicts[:grid] = grid
m.dicts[:SS_stats] = SS_stats
m.dicts[:param] = param
# m.grids[:StateSS] = StateSS
# m.grids[:ControlSS] = ControlSS
# m.grids[:mu_dist] = mu_dist
# m.grids[:Value] = Value
# m.grids[:mutil_c] = mutil_c
# m.grids[:Va] = Va
# m.dicts[:Jacob_base] = Jacob_base2

H_aux, P_ref, Q_ref, HQ, SIGMA_full, SIGMA, ZLB_indicator, ZLB_duration_1, unique_EZLB_duration_1, Ps_aux, Ds_aux, Es_aux, D_ZLB, indicator = DSGE.pq(m, hx, gx, F1_aux, F2_aux, F3_aux, F4_aux, indicator_1, F1_ZLB_aux, F2_ZLB_aux, F3_ZLB_aux, F4_ZLB_aux, data_est, H_aux, ZLB_duration_1, Fss_ZLB)



df = CSV.read("../data/data.csv", DataFrame; header=false)
df = df[1:end-1, 2:end]  # remove first column
df = DataFrame(Matrix(df)', :auto)  

observable_names = [
    :ygpc,          # Real GDP Growth per capita
    :cgpc,          # Real Consumption Growth per capita
    :igpc,          # Real Investment Growth
    :pi,             # Inflation
    :rcb,           # Nominal Interest Rate
    :wg,            # Real Wage Growth
    :u,             # Unemployment Rate
    :lt,            # Lumpsum Transfer
    :profit,        # Profit
    :xcb,           # Central Bank Assets
    # :sp500,         # Stock Returns
]

rename!(df, observable_names)

# Quarter-end dates: first row = 1992 Q1 (last day 1992-03-31), then successive quarters.
n = nrow(df)
quarter_dates = Vector{Date}(undef, n)
for i in 1:n
    off = i - 1
    yy = 1992 + off ÷ 4
    qq = 1 + (off % 4)
    quarter_dates[i] =
        qq == 1 ? Date(yy, 3, 31) :
        qq == 2 ? Date(yy, 6, 30) :
        qq == 3 ? Date(yy, 9, 30) :
                  Date(yy, 12, 31)
end
insertcols!(df, 1, :date => quarter_dates)


Ny_data, Nt_data = size(data_est)
if Ny_data != length(observable_names)
    error("Number of rows in data_est does not match observable_names")
end
if Ny_data != size(H_aux, 1)
    error("Number of rows in data_est does not match number of rows in H_aux")
end
if ncol(df) != 1 + length(observable_names)
    error("Unexpected number of columns in df")
end

Ny = Ny_data
Nt = min(Nt_data, nrow(df))
if Nt < 1
    error("No data to process (Nt < 1)")
end

y = zeros(Float64, Ny, Nt)
for (row, sym) in enumerate(observable_names)
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

zi = zlb_indicator_quarters(ZLB_indicator, Nt)

udur = Float64.(unique_EZLB_duration_1)
if !issorted(udur)
    error("udur is not sorted")
end

zdur = zlb_duration_per_period(vec(ZLB_duration_1), zi, Nt)

kk = zeros(Int, Nt)
for t in 1:Nt
    kk[t] = occbin_slice_index(udur, zdur[t], zi[t])
end

regime_inds, regime_keys = occbin_regime_partition(zi, kk)

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

loglh, _s_pred, _P_pred, _s_filt, _P_filt, _s0, _P0, _sT, _PT =
    kalman_filter(regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es;
                  outputs = [:loglh, :pred, :filt])

if length(loglh) != Nt
    error("Length of loglh does not match Nt")
end
if !all(isfinite, loglh)
    error("Non-finite value(s) found in loglh")
end
if !isfinite(sum(loglh))
    error("Sum of loglh is not finite")
end
