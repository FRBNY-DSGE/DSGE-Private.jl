using Revise
using Test
using JLD2
using DSGE
using MAT
using OrderedCollections: OrderedDict
using CSV
using DataFrames
using Dates


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
df = df[:, 2:end]  # remove first column
df = DataFrame(Matrix(df)', :auto)  

observable_names = [
    :ygpc,
    :cgpc,
    :igpc,
    :rcb,
    :wg,
    :u,
    :lt,
    :p,
    :xcb,
    :sp500,
]

rename!(df, Dict(names(df)[1:length(observable_names)] .=> observable_names))

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
