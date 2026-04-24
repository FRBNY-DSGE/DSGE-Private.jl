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

#include(joinpath(@__DIR__, "../reference/kalman_filter.jl"))

mat_contents = matread("../models/mBBQ/data/pq_in.mat")
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

# mat_contents = matread("../models/mBBQ/data/pq_in2.mat")
# # data_est       = mat_contents["data_est"]
# # H_aux          = mat_contents["H_aux"]
# ZLB_duration_1 = mat_contents["ZLB_duration_1"]
# Fss_ZLB        = mat_contents["Fss_ZLB"]





jld2file = "../models/mBBQ/data/XssYss.jld2"
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
# m.dicts[:ZLB_duration_1] = ZLB_duration_1
# Fss_ZLB = zeros(534, 1)
# m.dicts[:Fss_ZLB] = Fss_ZLB


df = DSGE.load_data_bbq(m)

H_aux, P_ref, Q_ref, HQ, SIGMA_full, SIGMA = DSGE.pq(m, hx, gx, F1_aux, F2_aux, F3_aux, F4_aux)

regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es = DSGE.prepare_PQ_regimes(m, df, H_aux, P_ref, Q_ref, HQ, SIGMA_full, SIGMA)
t0 = time()

loglh, _s_pred, _P_pred, _s_filt, _P_filt, _s0, _P0, _sT, _PT =
    kalman_filter(y, Ts[1], Rs[1], Cs[1], Qs[1], Zs[1], Ds[1], Es[1];
                  outputs = [:loglh, :pred, :filt])

                  t1 = time()
                  println("kalman: $(t1 - t0) seconds")