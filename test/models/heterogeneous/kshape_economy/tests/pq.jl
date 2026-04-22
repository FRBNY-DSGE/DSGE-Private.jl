using Revise
using Test
using JLD2
using DSGE
using MAT
using OrderedCollections: OrderedDict
using CSV
using DataFrames


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

mat_contents = matread("../data/pq_out.mat")
data_est_original            = mat_contents["data_est"]
H_aux_original               = mat_contents["H_aux"]
P_ref_original               = mat_contents["P_ref"]
Q_ref_original               = mat_contents["Q_ref"]
HQ_original                  = mat_contents["HQ"]
SIGMA_full_original          = mat_contents["SIGMA_full"]
SIGMA_original               = mat_contents["SIGMA"]
ZLB_indicator_original       = mat_contents["ZLB_indicator"]
ZLB_duration_1_original      = mat_contents["ZLB_duration_1"]
unique_EZLB_duration_1_original = mat_contents["unique_EZLB_duration_1"]
Ps_aux_original              = mat_contents["Ps_aux"]
Ds_aux_original              = mat_contents["Ds_aux"]
Es_aux_original              = mat_contents["Es_aux"]
D_ZLB_original               = mat_contents["D_ZLB"]
indicator_original           = mat_contents["indicator"]

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


@testset "pq output comparison" begin
    tol = 1e-5

    @test size(H_aux) == size(H_aux_original)
    @test H_aux ≈ H_aux_original atol=tol

    @test size(P_ref) == size(P_ref_original)
    @test P_ref ≈ P_ref_original atol=tol

    @test size(Q_ref) == size(Q_ref_original)
    @test Q_ref ≈ Q_ref_original atol=tol

    @test size(HQ) == size(HQ_original)
    @test HQ ≈ HQ_original atol=tol

    @test size(SIGMA_full) == size(SIGMA_full_original)
    @test SIGMA_full ≈ SIGMA_full_original atol=tol

    @test size(SIGMA) == size(SIGMA_original)
    @test SIGMA ≈ SIGMA_original atol=tol

    @test size(ZLB_indicator) == size(ZLB_indicator_original)
    @test ZLB_indicator ≈ ZLB_indicator_original atol=tol

    @test size(ZLB_duration_1) == size(ZLB_duration_1_original)
    @test ZLB_duration_1 ≈ ZLB_duration_1_original atol=tol

    @test size(unique_EZLB_duration_1) == size(unique_EZLB_duration_1_original)
    @test unique_EZLB_duration_1 ≈ unique_EZLB_duration_1_original atol=tol

    @test size(Ps_aux) == size(Ps_aux_original)
    @test Ps_aux ≈ Ps_aux_original atol=tol

    @test size(Ds_aux) == size(Ds_aux_original)
    @test Ds_aux ≈ Ds_aux_original atol=tol

    @test size(Es_aux) == size(Es_aux_original)
    @test Es_aux ≈ Es_aux_original atol=tol

    @test size(D_ZLB) == size(D_ZLB_original)
    @test D_ZLB ≈ D_ZLB_original atol=tol

    @test indicator == indicator_original
end


aligned_elements_tol = sum(abs.(Ps_aux .- Ps_aux_original) .< 1e-5)
total_elements_Ps_aux = length(Ps_aux)

aligned_elements_Es_aux_tol = sum(abs.(Es_aux .- Es_aux_original) .< 1e-5)
total_elements_Es_aux = length(Es_aux)
