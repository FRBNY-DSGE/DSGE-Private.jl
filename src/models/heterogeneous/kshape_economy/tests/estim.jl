using Revise
using Test
using JLD2
using DSGE
using MAT
using OrderedCollections: OrderedDict
using CSV
using StateSpaceRoutines

input_path = joinpath(@__DIR__, "../models/mBBQ/data/inputs/state_reduc_tvcopula.mat")
println("\nLoading input fixtures from $input_path")
input_data = matread(input_path)
#TODO: THESE SHOULD BE JLD2 inSTEAD OF MAT
mu_dist = input_data["mu_dist"]
Value = input_data["Value"]
mutil_c = input_data["mutil_c"]
Va = input_data["Va"]

jld2file = "../models/mBBQ/data/sgufx.jld2"
@load jld2file F1_aux_rep F2_aux_rep F3_aux_rep F4_aux_rep

jld2file = "../models/mBBQ/data/XssYss.jld2"
@load jld2file StateSS ControlSS SS_stats grid param


m = mBBQ()

#set param values
#original post mode
# INIT = [
#     0.0215, 0.9085, 0.3205, 5.1115, 1.4815, 0.3855, 0.9985, 0.6135, 0.8335,
#     0.7315, 0.8825, 0.9515, 0.8885, 0.9575, 13.0915, 0.6725, 0.1085, 0.7965,
#     3.6715, 0.0915, 3.2885, 0.2015, 0.1755, 1.0145, 0.9185, 0.0555
# ]

INIT = [
    0.0215, 0.9085, 0.3205, 5.1115, 1.4815, 0.3855, 0.9985, 0.6135, 0.8335,
    0.85, 0.8825, 0.9515, 0.8885, 0.9575, 13.0915, 0.6725, 0.1085, 0.7965,
    3.6715, 0.0915, 3.2885, 0.2015, 0.1755, 1.0145, 0.9185, 0.06211909117098097
]


m[:κ] = INIT[1]
m[:ρ_w] = INIT[2]
m[:d] = 1 - INIT[3]
m[:ϕ] = INIT[4] * 10
m[:ϕ_π] = INIT[5]
m[:ϕ_u] = INIT[6]
m[:ρ_BB] = INIT[7]
m[:ρ_B] = INIT[8]
m[:ρ_D] = INIT[9]
m[:ρ_G] = INIT[10]
m[:ρ_R] = INIT[11]
m[:ρ_Z] = INIT[12]
m[:ρ_η] = INIT[13]
m[:ρ_ι] = INIT[14]
m[:σ_D] = INIT[15] / 100
m[:σ_G] = INIT[16] / 100
m[:σ_R] = INIT[17] / 100
m[:σ_Z] = INIT[18] / 100
m[:σ_η] = INIT[19] / 100
m[:σ_ι] = INIT[20] / 100
m[:σ_w] = INIT[21] / 100
m[:σ_BB] = INIT[22] / 100
m[:γ] = INIT[23]
m[:σ_B_F] = INIT[24] / 100
m[:ρ_B_F] = INIT[25]
m[:ι] = INIT[26]





m.dicts[:grid] = grid
m.dicts[:SS_stats] = SS_stats
m.dicts[:param] = param
m.grids[:StateSS] = StateSS
m.grids[:ControlSS] = ControlSS

m.grids[:mu_dist] = mu_dist
m.grids[:Value] = Value
m.grids[:mutil_c] = mutil_c
m.grids[:Va] = Va

Jacob_base2 = DSGE.save_jacob_base(F1_aux_rep, F2_aux_rep, F3_aux_rep, F4_aux_rep, m)

m.dicts[:Jacob_base] = Jacob_base2


# if flag == true
#     loglh, _s_pred, _P_pred, _s_filt, _P_filt, _s0, _P0, _sT, _PT =
#     StateSpaceRoutines.kalman_filter(regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es;
#                   outputs = [:loglh, :pred, :filt])
                

# else
#     loglh = Inf

df = DSGE.load_data_bbq(m)
df_obs = df[:, 2:end]
Ny_data = size(df_obs, 2)
Nt = size(df_obs, 1)
data = zeros(Float64, Ny_data, Nt)
for (row, sym) in enumerate(m.grids[:obs])
    data[row, :] .= Float64.(df_obs[1:Nt, sym])
end


t0 = time()

# system = DSGE.compute_system(m)
# lh = sum(DSGE.filter_likelihood(m, data, system))
# println(lh)

DSGE.likelihood(m, df)

t1 = time()
println("kalman: $(t1 - t0) seconds")
