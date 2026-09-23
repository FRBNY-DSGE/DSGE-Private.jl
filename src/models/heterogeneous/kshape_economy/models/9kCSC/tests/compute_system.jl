using Revise
using Test
using JLD2
using DSGE
using MAT
using OrderedCollections: OrderedDict
using CSV
using StateSpaceRoutines

input_path = joinpath(@__DIR__, "../input_data/state_reduc_tvcopula.mat")
println("\nLoading input fixtures from $input_path")
input_data = matread(input_path)
#TODO: THESE SHOULD BE JLD2 inSTEAD OF MAT
mu_dist = input_data["mu_dist"]
Value = input_data["Value"]
mutil_c = input_data["mutil_c"]
Va = input_data["Va"]

jld2file = "FXaux.jld2"
@load jld2file F1_aux F2_aux F3_aux F4_aux

jld2file = "XssYss.jld2"
@load jld2file Xss Yss SS_stats grid param
# grid = Dict{Symbol, Any}(Symbol(k) => grid[k] for k in ["nb", "na", "nse", "ns", "nCOP", "oc", "os", "numstates", "s_dist", "s", "K"])

# mat_contents = matread("../models/mBBQ/data/pq_in2.mat")
# ZLB_duration_1 = mat_contents["ZLB_duration_1"]
# Fss_ZLB        = mat_contents["Fss_ZLB"]


m = kCSC()

#set param values
#original post mode
# INIT = [
#     0.0215, 0.9085, 0.3205, 5.1115, 1.4815, 0.3855, 0.9985, 0.6135, 0.8335,
#     0.7315, 0.8825, 0.9515, 0.8885, 0.9575, 13.0915, 0.6725, 0.1085, 0.7965,
#     3.6715, 0.0915, 3.2885, 0.2015, 0.1755, 1.0145, 0.9185, 0.0555
# ]

#prev working
# INIT = [
#     0.0215, 0.9085, 0.3205, 5.1115, 1.4815, 0.3855, 0.9985, 0.6135, 0.8335,
#     0.85, 0.8825, 0.9515, 0.8885, 0.9575, 13.0915, 0.6725, 0.1085, 0.7965,
#     3.6715, 0.0915, 3.2885, 0.2015, 0.1755, 1.0145, 0.9185, 0.06211909117098097
# ]

# INIT = [
#     0.0215, 0.9085, 0.3205, 5.1115, 1.4815, 0.3855, 0.9985, 0.6135, 0.8335,
#     0.85, 0.8825, 0.9515, 0.8885, 0.9575, 13.0915, 0.6725, 0.1085, 0.7965,
#     3.6715, 0.0915, 3.2885, 0.2015, 0.1755, 1.0145, 0.9185, 0.06211909117098097
# ]

# for i = 1:20

# using Random
# idx = rand(1:length(INIT))
# INIT[idx] += 0.1 * (2rand() - 1)



# m[:κ] = INIT[1]
# m[:ρ_w] = INIT[2]
# m[:d] = 1 - INIT[3]
# m[:ϕ] = INIT[4] * 10
# m[:ϕ_π] = INIT[5]
# m[:ϕ_u] = INIT[6]
# m[:ρ_BB] = INIT[7]
# m[:ρ_B] = INIT[8]
# m[:ρ_D] = INIT[9]
# m[:ρ_G] = INIT[10]
# m[:ρ_R] = INIT[11]
# m[:ρ_Z] = INIT[12]
# m[:ρ_η] = INIT[13]
# m[:ρ_ι] = INIT[14]
# m[:σ_D] = INIT[15] / 100
# m[:σ_G] = INIT[16] / 100
# m[:σ_R] = INIT[17] / 100
# m[:σ_Z] = INIT[18] / 100
# m[:σ_η] = INIT[19] / 100
# m[:σ_ι] = INIT[20] / 100
# m[:σ_w] = INIT[21] / 100
# m[:σ_BB] = INIT[22] / 100
# m[:γ] = INIT[23]
# m[:σ_B_F] = INIT[24] / 100
# m[:ρ_B_F] = INIT[25]
# m[:ι] = INIT[26]





m.dicts[:grid] = grid
m.dicts[:SS_stats] = SS_stats
m.dicts[:param] = param
m.grids[:StateSS] = Xss
m.grids[:ControlSS] = Yss

m.grids[:mu_dist] = mu_dist
m.grids[:Value] = Value
m.grids[:mutil_c] = mutil_c
m.grids[:Va] = Va
# m.dicts[:ZLB_duration_1] = ZLB_duration_1

Jacob_base2 = DSGE.save_jacob_base(F1_aux, F2_aux, F3_aux, F4_aux, m)


#load in basejacob
# mat_contents = matread("../models/mBBQ/data/TESTJACOBBASE.mat")
# Jacob_base = mat_contents["Jacob_base"] #for testing use, this but in prod use save_jacob_base based on tvcopula

m.dicts[:Jacob_base] = Jacob_base2

# TESTING!, need to make call to F_sys_ZLB_QE
# Fss_ZLB = zeros(632, 1)
# m.dicts[:Fss_ZLB] = Fss_ZLB

system = DSGE.compute_system!(m)
t0 = time()
if !isnothing(system)
    df = DSGE.load_data_bbq(m)[:, 2:end]
    Ny_data = size(df, 2)
    Nt = size(df, 1)
    y = zeros(Float64, Ny_data, Nt)
    for (row, sym) in enumerate(m.grids[:obs])
        y[row, :] .= Float64.(df[1:Nt, sym])
    end

    regime_inds = UnitRange{Int}[1:Nt]
    Ts = [system[:TTT]]
    Rs = [system[:RRR]]
    Cs = [system[:CCC]]
    Qs = [system[:QQ]]
    Zs = [system[:ZZ]]
    Ds = [system[:DD]]
    Es = [system[:EE]]

    loglh, _s_pred, _P_pred, _s_filt, _P_filt, _s0, _P0, _sT, _PT =
    StateSpaceRoutines.kalman_filter(regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es;
                  outputs = [:loglh, :pred, :filt])
                
println(sum(loglh))
else
    loglh = Inf

end
t1 = time()
println("kalman: $(t1 - t0) seconds")
