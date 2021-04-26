using DSGE, ModelConstructors, OrderedCollections, SparseArrays, ForwardDiff, Plots, Random, GR
using KrylovKit, JLD2
using FFTW: dct
gr()
GR.inline("pdf")
include("../LinearizationFunctions/FSYS_agg.jl")
include("../LinearizationFunctions/FSYS.jl")
include("../LinearizationFunctions/SolveDiffEq.jl")
include("../LinearizationFunctions/SGU.jl")

run_prep = false
run_ss = false

if run_prep
    m = BayerBornLuetticke()

    if run_ss
        @time KSS, VmSS, VkSS, distrSS = DSGE.find_steadystate(m; verbose = :high)
        JLD2.jldopen("steadystateout.jld2", true, true, true, IOStream) do file
            write(file, "KSS", KSS)
            write(file, "VmSS", VmSS)
            write(file, "VkSS", VkSS)
            write(file, "distrSS", distrSS)
        end
    else
        out = JLD2.jldopen("steadystateout.jld2", "r")
        KSS = out["KSS"]
        VkSS = out["VkSS"]
        VmSS = out["VmSS"]
        distrSS = out["distrSS"]
        DSGE.init_grids!(m)
    end

    @time DSGE.prepare_linearization(m, KSS, VmSS, VkSS, distrSS; verbose = :high)
end

m = BayerBornLuetticke()
DSGE.init_grids!(m; coarse = true)
θ = parameters2namedtuple(m)
# bblout = JLD2.jldopen("bbl_steadystate_out.jld2", "r")
# DSGE.prepare_linearization(m, bblout["KSS"], bblout["VmSS"], bblout["VkSS"], bblout["distrSS"]; verbose = :high)
# out = JLD2.jldopen("steadystateout.jld2", "r")
# DSGE.prepare_linearization(m, out["KSS"], out["VmSS"], out["VkSS"], out["distrSS"]; verbose = :high)
count = 196
input = JLD2.jldopen("ksupply_$(count)_input.jld2", "r")

n = input["n"]
EVk = reshape(reshape(input["Vk"], (n[1] * n[2], n[3])) * input["Pi"]', (n[1], n[2], n[3]))
EVm = reshape((reshape(input["eff_int"], (n[1] * n[2], n[3])) .*
               reshape(input["Vm"], (n[1] * n[2], n[3]))) * input["Pi"]', (n[1], n[2], n[3]))

# Policy update step
c_a_star, m_a_star, k_a_star, c_n_star, m_n_star =
    DSGE.EGM_policyupdate(EVm, EVk, 1., m[:π].value, input["RB"], 1.0, input["inc"], θ, m.grids, false)

# marginal value update step
Vk_new, Vm_new  = DSGE.updateV(EVk, c_a_star, c_n_star, m_n_star, input["R"] - 1.0, 1., θ, get_gridpts(m, :m_grid), input["Pi"])

output = JLD2.jldopen("ksupply_$(count)_output.jld2", "r")

@show maximum(abs, output["c_a_star"] - c_a_star)
@show maximum(abs, output["m_a_star"] - m_a_star)
@show maximum(abs, output["k_a_star"] - k_a_star)
@show maximum(abs, output["c_n_star"] - c_n_star)
@show maximum(abs, output["m_n_star"] - m_n_star)
@show maximum(abs, output["Vm_new"] - Vm_new)
@show maximum(abs, output["Vk_new"] - Vk_new)

K_guess = 4.835647750603774
grids = m.grids
    H                   = m.grids[:H]
    HW                  = m.grids[:HW]
    N           = DSGE._bbl_employment(K_guess, 1.0 / (m[:μ_p] * m[:μ_w]), m[:α],      # employment
                                  m[:τ_lev], m[:τ_prog], m[:γ])
    w           = DSGE._bbl_wage(K_guess, 1.0 / m[:μ_p], N, m[:α])                     # wages
    rk          = DSGE._bbl_interest(K_guess, 1.0 / m[:μ_p], N, m[:α], m[:δ_0])        # Return on illiquid asset
    profits     = (1.0 - 1.0 / m[:μ_p]) .* DSGE._bbl_output(K_guess, 1.0, N, m[:α])    # Profit income
    RB          = m[:RB] / m[:π]                                                  # Real return on liquid assets
    neg_liq_ret = RB + m[:Rbar]
    eff_int     = [x <= 0. ? neg_liq_ret : RB for x in grids[:m_ndgrid]]        # effective rate depending on assets
    GHHFA       = (m[:γ] + m[:τ_prog]) / (m[:γ] + 1.0)                            # transformation (scaling) for composite good
    Paux            = m.grids[:Paux]                                 # Grab ergodic income distribution from transitions
    distr_y         = Paux[1, :]                                                 # stationary income distribution
    inc             = Array{Array{Float64, 3}}(undef, 4)                         # container for income
    mcw             = 1.0 / m[:μ_w]                                              # wage markup
    incgross        = grids[:y_grid].points .* mcw .* w .* N ./ H      # gross income workers (wages)
    incgross[end]   = grids[:y_grid].points[end] * profits                     # gross income entrepreneurs (profits)
    incnet          = m[:τ_lev] * incgross .^ (1.0 - m[:τ_prog])

    # average tax rate
    av_tax_rate     = dot((incgross - incnet), distr_y) / dot(incgross, distr_y)

    # TODO: replace the y_ndgrid calculation with just repeating the incnet vector OR use list comprehension later on
    ny              = get_setting(m, :coarse_ny)
    av_tax_rate     = dot((incgross - incnet), distr_y) / dot(incgross, distr_y)

    inc[1]          = GHHFA .* m[:τ_lev] .* (grids[:y_ndgrid] .* mcw .* w .* N ./ H) .^ (1.0 - m[:τ_prog]) .+
        (1.0 .- mcw) .* w .* N * (1.0 .- av_tax_rate) .* HW         # labor income net of taxes incl. union profits
    inc[1][:,:,end] = m[:τ_lev] .* (grids[:y_ndgrid][ :, :, end] * profits) .^ (1.0 - m[:τ_prog]) # profit income net of taxes

    # incomes out of wealth # TODO: replace these steps OR use list comprehension later on
    inc[2]          = rk .* grids[:k_ndgrid]                                  # rental income
    inc[3]          = eff_int .* grids[:m_ndgrid]                             # liquid asset income
    inc[4]          = grids[:k_ndgrid]                                        # capital liquidation income (q=1 in steady state)

@show maximum(abs, inc[1] - input["inc"][1])
@show maximum(abs, inc[2] - input["inc"][2])
@show maximum(abs, inc[3] - input["inc"][3])
@show maximum(abs, inc[4] - input["inc"][4])

kdiff_init = JLD2.jldopen("kdiff_initial.jld2", "r")
@assert false
#=
nt = DSGE.construct_steadystate_namedtuple(m)
id = DSGE.construct_prime_and_noprime_indices(m; only_aggregate = true)
x  = zeros(length(id) ÷ 2)
x′ = zeros(size(x))
θ = parameters2namedtuple(m)
agg_out = Fsys_agg(x, x′, θ, m.grids, id, nt, DSGE.get_aggregate_equilibrium_conditions(m))
length_X0   = length(id) ÷ 2
aggr_eqconds = DSGE.get_aggregate_equilibrium_conditions(m)
BA          = ForwardDiff.jacobian(x -> DSGE.Fsys_agg(x[1:length_X0], x[length_X0+1:end], θ, m.grids, id, nt,
                                                      aggr_eqconds), zeros(2*length_X0))
Aa          = BA[:,length_X0+1:end]
Ba          = BA[:,1:length_X0]
bblBA = JLD2.jldopen("Fsys_agg_jac.jld2", "r")["BA"]
=#
nt = DSGE.construct_steadystate_namedtuple(m)
id = DSGE.construct_prime_and_noprime_indices(m; only_aggregate = false)
θ = parameters2namedtuple(m)
Γ = DSGE.shuffleMatrix(m[:distr_star])
nm, nk, ny = DSGE.get_idiosyncratic_dims(m)
DC = Array{Array{Float64,2},1}(undef,3)
DC[1]  = DSGE.mydctmx(nm)
DC[2]  = DSGE.mydctmx(nk)
DC[3]  = DSGE.mydctmx(ny)
IDC    = [DC[1]', DC[2]', DC[3]']

DCD = Array{Array{Float64,2},1}(undef,3)
DCD[1]  = DSGE.mydctmx(nm-1)
DCD[2]  = DSGE.mydctmx(nk-1)
DCD[3]  = DSGE.mydctmx(ny-1)
IDCD    = [DCD[1]', DCD[2]', DCD[3]']

#=
x  = zeros(get_setting(m, :n_vars))
x′ = zeros(size(x))
Fsys_out = Fsys(x, x′, θ, m.grids, id, nt, m.equilibrium_conditions,
           get_setting(m, :dct_compression_indices), Γ,
           DC, IDC, DCD, IDCD)
=#
n_vars = get_setting(m, :n_vars)
A = zeros(n_vars, n_vars)
B = zeros(n_vars, n_vars)

    n_dct_Vm    = length(get_setting(m, :dct_compression_indices)[:Vm])
    n_dct_Vk    = length(get_setting(m, :dct_compression_indices)[:Vk])

length_X0 = get_setting(m, :n_vars)
    nxB         = length_X0 - n_dct_Vm - n_dct_Vk
    nxA         = length_X0 - length(id[:marginal_pdf_y_t]) - length(id[:marginal_pdf_m_t]) - length(id[:marginal_pdf_k_t])

#=length_X0 = get_setting(m, :n_vars) - length(id[:marginal_pdf_y_t]) - length(id[:marginal_pdf_m_t]) -
    length(id[:marginal_pdf_k_t]) - length(id[:copula_t])
    nxB         = length_X0 - n_dct_Vm - n_dct_Vk
    nxA         = length_X0 - n_dct_Vm - n_dct_Vk=#

estim = false
Random.seed!(1793)
#=obj_fnct    = x -> Fsys([zeros(length(id[:marginal_pdf_y_t]) + length(id[:marginal_pdf_m_t]) +
                               length(id[:marginal_pdf_k_t]) + length(id[:copula_t])); x[1:22];
                         zeros(n_dct_Vm + n_dct_Vk); x[22+1:nxB]],
                        [zeros(length(id[:marginal_pdf_y_t]) + length(id[:marginal_pdf_m_t]) +
                               length(id[:marginal_pdf_k_t]) + length(id[:copula_t]));
                         x[nxB+1:nxB+22]; zeros(n_dct_Vm + n_dct_Vk); x[nxB+22+1:end]],
                        θ, m.grids, id, nt, m.equilibrium_conditions,
                        get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD)=#

#=obj_fnct    = x -> Fsys([zeros(length(id[:marginal_pdf_y_t]) + length(id[:marginal_pdf_m_t]) +
                               length(id[:marginal_pdf_k_t]) + length(id[:copula_t])); zero(1);
                         x[1:2]; zero(1); x[3:22-2]; zeros(n_dct_Vm + n_dct_Vk);
                         zero(1); x[22-2+1:22-2+2]; zero(1); x[22-2+3:nxB]],
                        [zeros(length(id[:marginal_pdf_y_t]) + length(id[:marginal_pdf_m_t]) +
                               length(id[:marginal_pdf_k_t]) + length(id[:copula_t])); zero(1);
                         x[nxB+1:nxB+2]; zero(1); x[nxB+3:nxB+22-2]; zeros(n_dct_Vm + n_dct_Vk);
                         zero(1); x[nxB+22-2+1:nxB+22-2+2]; zero(1); x[nxB+22-2+3:end]],
                        θ, m.grids, id, nt, m.equilibrium_conditions,
                        get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD)=#

obj_fnct    = x -> Fsys([x[1:id[:Vm_t][1]-1]; zeros(n_dct_Vm + n_dct_Vk); x[id[:Vm_t][1]:nxB]],
                        [zeros(length(id[:marginal_pdf_y_t]) + length(id[:marginal_pdf_m_t]) +
                               length(id[:marginal_pdf_k_t])); x[nxB+1:end]],
                        θ, m.grids, id, nt, m.equilibrium_conditions,
                        get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD)
BA          = ForwardDiff.jacobian(obj_fnct, zeros(nxB+nxA))
# bblout = JLD2.jldopen("Fsys_jac_nodistr.jld2", "r")
# bblout = JLD2.jldopen("Fsys_jac_nodistr_skipbadscalars.jld2", "r")
#=err = abs.(BA - bblout["BA"])
@show maximum(err[:, 1])
@show maximum(err[:, 4])
@show maximum(err[:, 23])
@show maximum(err[:, 26])=#
# big errors are at index 1, 4, 23, 26 => A, RB, rk, π (maybe rk, not sure)
# for A, RB, and π, the error is in I90 share and I90 share net
# for rk (maybe), it's the Vm marginal value
# @assert false

    B[:,1:id[:Vm_t][1]-1]                 = BA[:,1:id[:Vm_t][1]-1]
    B[:,id[:Vk_t][end]+1:end]             = BA[:,id[:Vm_t][1]:nxB]
    A[:,id[:marginal_pdf_y_t][end]+1:end] = BA[:,nxB+1:end]

    for i in id[:Vm_t] # this is problematic if id[:Vm_t] is not on the diagonal
        B[i, i] = 1.0
    end
    for i in id[:Vk_t]
        B[i, i] = 1.0
    end

    for count = 1:get_setting(m, :nm)-1 #in eachcol(A)
        i = id[:marginal_pdf_m_t][count]
        A[id[:marginal_pdf_m_t],i] = -Γ[1][1:end-1,count]
    end
    for count = 1:get_setting(m, :nk)-1 #in eachcol(A)
        i = id[:marginal_pdf_k_t][count]
        A[id[:marginal_pdf_k_t],i] = -Γ[2][1:end-1,count]
    end
    for count = 1:get_setting(m, :ny)-1 #in eachcol(A)
        i = id[:marginal_pdf_y_t][count]
        A[id[:marginal_pdf_y_t],i] = -Γ[3][1:end-1,count]
    end

lr = JLD2.jldopen("Fsys_jac_lr.jld2", "r")
A_err = A - lr["A"]
B_err = B - lr["B"]
println("Errors ignoring the distributional aggregate scalar equations")
inds = vcat(1:876, 883:895)
println("Columns 856 (price inflation π)")
println(maximum(abs.(B_err[inds, 856])))
println("Columns 191:855 (aggregate states, value functions, some jumps)")
println(maximum(abs.(B_err[inds, 191:855])))
println("Columns 857:end (remaining jumps)")
println(maximum(abs.(B_err[inds, 857:end])))
println("Columns 1:79 (marginal m)") # DEFINITELY STILL AN ERROR
println(maximum(abs.(B_err[inds, 1:79])))
println("Columns 80:158 (marginal k)")
println(maximum(abs.(B_err[inds, 80:158])))
println("Columns 159:179 (marginal y)")
println(maximum(abs.(B_err[inds, 159:179])))
println("Columns 180:190 (copula)")
println(maximum(abs.(B_err[inds, 180:190])))

inds = 877:882
println("Errors for the distributional aggregate scalar equations")
println("Columns 856 (price inflation π)")
println(maximum(abs.(B_err[inds, 856])))
println("Columns 191:855 (aggregate states, value functions, some jumps)")
println(maximum(abs.(B_err[inds, 191:855])))
println("Columns 857:end (remaining jumps)")
println(maximum(abs.(B_err[inds, 857:end])))
println("Columns 1:79 (marginal m)") # especially for these aggregate distributional scalars
println(maximum(abs.(B_err[inds, 1:79])))
println("Columns 80:158 (marginal k)")
println(maximum(abs.(B_err[inds, 80:158])))
println("Columns 159:179 (marginal y)")
println(maximum(abs.(B_err[inds, 159:179])))
println("Columns 180:190 (copula)")
println(maximum(abs.(B_err[inds, 180:190])))

#     gx, hx, _ = SolveDiffEq(m, A, outAB["B"], estim)
    gx, hx, _ = SolveDiffEq(m, A, B, estim)

TTT = zeros(get_setting(m, :n_vars), get_setting(m, :n_vars))
TTT[1:get_setting(m, :n_states), 1:get_setting(m, :n_states)] = hx
TTT[get_setting(m, :n_states)+1:end, 1:get_setting(m, :n_states)] = gx * hx

TTT_bbl = zeros(get_setting(m, :n_vars), get_setting(m, :n_vars))
TTT_bbl[1:get_setting(m, :n_states), 1:get_setting(m, :n_states)] = lr["hx"]
TTT_bbl[get_setting(m, :n_states)+1:end, 1:get_setting(m, :n_states)] = lr["gx"] * lr["hx"]

errs = Dict()
max_errs = Dict()
argmax_errs = Dict()
for shock in [:A′_t, :Z′_t, :Ψ′_t, :μ_p′_t, :μ_w′_t, :G_sh′_t, :P_sh′_t, :R_sh′_t, :S_sh′_t]
    shock_name = string(DSGE.detexify(shock))[1:end - 3]
    x0 = zeros(get_setting(m, :n_vars))
    x0[first(m.endogenous_states[shock])] = 100. * .01
    T = 40
    myirfs = zeros(get_setting(m, :n_vars), T)
    bblirfs = zeros(get_setting(m, :n_vars), T)
    myirfs[:, 1] = x0
    bblirfs[:, 1] = x0
    @views for t in 2:T
        myirfs[:, t] = TTT * myirfs[:, t - 1]
        bblirfs[:, t] = TTT_bbl * bblirfs[:, t - 1]
    end

    p = Plots.plot(1:T, myirfs[first(m.endogenous_states[:C′_t]), :], label = "Mine", linewidth = 1)
    Plots.plot!(1:T, bblirfs[first(m.endogenous_states[:C′_t]), :], label = "BBL", linewidth = 1, linestyle = :dash,
          title = "C")
    DSGE.save_plot(p, "myirf_$(shock_name)_C.pdf")
    p = Plots.plot(1:T, myirfs[first(m.endogenous_states[:π′_t]), :], label = "Mine", linewidth = 1)
    Plots.plot!(1:T, bblirfs[first(m.endogenous_states[:π′_t]), :], label = "BBL", linewidth = 1, linestyle = :dash,
          title = "pi")
    DSGE.save_plot(p, "myirf_$(shock_name)_pi.pdf")
    p = Plots.plot(1:T, myirfs[219, :], label = "Mine", linewidth = 1)
    Plots.plot!(1:T, bblirfs[219, :], label = "BBL", linewidth = 1, linestyle = :dash,
          title = "Vm index 9")
    DSGE.save_plot(p, "myirf_$(shock_name)_Vm_index9.pdf")
    p = Plots.plot(1:T, myirfs[26, :], label = "Mine", linewidth = 1)
    Plots.plot!(1:T, bblirfs[26, :], label = "BBL", linewidth = 1, linestyle = :dash,
          title = "Marginal m index 26")
    DSGE.save_plot(p, "myirf_$(shock_name)_marginal_pdf_m_index26.pdf")
    p = Plots.plot(1:T, myirfs[80, :], label = "Mine", linewidth = 1)
    Plots.plot!(1:T, bblirfs[80, :], label = "BBL", linewidth = 1, linestyle = :dash,
          title = "Marginal k index 1")
    DSGE.save_plot(p, "myirf_$(shock_name)_marginal_pdf_k_index1.pdf")
    for k in DSGE.get_aggregate_jump_variables(m)[25:30]
        p = Plots.plot(1:T, myirfs[first(m.endogenous_states[k]), :], label = "Mine", linewidth = 1)
        Plots.plot!(1:T, bblirfs[first(m.endogenous_states[k]), :], label = "BBL", linewidth = 1, linestyle = :dash,
              title = "$(string(DSGE.detexify(k))[1:end-3])")
        DSGE.save_plot(p, "myirf_$(shock_name)_$(string(DSGE.detexify(k))[1:end-3]).pdf")
    end
    errs[shock] = abs.(myirfs - bblirfs)
    max_errs[shock] = maximum((errs[shock]))
    argmax_errs[shock] = argmax((errs[shock]))
end

# Next steps. Git save everything. Don't try to speed it up just yet, but instead produce
# IRFs to all aggregate dynamics. If the aggregate dynamics look the same, then maybe differences are fine in the end
# and can be attributed to numerical differences arising from differences in my solution code.
# @time out = SGU(m, A, B; estim = false)
