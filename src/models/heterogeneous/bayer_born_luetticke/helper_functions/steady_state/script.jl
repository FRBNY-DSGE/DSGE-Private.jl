using DSGE, ModelConstructors, OrderedCollections, SparseArrays, ForwardDiff, Plots, Random, GR
using KrylovKit, JLD2
using FFTW: dct
gr()
GR.inline("pdf")
include("../linearization/SolveDiffEq.jl")
include("../linearization/SGU.jl")

run_prep = true
run_ss = true

if run_prep
    m = BayerBornLuetticke()

    if run_ss
        Random.seed!(1793)
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
#=    θ = parameters2namedtuple(m)
    Random.seed!(1793)
    incgross, incnet, NSS, rkSS, wSS, YSS, ProfitsSS, ISS, RBSS, taxrev, tot_taxrev, avg_tax_rateSS, eff_int = DSGE._bbl_incomes(θ, m.grids, KSS, distrSS)

    KSS, BSS, TransitionMatSS, TransitionMat_aSS, TransitionMat_nSS,
        c_a_starSS, m_a_starSS, k_a_starSS, c_n_starSS, m_n_starSS, VmSS, VkSS, distrSS =
                          DSGE.Ksupply(RBSS, 1.0 + rkSS, m, VmSS, VkSS, distrSS, incnet, eff_int)
#=    JLD2.jldopen("my_ksupply_out.jld2", true, true, true, IOStream) do file
        write(file, "KSS", KSS)
        write(file, "VmSS", VmSS)
        write(file, "VkSS", VkSS)
        write(file, "distrSS", distrSS)
        write(file, "incnet", incnet)
        write(file, "eff_int", eff_int)
        write(file, "c_a_starSS", c_a_starSS)
        write(file, "m_a_starSS", m_a_starSS)
        write(file, "k_a_starSS", k_a_starSS)
        write(file, "c_n_starSS", c_n_starSS)
        write(file, "m_n_starSS", m_n_starSS)
    end=#
    VmSS                = log.(VmSS)
    VkSS                = log.(VkSS)
    # Calculate taxes and government expenditures
    TSS                 = (distrSS[:]' * taxrev[:] + avg_tax_rateSS*((1.0 .- 1.0 ./ m[:μ_w]).*wSS.*NSS))
    GSS                 = TSS - (m[:RB]./m[:π]-1.0)*BSS
    # Random.seed!(1793)
    distr_m_SS, distr_k_SS, distr_y_SS, share_borrowerSS, GiniWSS, I90shareSS,I90sharenetSS, GiniXSS,
    sdlogxSS, P9010CSS, GiniCSS, sdlogCSS, P9010ISS, GiniISS, sdlogySS, W90shareSS, P10CSS, P50CSS, P90CSS =
        DSGE.distrSummaries(distrSS, c_a_starSS, c_n_starSS, incnet, incgross, θ, DSGE.get_idiosyncratic_dims(m), m.grids)
    # @show sdlogxSS, sdlogCSS, sdlogySS

    nm, nk, ny = size(distrSS)
    ThetaVm             = vec(DSGE.dct(VmSS))                             # Discrete cosine transformation of marginal liquid asset value
    ind                 = sortperm(abs.(vec(ThetaVm)); rev = true)   # Indexes of coefficients sorted by their absolute size
    coeffs              = 1                                          # Container to store the number of retained coefficients

    # Find the important basis functions (discrete cosine) for VmSS (in L2 norm)
    while norm(view(ThetaVm, view(ind, 1:coeffs))) / norm(ThetaVm) < 1. - get_setting(m, :dct_energy_loss)
            global coeffs     += 1                                          # add retained coefficients until only some share of energy is lost
    end
    compressionIndexesVm = ind[1:coeffs]                             # store indexes of retained coefficients

    ThetaVk             = vec(DSGE.dct(VkSS))                             # Discrete cosine transformation of marginal illiquid asset value
    ind                 = sortperm(abs.(vec(ThetaVk)); rev = true)   # Indexes of coefficients sorted by their absolute size
    coeffs              = 1                                          # Container to store the number of retained coefficients

    # Find the important basis functions (discrete cosine) for VkSS
    while norm(view(ThetaVk, view(ind, 1:coeffs))) / norm(ThetaVk) < 1. - get_setting(m, :dct_energy_loss)
            global coeffs     += 1                                          # add retained coefficients until only some share of energy is lost
    end
    compressionIndexesVk = ind[1:coeffs]                             # store indexes of retained coefficients

    distr_LOL           = view(distrSS, 1:nm-1, 1:nk-1, 1:ny-1)      # Leave out last entry of histogramm (b/c it integrates to 1)
    ThetaD              = vec(DSGE.dct(distr_LOL))                        # Discrete cosine transformation of Copula
    ind                 = sortperm(abs.(vec(ThetaD)); rev = true)    # Indexes of coefficients sorted by their absolute size
    n_copula_coefs      = get_setting(m, :n_copula_dct_coefficients) # keep n_copula_coefs coefficients, but
    compressionIndexesD = ind[2:1+n_copula_coefs]                    # leave out index no. 1 as this shifts the constant

    compressionIndexes  = Array{Array{Int, 1}, 1}(undef, 3)          # Container to store all retained coefficients in one array
    compressionIndexes[1] = compressionIndexesVm
    compressionIndexes[2] = compressionIndexesVk
    compressionIndexes[3] = compressionIndexesD

    distr_m_SS          = sum(distrSS,dims=(2,3))[:]            # Marginal distribution (pdf) of liquid assets
    distr_k_SS          = sum(distrSS,dims=(1,3))[:]            # Marginal distribution (pdf) of illiquid assets
    distr_y_SS          = sum(distrSS,dims=(1,2))[:]            # Marginal distribution (pdf) of income

    bbl = JLD2.jldopen("bbl_internals_prep_linearization_seed1793.jld2", "r")
    for k in keys(bbl)
        @show k
        tmp = eval(Symbol(k))
        if !isa(tmp, AbstractArray{Float64}) && !isa(tmp, Array{Int64})
            for j in 1:length(tmp)
                @show maximum(abs, tmp[j] - bbl[k][j])
            end
        else
            @show maximum(abs, bbl[k] - tmp)
        end
    end
    @assert false=#
    @time DSGE.prepare_linearization(m, KSS, VmSS, VkSS, distrSS; verbose = :high)
end

include("compare_ss.jl")

# replace TY_star, BY_star, C_star, C_l1_star to make sure they don't differ b/c very small floating point differences,
m[:TY_star] = out["TY_star"] # although the implied gx and hx are still very similar
m[:BY_star] = out["BY_star"]
m[:C_star] = out["C_star"]
m[:C_l1_star] = out["C_l1_star"]
nt = DSGE.construct_steadystate_namedtuple(m)
id = DSGE.construct_prime_and_noprime_indices(m; only_aggregate = false)
θ = parameters2namedtuple(m)
Γ = DSGE.shuffle_matrix(m[:distr_star])
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

n_vars = get_setting(m, :n_vars)
A = zeros(n_vars, n_vars)
B = zeros(n_vars, n_vars)

    n_dct_Vm    = length(get_setting(m, :dct_compression_indices)[:Vm])
    n_dct_Vk    = length(get_setting(m, :dct_compression_indices)[:Vk])

length_X0 = get_setting(m, :n_vars)
    nxB         = length_X0 - n_dct_Vm - n_dct_Vk
    nxA         = length_X0 - length(id[:marginal_pdf_y_t]) - length(id[:marginal_pdf_m_t]) - length(id[:marginal_pdf_k_t])

estim = false

obj_fnct    = x -> DSGE.Fsys([x[1:id[:Vm_t][1]-1]; zeros(n_dct_Vm + n_dct_Vk); x[id[:Vm_t][1]:nxB]],
                             [zeros(length(id[:marginal_pdf_y_t]) + length(id[:marginal_pdf_m_t]) +
                                    length(id[:marginal_pdf_k_t])); x[nxB+1:end]],
                             θ, m.grids, id, nt, m.equilibrium_conditions,
                             get_setting(m, :dct_compression_indices), Γ, DC, IDC, DCD, IDCD)
BA          = ForwardDiff.jacobian(obj_fnct, zeros(nxB+nxA))

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

gx, hx, _ = SolveDiffEq(m, A, B, estim)

lr = JLD2.jldopen("linearize_full_model_output_seed1793.jld2", "r")
A_err = A - lr["A"]
B_err = B - lr["B"]
println("Difference in A, B, gx, and hx")
println(maximum(abs, A_err))
println(maximum(abs, B_err))
println(maximum(abs, gx - lr["gx"]))
println(maximum(abs, hx - lr["hx"]))

#=@assert false
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
=#
