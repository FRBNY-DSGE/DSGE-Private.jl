"""
```
prepare_linearization(m, KSS, VmSS, VkSS, distrSS)
```
Compute a number of equilibrium objects needed for linearization.

# Arguments
- `KSS`: steady-state capital stock
- `VmSS`, `VkSS`: marginal value functions
- `distrSS::Array{Float64,3}`: steady-state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)

# Returns
- `XSS::Array{Float64,1}`, `XSSaggr::Array{Float64,1}`: steady state vectors produced
- `indexes`, `indexes_aggr`: `struct`s for accessing `XSS`,`XSSaggr` by variable names
- `compressionIndexes::Array{Array{Int,1},1}`: indexes for compressed marginal value functions (``V_m`` and ``V_k``)
- `Copula(x,y,z)`: function that maps marginals `x`,`y`,`z` to approximated joint distribution, produced by
        [`mylinearinterpolate3()`](@ref)
- `CDF_SS`, `CDF_m`, `CDF_k`, `CDF_y`: cumulative distribution functions (joint and marginals)
- `distrSS::Array{Float64,3}`: steady state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)
"""
function prepare_linearization(m::BayerBornLuetticke, KSS, VmSS, VkSS, distrSS; verbose::Symbol = :none)

    # Set up
    if verbose in [:low, :high]
        println("Running reduction step to prepare linearization")
    end
    θ  = parameters2namedtuple(m)
    H  = get_untransformed_values(m[:H_star])
    HW = get_untransformed_values(m[:HW_star])
    nm, nk, ny = DSGE.get_idiosyncratic_dims(m)

    # Calculate other equilibrium quantities
    incgross, incnet, NSS, rkSS, wSS, YSS, ProfitsSS, ISS, RBSS, taxrev, tot_taxrev, avg_tax_rateSS, eff_int = _bbl_incomes(θ, m.grids, H, HW, KSS, distrSS)

    # obtain other steady state variables
    KSS, BSS, TransitionMatSS, TransitionMat_aSS, TransitionMat_nSS,
        c_a_starSS, m_a_starSS, k_a_starSS, c_n_starSS, m_n_starSS, VmSS, VkSS, distrSS =
            Ksupply(RBSS, 1.0 + rkSS, m, VmSS, VkSS, distrSS, incnet, eff_int)
            # not passing verbose to Ksupply b/c any print statements in Ksupply are redundant

    VmSS                = log.(VmSS)
    VkSS                = log.(VkSS)

    # Calculate taxes and government expenditures
    TSS                 = (tot_taxrev + avg_tax_rateSS * ((1.0 .- 1.0 ./ θ[:μ_w]) .* wSS .* NSS))
    GSS                 = TSS - (θ[:RB] ./ θ[:π] - 1.0) * BSS

    # Produce distributional summary statistics
    distr_m_SS, distr_k_SS, distr_y_SS, share_borrowerSS, GiniWSS, I90shareSS,I90sharenetSS, GiniXSS,
            sdlogxSS, P9010CSS, GiniCSS, sdlogCSS, P9010ISS, GiniISS, sdlogySS, w90shareSS, P10CSS, P50CSS, P90CSS =
            distrSummaries(distrSS, c_a_starSS, c_n_starSS, incnet, incgross, θ, DSGE.get_idiosyncratic_dims(m), m.grids)

    ## Store quantities in m

    # Aggregate scalars
    m[:K_star] = log(KSS)
    m[:N_star] = log(NSS)
    m[:Y_star] = log(YSS)
    m[:G_star] = log(GSS)
    m[:w_star] = log(wSS)
    m[:T_star] = log(TSS)
    m[:I_star] = log(ISS)
    m[:B_star] = log(BSS)
    m[:avg_tax_rate_star] = log(avg_tax_rateSS)

    # Scalar summary statistics
    m[:share_borrower_star]       = log(share_borrowerSS)
    m[:Gini_wealth_star]          = log(GiniWSS)
    m[:P90_wealth_share_star]     = log(w90shareSS)
    m[:P90_income_star]           = log(I90shareSS)
    m[:P90_income_share_star]     = log(I90sharenetSS)
    m[:Gini_income_star]          = log(GiniCSS)
    m[:P90_minus_P10_income_star] = log(P9010ISS)
    m[:sd_log_income_star]        = log(sdlogySS)
    m[:Gini_X_star]               = log(GiniXSS) # TODO: figure out exactly what this X is
    m[:sd_log_X_star]             = log(sdlogxSS)
    m[:P90_minus_P10_C_star]      = log(P9010CSS)
    m[:Gini_C_star]               = log(GiniCSS)
    m[:sd_log_C_star]             = log(sdlogCSS)
    m[:P10_C_star]                = log(P10CSS)
    m[:P50_C_star]                = log(P50CSS)
    m[:P90_C_star]                = log(P90CSS)

    # Functional/distributional variables
    m[:distr_star]          = distrSS    # note the distributional variables aren't logged
    m[:marginal_pdf_m_star] = distr_m_SS
    m[:marginal_pdf_k_star] = distr_k_SS
    m[:marginal_pdf_y_star] = distr_y_SS
    m[:Vm_star]             = VmSS       # note that VmSS and VkSS are both already logged
    m[:Vk_star]             = VkSS

    # ------------------------------------------------------------------------------
    ## STEP 2: Dimensionality reduction
    # ------------------------------------------------------------------------------
    # 2 a.) Discrete cosine transformation of marginal value functions
    # ------------------------------------------------------------------------------
    ThetaVm             = dct(VmSS)[:]                               # Discrete cosine transformation of marginal liquid asset value
    ind                 = sortperm(abs.(ThetaVm[:]); rev = true)     # Indexes of coefficients sorted by their absolute size
    coeffs              = 1                                          # Container to store the number of retained coefficients

    # Find the important basis functions (discrete cosine) for VmSS (in L2 norm)
    while norm(view(ThetaVm, view(ind, 1:coeffs))) / norm(ThetaVm) < 1. - get_setting(m, :dct_energy_loss)
            coeffs     += 1                                          # add retained coefficients until only some share of energy is lost
    end
    compressionIndexesVm = ind[1:coeffs]                             # store indexes of retained coefficients

    ThetaVk             = dct(VkSS)[:]                               # Discrete cosine transformation of marginal illiquid asset value
    ind                 = sortperm(abs.(ThetaVk[:]); rev = true)     # Indexes of coefficients sorted by their absolute size
    coeffs              = 1                                          # Container to store the number of retained coefficients

    # Find the important basis functions (discrete cosine) for VkSS
    while norm(view(ThetaVk, view(ind, 1:coeffs))) / norm(ThetaVk) < 1. - get_setting(m, :dct_energy_loss)
            coeffs     += 1                                          # add retained coefficients until only some share of energy is lost
    end
    compressionIndexesVk = ind[1:coeffs]                             # store indexes of retained coefficients

    distr_LOL           = view(distrSS, 1:nm-1, 1:nk-1, 1:ny-1)      # Leave out last entry of histogramm (b/c it integrates to 1)
    ThetaD              = dct(distr_LOL)[:]                          # Discrete cosine transformation of Copula
    ind                 = sortperm(abs.(ThetaD[:]); rev = true)      # Indexes of coefficients sorted by their absolute size
    n_copula_coefs      = get_setting(m, :n_copula_dct_coefficients) # keep n_copula_coefs coefficients, but
    compressionIndexesD = ind[2:2+n_copula_coefs]                    # leave out index no. 1 as this shifts the constant

    compressionIndexes  = Array{Array{Int, 1}, 1}(undef, 3)          # Container to store all retained coefficients in one array
    compressionIndexes[1] = compressionIndexesVm
    compressionIndexes[2] = compressionIndexesVk
    compressionIndexes[3] = compressionIndexesD

    # Store reduction parameters (coefficients go as SteadyStateParameterGrid, indices go as settings)
    m[:dct_Vm_star]     = ThetaVm
    m[:dct_Vk_star]     = ThetaVk
    m[:dct_copula_star] = ThetaD

    DSGE.update_compression_indices!(m, [:Vm, :Vk, :copula],
                                     compressionIndexesVm, compressionIndexesVk, compressionIndexesD)

    # TODO: move this step to the indices/dimensions update (setting is n_states)
    # add to no. of states the coefficients that perturb the copula
    # @set! n_par.nstates = n_par.ny + n_par.nk + n_par.nm + n_par.naggrstates - 3 + length(compressionIndexes[3])

    # ------------------------------------------------------------------------------
    # 2b.) Produce the Copula as an interpolant on the distribution function
    #      and its marginals
    # ------------------------------------------------------------------------------
    # TODO: use our general CDF and marginal CDF quadrature here rather than this implementation
    # CDF_SS              = zeros(nm + 1, nk + 1, ny + 1) # Produce CDF of asset-income distribution (container here)
    CDF_SS                    = Array{Float64}(undef, nm + 1, nk + 1, ny + 1) # Produce CDF of asset-income distribution (container here)
    CDF_SS[1, :, :]          .= 0.
    CDF_SS[:, 1, :]          .= 0.
    CDF_SS[:, :, 1]          .= 0.
    CDF_SS[2:end,2:end,2:end] = cumsum(cumsum(cumsum(distrSS,dims=1),dims=2),dims=3) # Calculate CDF from PDF
    CDF_m                     = cumsum([0.0; distr_m_SS[:]])          # Marginal distribution (cdf) of liquid assets
    CDF_k                     = cumsum([0.0; distr_k_SS[:]])          # Marginal distribution (cdf) of illiquid assets
    CDF_y                     = cumsum([0.0; distr_y_SS[:]])          # Marginal distribution (cdf) of income

    # TODO: create notion of "steady-state function parameter" and store this function there b/c we will need it for MCMC/SMC
    Copula(x::Vector, y::Vector, z::Vector) =
        mylinearinterpolate3(CDF_m, CDF_k, CDF_y, CDF_SS, x, y, z) # Define Copula as a function (used only if not perturbed)

    # Store reduction parameters (cdfs go as SteadyStateParameterGrid, copula go as settings)
    m[:marginal_cdf_m_star] = CDF_m
    m[:marginal_cdf_k_star] = CDF_k
    m[:marginal_cdf_y_star] = CDF_y

    m <= Setting(:copula, Copula)
    # ------------------------------------------------------------------------------

#     @include "../3_Model/input_aggregate_steady_state.jl"
    aggregate_steadystate!(m)
    # write to XSS vector
#    @writeXSS

    # produce indexes to access XSS etc.

    # TODO: check the contents of these indices and then create them
    # store compressionIndexes in some sort of object or Setting or field
    #= indexes               = produce_indexes(n_par, compressionIndexesVm, compressionIndexesVk, compressionIndexesD)
    indexes_aggr          = produce_indexes_aggr(n_par)
    ntotal                = indexes.profits # Convention: profits is the last control in the list of control variables
    @set! n_par.ntotal    = ntotal
    @set! n_par.ncontrols = length(compressionIndexes[1]) + length(compressionIndexes[2]) + n_par.naggrcontrols
    @set! n_par.LOMstate_save = zeros(n_par.nstates, n_par.nstates)
    @set! n_par.State2Control_save = zeros(n_par.ncontrols, n_par.nstates)=#

#=    return XSS, XSSaggr, indexes, indexes_aggr, compressionIndexes, Copula, #=
            =# CDF_SS, CDF_m, CDF_k, CDF_y, distrSS=#

m
end
