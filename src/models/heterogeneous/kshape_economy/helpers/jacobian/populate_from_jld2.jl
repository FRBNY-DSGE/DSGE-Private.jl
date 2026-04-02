function populate_from_jld2!(m::mBBQ, out::String)
    m <= Setting(:ss_file_path, out)
    return populate_from_jld2!(m)
end

function populate_from_jld2!(m::mBBQ)
    grids = m.grids
    ref = nothing
    try
        file = get_setting(m, :ss_file_path)
        ref = JLD2.jldopen(file, "r")
    catch e
        error("There is no jld2 file associated with this model")
    end

    #Xss, Yss
    m <= Setting(:Xss, ref["Xss"], "Xss")
    m <= Setting(:Yss, ref["Yss"], "Yss")

    #Storing Γ matrices and Chebyshev Polynomials
    m <= Setting(:Γ_state, ref["Gamma_state"], "Gamma State matrix")
    m <= Setting(:Γ_control, ref["Gamma_control"], "Gamma Control matrix")
    m <= Setting(:InvGamma, ref["InvGamma"], " ")
    m <= Setting(:Poly, ref["Poly"], "") #add description
    m <= Setting(:Γ2, ref["Gamma2"], " ")

    # Should I save this?
    distrSS      = ref["distrSS"]

    #Save marginal distributions for liquid + illiquid + income
    m[:marginal_cdf_b_star] = ref["CDF_b_SS"]
    m[:marginal_cdf_a_star] = ref["CDF_a_SS"]
    m[:marginal_cdf_se_star] = ref["CDF_se_SS"]
    m[:marginal_pdf_b_star] = ref["distr_b_SS"]
    m[:marginal_pdf_a_star] = ref["distr_a_SS"]
    m[:marginal_pdf_se_star] = ref["distr_se_SS"]



    # Populating the grid
    ref_grid = ref["grid"]

    #Saving in grid
    grids[:nb_copula] = Float64(ref_grid["nb_copula"])
    grids[:na_copula] = Float64(ref_grid["na_copula"])
    grids[:nse_copula] = Float64(ref_grid["nse_copula"])

    grids[:copula_marginal_b] = ref_grid["copula_marginal_se"]
    grids[:copula_marginal_a] = ref_grid["copula_marginal_se"]
    grids[:copula_marginal_se] = ref_grid["copula_marginal_se"]

    # Settings
    m <= Setting(:nPoly, Int(ref["nPoly"]), "Number of Chebyshev basis functions in the 
                 sparse polynomial approximation of value functions")
    m <= Setting(:nFullCtrl, Int(ref["nFullCtrl"]), "Number of controls on the full asset-income grid 
                 (3 policy functions × full grid + aggregate controls)")
    m <= Setting(:nRedCtrl, Int(ref["nRedCtrl"]), "Number of controls compressed via Chebyshev basis 
                 (3 policy functions × nPoly coefficients + aggregate controls)")
    m <= Setting(:nFullMarg, Int(ref["nFullMarg"]), "Total number of marginal distribution 
                 bins (nb + na + nse)")
    m <= Setting(:nRedMarg, Int(ref["nRedMarg"]), "Reduced marginal distribution bins after removing 
                 3 redundant (sum-to-one) degrees of freedom")
    m <= Setting(:nRedStates, Int(ref["nRedStates"]), "Total reduced state vector dimension
                 (compressed marginals + copula coefficients + aggregate exogenous states)")
    m <= Setting(:oc, Int(ref_grid["oc"]), "Number of aggregate control variables (non-value-function controls)")
    m <= Setting(:oc_agg, Int(ref_grid["oc_agg"]), "Number of purely aggregate control variables, 
                 excluding summary statistics (Gini, wage shares, etc.)")
    m <= Setting(:os_process, Int(ref_grid["os_process"]), "Number of exogenous process state 
                 variables (AR(1) shocks and their lags)")
    m <= Setting(:os_agg, Int(ref_grid["os_agg"]), "Number of aggregate endogenous state variables 
                 (lagged prices,rates, etc.), excluding exogenous shock processes")
    m <= Setting(:numstates, Int(ref_grid["numstates"]), "Total reduced state vector 
                 dimension (compressed marginals + copula coefficients + 
                 aggregate exogenous states including shocks)")
    m <= Setting(:numstates_shocks, Int(ref_grid["numstates_shocks"]), "Number of exogenous shock processes")
    m <= Setting(:numstates_endo, Int(ref_grid["numstates_endo"]), "Number of endogenous state variables 
                 (numstates minus shock processes)")
    m <= Setting(:numcontrols, Int(ref_grid["numcontrols"]), "Number of compressed control variables (
                 3 × nPoly value function coefficients + aggregate controls)")
    m <= Setting(:num_endo, Int(ref_grid["num_endo"]), "Total number of endogenous variables 
                 (endogenous states + controls), i.e. size of the linearized system excluding shocks")

    # DCT matrices (reconstructed from dimensions)
    m <= Setting(:DC, ref["DC"])
    m <= Setting(:IDC, ref["IDC"])
    m <= Setting(:DCD, ref["DCD"])
    m <= Setting(:IDCD, ref["IDCD"])

    

    # Copula compression indices
    m <= Setting(:nb_copula, Int(ref_grid["nb_copula"]), "Number of nodes in the liquid asset 
                 dimension of the copula grid")
    m <= Setting(:na_copula, Int(ref_grid["na_copula"]), "Number of nodes in the illiquid asset 
                 dimension of the copula grid")
    m <= Setting(:nse_copula, Int(ref_grid["nse_copula"]), "Number of nodes in the income 
                 dimension of the copula grid")
    m <= Setting(:nCOP, Int(ref_grid["nCOP"]), "Number of retained copula Chebyshev coefficients after sparse
                 truncation (entries satisfying I+J+K <= reduc_copula, excluding cross-marginal terms)")
    m <= Setting(:compressionIndexesCOP, Vector{Int}(ref_grid["compressionIndexesCOP"]), "Linear indices into 
                 the  nb_copula × na_copula × nse_copula DCT coefficient array identifying the retained sparse 
                 copula coefficients")   
end
