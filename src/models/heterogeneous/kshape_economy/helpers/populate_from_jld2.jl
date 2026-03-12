using JLD2

function populate_from_jld2(m::mBBQ, out::JLD2)
    
    #cop = JLD2.jldread(JLD2 FILE)
    m <= Setting(:Γ, cop["Gamma_state"], "Gamma matrix")
    m <= Setting(:copula, cop["Copula"], "Copula")


    Gamma_control = cop["Gamma_control"]
    distrSS      = cop["distrSS"]

    #Save marginal distributions forliquid + illiquid + income
    m[:marginal_cdf_b_star] = cop["CDF_b_SS"]
    m[:marginal_cdf_a_star] = cop["CDF_a_SS"]
    m[:marginal_cdf_se_star] = cop["CDF_se_SS"]
    m[:marginal_pdf_b_star] = cop["distr_b_SS"]
    m[:marginal_pdf_a_star] = cop["distr_a_SS"]
    m[:marginal_pdf_se_star] = cop["distr_se_SS"]

    # Key dimensions
    oc          = Int(grid["oc"])
    numstates   = Int(grid["numstates"])
    nb_cop = Int(grid["nb_copula"])
    na_cop = Int(grid["na_copula"])
    nse_cop = Int(grid["nse_copula"])
    nPoly   = cop["nPoly"]

    # Settings
    m <= Setting(:nb_copula, nb_cop, "Compression Index for Liquid Assets")
    m <= Setting(:na_copula, na_cop, "Compression Index for Illiquid Assets")
    m <= Setting(:nse_copula, nse_cop "Compression Index for Income")
    m <= Setting(:nPoly, nPoly, "Degree of Chebyshev Polynomials")
    m <= Setting(:ny, oc, "Total number of Chebyshev polynomial coefficients across 
                 all distributions" )

    # 4. DCT matrices (reconstructed from dimensions)
    m <= Setting(:DC, cop["DC"])
    m <= Setting(:IDC, cop["IDC"])
    m <= Setting(:DCD, cop["DCD"])
    m <= Setting(:IDCD, cop["IDCD"])

    #Copula compression indices
    comprCOP = Vector{Int}(grid["compressionIndexesCOP"])
    get_setting(m, :dct_compression_coefficients)[:copula] = comprCOP
    get_setting(m, :n_copula_dct_coefficients) = nCOP
    # Steady-state NamedTuple (reuses build_ss from index2.jl)
    ss = build_ss(param, SS_stats)
    m <= Setting(:ss, ss)

    #  Build index dicts (reuses build_indices from index2.jl, after fixes)
    grid_sym = Dict{Symbol,Any}(Symbol(k) => v for (k,v) in grid)
    state_id, control_id = build_indices(grid_sym, length(StateSS), length(ControlSS))

    for (k, v) in m.endogenous_states
     m.equilibrium_conditions[Symbol("eq_" * string(k))] = v
    end


    #prime_and_noprime settings -- NEEDS TO BE TESTED
    full_id = OrderedDict{Symbol,UnitRange{Int}}()
    for (k, v) in merge(state_id, control_id)
     full_id[k] = v isa Int ? (v:v) : v
    end
    m <= Setting(:prime_and_noprime_indices, full_id)

    agg_id = OrderedDict{Symbol,Int}()
    for sym in keys(m.aggregate_endogenous_states)
     agg_id[sym] = m.aggregate_endogenous_states[sym]
    end
    m <= Setting(:prime_and_noprime_aggregate_indices, agg_id)


end
