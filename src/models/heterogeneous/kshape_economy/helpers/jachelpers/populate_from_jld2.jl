function populate_from_jld2!(m::mBBQ, out::String)
    m <= Setting(:ss_file_path, out)
    return populate_from_jld2!(m)
end

function populate_from_jld2!(m::mBBQ)
    try
        file = get_setting(m, :ss_file_path)
        ref = JLD2.jldopen(file, "r")
    catch e
        error("There is no jld2 file associated with this model")
    end

    m <= Setting(:Γ, ref["Gamma_state"], "Gamma matrix")
    m <= Setting(:copula, ref["Copula"], "Copula")


    Gamma_control = ref["Gamma_control"]
    distrSS      = ref["distrSS"]

    #Save marginal distributions forliquid + illiquid + income
    m[:marginal_cdf_b_star] = ref["CDF_b_SS"]
    m[:marginal_cdf_a_star] = ref["CDF_a_SS"]
    m[:marginal_cdf_se_star] = ref["CDF_se_SS"]
    m[:marginal_pdf_b_star] = ref["distr_b_SS"]
    m[:marginal_pdf_a_star] = ref["distr_a_SS"]
    m[:marginal_pdf_se_star] = ref["distr_se_SS"]

    # Key dimensions
    oc          = Int(grid["oc"])
    numstates   = Int(grid["numstates"])
    nb_cop = Int(grid["nb_copula"])
    na_cop = Int(grid["na_copula"])
    nse_cop = Int(grid["nse_copula"])
    nPoly   = ref["nPoly"]

    # Settings
    m <= Setting(:nb_copula, nb_cop, "Compression Index for Liquid Assets")
    m <= Setting(:na_copula, na_cop, "Compression Index for Illiquid Assets")
    m <= Setting(:nse_copula, nse_cop, "Compression Index for Income")
    m <= Setting(:nPoly, nPoly, "Degree of Chebyshev Polynomials")
    m <= Setting(:ny, oc, "Total number of Chebyshev polynomial coefficients across 
                 all distributions" )

    # 4. DCT matrices (reconstructed from dimensions)
    m <= Setting(:DC, ref["DC"])
    m <= Setting(:IDC, ref["IDC"])
    m <= Setting(:DCD, ref["DCD"])
    m <= Setting(:IDCD, ref["IDCD"])

    #Copula compression indices
    comprCOP = Vector{Int}(grid["compressionIndexesCOP"])
    get_setting(m, :dct_compression_coefficients)[:copula] = comprCOP
    #get_setting(m, :n_copula_dct_coefficients) = nCOP
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
