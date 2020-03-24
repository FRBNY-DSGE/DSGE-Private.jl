#TODO: implement saving draws
abstract type HMCDSGEProblem{TX, TM} end # abstract type for creating HMC-compatible model types

macro create_hmc_problem(structname, m, data)
    structname = Symbol(structname) # name of concrete subtype of HMCDSGEProblem
    expr = quote
        # Create concrete subtype
        mutable struct $(esc(structname)){TX, TM} <: HMCDSGEProblem{TX, TM}
            DATA::TX
            model::TM
        end

        # Add information for the HMC sampler
        function LogDensityProblems.capabilities(::$(esc(structname)))
            LogDensityProblems.LogDensityOrder{1}() # can do gradient
        end

        LogDensityProblems.dimension(::$(esc(structname))) = count(get_unfixed_parameter_indices($m))

        function LogDensityProblems.logdensity_and_gradient(problem::$(esc(structname)), x)
            DATA  = problem.DATA
            model = problem.model

            logdens = hmc_likelihood(model, x, DATA)
            grad    = hmc_gradient(model, x, DATA; matrices = matrices,
                                   ordered = ordered)

            return logdens, grad
        end

        $(esc(structname))($(esc(data)), $(esc(m))) # Create HMC problem
    end
    return expr
end

function hmc(problem::MD; matrices::Vector{Symbol} = [:TTT, :RRR, :ZZ, :DD],
             ordered::Bool = false, output_vars::Vector{Symbol} = [:posterior, :hmc_results],
             verbose::Symbol = :low, kwargs...) where {MD <: HMCDSGEProblem}

    # Call DynamicHMC!
    hmc_results = mcmc_wih_warmup(Random.GLOBAL_RNG, problem, get_setting(m, :n_hmc_iterations),
                                  reporter = :verbose == :high ? default_reporter() : NoProgressReport(),
                                  kwargs...)
    posterior_vals = hcat(map(vals -> map((p, y) -> transform_to_model_space(p, y),
                                          m.parameters[get_unfixed_parameter_indices(m)], vals), hmc_results.chain)...)

    if (:posterior in output_vars) && (:hmc_results in output_vars)
        return posterior_vals, hmc_results
    elseif :posterior in output_vars
        return posterior_vals
    elseif :hmc_results in output_vars
        return hmc_results
    else
        error("Invalid output_vars entry. Can only use `:posterior` and `:hmc_results`")
    end
end
