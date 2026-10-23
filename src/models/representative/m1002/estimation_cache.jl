# These Model1002 parameters affect shock covariances or measurement loadings,
# but none of the steady-state calculations in m1002.jl. Keep this list explicit:
# a parameter named σ_* can also enter equilibrium conditions (for example σ_c).
const MODEL1002_NO_STEADYSTATE_PARAMETERS = Set{Symbol}((
    :σ_g, :σ_b, :σ_μ, :σ_ztil, :σ_λ_f, :σ_λ_w, :σ_r_m,
    :σ_σ_ω, :σ_μ_e, :σ_γ, :σ_π_star, :σ_lr, :σ_z_p,
    :σ_tfp, :σ_gdpdef, :σ_corepce, :σ_gdp, :σ_gdi,
))

function skip_steadystate_update(m::Model1002, values::AbstractVector{<:Real})
    length(values) == length(m.parameters) || return false
    changed_covariance = false
    for (parameter, value) in zip(m.parameters, values)
        if !isequal(parameter.value, value)
            parameter.key in MODEL1002_NO_STEADYSTATE_PARAMETERS || return false
            changed_covariance = true
        end
    end
    return changed_covariance
end

function skip_steadystate_update(m::Model1002, values::ParameterVector)
    length(values) == length(m.parameters) || return false
    changed_covariance = false
    for (parameter, value) in zip(m.parameters, values)
        if !isequal(parameter.value, value.value)
            parameter.key in MODEL1002_NO_STEADYSTATE_PARAMETERS || return false
            changed_covariance = true
        end
    end
    return changed_covariance
end

# Cache the raw Gensys solution, before regime-specific state augmentation and
# measurement. Matching all five canonical matrices prevents reuse when Ψ changes,
# while a change to QQ alone leaves this entry valid. Exact equality follows the
# hash check so hash collisions cannot change a solution.
struct Model1002GensysEntry
    signature::UInt
    Γ0::Matrix{Float64}
    Γ1::Matrix{Float64}
    C::Vector{Float64}
    Ψ::Matrix{Float64}
    Π::Matrix{Float64}
    div::Float64
    TTT::Matrix{Float64}
    CCC::Vector{Float64}
    RRR::Matrix{Float64}
    eu::Vector{Int}
end

const MODEL1002_GENSYS_CACHE = WeakKeyDict{Any, Vector{Model1002GensysEntry}}()
const MODEL1002_GENSYS_CACHE_LOCK = ReentrantLock()
const MODEL1002_GENSYS_CACHE_CAPACITY = 32

function model1002_gensys_signature(Γ0, Γ1, C, Ψ, Π, div)
    signature = hash(div)
    for coefficients in (Γ0, Γ1, C, Ψ, Π)
        signature = hash(size(coefficients), signature)
        for coefficient in coefficients
            signature = hash(coefficient, signature)
        end
    end
    return signature
end

function gensys_for_model(m::Model1002, Γ0::Matrix{Float64}, Γ1::Matrix{Float64},
                          C::Vector{Float64}, Ψ::Matrix{Float64}, Π::Matrix{Float64},
                          div::Float64; verbose::Symbol = :high)
    signature = model1002_gensys_signature(Γ0, Γ1, C, Ψ, Π, div)
    cached = lock(MODEL1002_GENSYS_CACHE_LOCK) do
        entries = get(MODEL1002_GENSYS_CACHE, m, Model1002GensysEntry[])
        index = findfirst(entry -> entry.signature == signature && entry.div == div &&
                         isequal(entry.Γ0, Γ0) && isequal(entry.Γ1, Γ1) &&
                         isequal(entry.C, C) && isequal(entry.Ψ, Ψ) &&
                         isequal(entry.Π, Π), entries)
        isnothing(index) ? nothing : entries[index]
    end
    if !isnothing(cached)
        return copy(cached.TTT), copy(cached.CCC), copy(cached.RRR), copy(cached.eu)
    end

    TTT, CCC, RRR, eu = gensys(Γ0, Γ1, C, Ψ, Π, div; verbose = verbose)
    if eu == [1, 1]
        entry = Model1002GensysEntry(signature, copy(Γ0), copy(Γ1), copy(C),
                                     copy(Ψ), copy(Π), div, copy(TTT), copy(CCC),
                                     copy(RRR), copy(eu))
        lock(MODEL1002_GENSYS_CACHE_LOCK) do
            entries = get!(MODEL1002_GENSYS_CACHE, m) do
                Model1002GensysEntry[]
            end
            push!(entries, entry)
            length(entries) > MODEL1002_GENSYS_CACHE_CAPACITY && popfirst!(entries)
        end
    end
    return TTT, CCC, RRR, eu
end
