"""
```
updateV(EVk::Array,
        c_a_star::Array,
        c_n_star::Array,
        m_n_star::Array,
        rk::Real, q::Real,
        θ::NamedTuple,
        m_grid::AbstractVector{T1},
        Π::Array;
        parallel::Bool = false) where {T1 <: Real}
```
updates value functions after running `EGM_policyupdate`
to compute consumption and savings policies (`c_a_star`, `c_n_star`, `m_n_star`)
implied by expected continuation values (`EVk` and `EVm`),
given today's return (`rk`) on and price (`q`) of illiquid capital.
The input `θ` maps the names of economic parameters to values, `m_grid` is
the liquid asset grid, and `Π` is the
transition matrix over idiosyncratic income states.
"""
function updateV(EVk::Array,
                 c_a_star::Array,
                 c_n_star::Array,
                 m_n_star::Array,
                 rk::Real, q::Real,
                 θ::NamedTuple,
                 m_grid::AbstractVector{T1},
                 Π::Array;
                 parallel::Bool = false) where {T1 <: Real}

    # Setup
    β::Float64 = θ[:β]
    n = size(c_n_star)

    #----------------------------------------------------------------------------
    ## Update Marginal Value Bonds
    #----------------------------------------------------------------------------
    mutil_c_n = _bbl_mutil(c_n_star, θ[:ξ])                         # marginal utility at consumption policy no adjustment
    mutil_c_a = _bbl_mutil(c_a_star, θ[:ξ])                         # marginal utility at consumption policy adjustment

    # Compute expected marginal utility at consumption policy (w &w/o adjustment)
    # Some special handling here to avoid an allocation that would be made if we did
    Vm  = mutil_c_n # Vm is just pointing to the same array as mutil_c_n now
    Vm .*= (1. - θ[:λ])
    Vm .+= θ[:λ] * mutil_c_a

    #----------------------------------------------------------------------------
    ## Update marginal Value of Capital
    ## i.e. linear interpolate expected future marginal value of capital using savings policy
    ## Then form expectations.
    #----------------------------------------------------------------------------

    # Use savings policy implied by m_n_star
    Vk = Array{eltype(EVk), 3}(undef, n)                       # Initialize Vk-container
    if parallel # multi-threading b/c this loop is quick, and we don't expect to use too many threads
        @inbounds @views begin
            Threads.@threads for kj in CartesianIndices((n[2], n[3]))
                k, j        = kj[1], kj[2]
                Vk[:, k, j] = mylinearinterpolate(m_grid, EVk[:, k, j], m_n_star[:, k, j]) # evaluate marginal value at policy
            end
        end
    else
        @inbounds @views begin
            for j::Int = 1:n[3]
                for k::Int = 1:n[2]
                    Vk[:, k, j] = mylinearinterpolate(m_grid, EVk[:, k, j], m_n_star[:, k, j]) # evaluate marginal value at policy
                end
            end
        end
    end

    # Form expectations to get expected marginal utility at consumption policy (w &w/o adjustment)
    Vk .*= (1. - θ[:λ]) * β # written this way to avoid allocations
    Vk .+= rk * Vm + (θ[:λ] * q) * mutil_c_a

    return Vk, Vm
end
