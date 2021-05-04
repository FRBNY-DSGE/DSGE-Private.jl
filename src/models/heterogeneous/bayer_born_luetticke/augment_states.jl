function augment_states(m::BayerBornLuetticke{T}, TTT::AbstractMatrix{T}, RRR::AbstractMatrix{T},
                        CCC::AbstractVector{T}) where {T <: Real}

    # Do nothing for now. Later, add augment states and measurement error
    return TTT, RRR, CCC
end
