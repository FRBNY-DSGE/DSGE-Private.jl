"""
```
measurement(m::PoolModel{T}, TTT::Matrix{T},
            RRR::Matrix{T}, CCC::Vector{T}) where {T<:AbstractFloat}
```
Assign measurement equation

```
y_t = ZZ * s_t
"""
function measurement(m::PoolModel{T}, TTT::Matrix{T},
                     RRR::Matrix{T}, CCC::Vector{T}}) where {T<:AbstractFloat}
    ZZ = zeros(n_states(m))
    return s_t * loglhv[1] + (1 - s_t) * loglhv[2]
end
