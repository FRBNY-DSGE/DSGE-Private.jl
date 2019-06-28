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
                     RRR::Matrix{T}, CCC::Vector{T}) where {T<:AbstractFloat}
    return getloglh(m,t) # return vector of loglikelihood at period t, may need to change this
end
