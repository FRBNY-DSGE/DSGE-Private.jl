"""
```
Compute measurement equation
```
"""

function measurement(s_t::T, loglhv::Vector{T}) where {T<:AbstractFloat}
    return s_t * loglhv[1] + (1 - s_t) * loglhv[2]
end
