"""
```
measurement(m::PoolModel{T}) where {T<:AbstractFloat}
```

Assign measurement equation

```
p(y_t) = λ_t * p_1 + (1 - λ_t) * p_2
```

where

```
λ_t = weight assigned to model 1
p_1 = predictive density according to model 1
p_2 = predictive density according to model 2
```
"""
function measurement(m::PoolModel{T}, data::Vector{T}) where {T<:AbstractFloat}
    @inline Ψ(x::Vector{T}) = dot(data, x)
    F_u = DiscreteUniform(0,0)
    return Ψ, F_u
end
