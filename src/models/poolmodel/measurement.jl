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
function measurement(m::PoolModel{T}) where {T<:AbstractFloat}
    obs = m.observables
    weight_type = get_setting(m, :weight_type)

    # Assumes λ to be weight on the first model
    F_u = DiscreteUniform(0,0)
    if weight_type == :dynamic
        Ψ_dynamic_pm(x::Vector{Float64}, data::Vector{Float64}) = dot(data, x)
        return Ψ_dynamic_pm, F_u
    elseif weight_type == :equal
        Ψ_equal_pm(x::Vector{Float64}, data::Vector{Float64}) = dot(data, [0.5; 0.5])
        return Ψ_equal_pm, F_u
    elseif weight_type == :static
        Ψ_static_pm(x::Vector{Float64}, data::Vector{Float64}) = dot(data, x)
        return Ψ_static_pm, F_u
    end
end
