"""
```
transition(x::Vector{T}, params::Dict{Symbol,Float)
transition(x::Float64
```
"""
function transition(x::T, params::Dict{Symbol,T}) where {T<:AbstractFloat}
    ϵ = rand(Normal(), 1)
    x1 = (1 - params[:ρ]) * params[:μ] + params[:ρ] * x + sqrt(1 - params[:ρ]^2) *
         params[:σ] * ϵ
    return cdf(Normal(), x1)
end
# function transition(x::Vector{T}, params::Dict{Symbol,T}) where {T<:AbstractFloat}
#     ϵ = rand(Normal(), length(x))
#     x1 = @. (1 - params[:ρ]) * params[:μ] + params[:ρ] * x + sqrt(1 - params[:ρ]^2) *
#          params[:σ] * ϵ
#     return cdf.(Normal(), x1)
# end
