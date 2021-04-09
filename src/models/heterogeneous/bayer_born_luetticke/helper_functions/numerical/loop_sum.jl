"""
```
loop_sum(x::AbstractArray)
```
computes the sum in a loop. This method can be
faster than sum for small to medium problems
(e.g. speed is comprable when `x = rand(10e6)`).

For larger problems or problems where you are summing large numbers,
`sum` will be more accurate because it accounts for rounding error.
"""
@inline function loop_sum(x::AbstractArray)
    out = zero(eltype(x))
    @inbounds @simd for i in eachindex(x)
        out += x[i]
    end
    return out
end
