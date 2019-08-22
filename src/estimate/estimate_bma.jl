"""
```
estimate_bma(m, data, prior; return_output = false)
```

Estimate a Bayesian Model Average.

### Arguments:
- `m::PoolModel`: PoolModel object

### Optional Arguments:
- `data`: well-formed data as `Matrix` or `DataFrame`. If this is not provided, the `load_data` routine will be executed.
- `prior::Float64`: prior probability between the two models

### Keyword Arguments:
- `return_output::Bool`: option to return output. If false, `nothing` is returned.

"""
function estimate_bma(m::PoolModel, df::DataFrame, prior::Float64 = 0.5;
                      return_output::Bool = false)
    return estimate_bma(m, df_to_matrix(m, df))
end

function estimate_bma(m::PoolModel, data::Matrix{Float64} = Matrix{Float64}(0,0),
                      prior::Float64 = 0.5; return_output::Bool = false)

    # Compute λ weights
    λ = zeros(size(data,2))
    λ[1] = prior * data[1,1] / (prior * data[1,1] + (1 - prior) * data[2,1])
    for t in 2:size(data,2)
        λ[t] = λ[t-1] * data[1,t] / (λ[t-1] * data[1,t] + (1 - λ[t-1]) * data[2,t])
    end

    # Save weights
    h5open(rawpath(m, "estimate", "bmasave.h5", filestring_addl), "w") do file
        write(file, "bmaparams", λ)
    end

    if return_output
        prob_vec = λ .* vec(data[1,:]) + (1 .- λ) .* vec(data[2,:])
        return λ, prob_vec
    end
end
