function persistent_skill_process(sH_over_sL::AbstractFloat, pLH::AbstractFloat,
                                  pHL::AbstractFloat, ns::Int)
    f1 = [[1-pLH pLH];[pHL 1-pHL]] # f1[i,j] is prob of going from i to j
    ss_skill_distr = [pHL/(pLH+pHL); pLH/(pLH+pHL)]
    slo    = 1.0 / (ss_skill_distr'*[1;sH_over_sL])
    sgrid  = slo*[1;sH_over_sL]
    sscale = sgrid[2] - sgrid[1]
    swts   = (sscale/ns)*ones(ns) # Quadrature weights
    f      = f1 ./ repeat(swts',ns,1)
    return f, sgrid, swts, sscale
end

"""
```
function interp_one(xpts::T, ypts::T, x::T) where {T<:Vector{Float64}}
```
Mimics behavior of interp1 in MATLAB. Takes set {(x_i, y_i)}, linearly interpolates between each (x_i, y_i) and (x_{i+1}, y_{i+1}), and then returns vector y for interpolated points in x.
"""
function interp_one(xpts::T1, ypts::T2,
                    x::T3) where {T1 <: AbstractVector{<: Real}, T2 <: AbstractVector{<: Real}, T3 <: AbstractVector{<: Real}}
    @assert length(xpts) == length(ypts)
    intf = extrapolate(interpolate((xpts,), ypts, Gridded(Linear())), Line())
    y = [intf(v) for v in x]
    return y
end

"""
```
function histc(points, grid)
```
Mimics behavior of histc function in MATLAB.
"""
function histc(points, grid) # TODO: see if I can avoid declaring these variables global, e.g. found probably doesn't need to be
    N      = size(grid, 1)
    ib_pol = zeros(Int64, size(points))
    points_copy = deepcopy(points)
    for i in eachindex(points)
        p = points[i]
        global min_ind, max_ind = 1, N
        global found = false

        if p > grid[max_ind]
           # @warn "Point not in grid"
            found = true
            ib_pol[i] = max_ind - 1
            points_copy[i] = grid[ib_pol[i]]
        end
        if p < grid[min_ind]
            ib_pol[i] = min_ind
            points_copy[i] = grid[ib_pol[i]]
            found = true
        end

        while !found
            ind = Int(floor((min_ind + max_ind) / 2))
            if p >= grid[ind]
                if p < grid[ind+1]
                    found = true
                    ib_pol[i] = ind
                elseif p == grid[ind+1]
                    found = true
                    ib_pol[i] = ind+1
                else
                    global min_ind = ind
                    if min_ind == max_ind - 1
                        found = true
                        ib_pol[i] = ind
                    end
                end
            else
                global max_ind = ind
            end
        end
    end
 #   ib_pol[ib_pol .== length(grid)] .= length(grid)-1
    wei = 1 .- ((points_copy - grid[ib_pol]) ./
        (grid[ib_pol.+1] - grid[ib_pol]))

    return ib_pol, wei
end

"""
```
generate_us_and_es(ni, ne)
```

There is no need to recall this function, unless one wants to undo the seeding
in all past saved output.

The us are draws from U[0, 1] used to construct nodes for the skill distribution.

The es are the grid nodes for the exogenous i.i.d productivity shock.
"""
function generate_us_and_es(ni, ne)
    us = rand(ni, 8)
    ue = rand(ni, 8)

    egrid  = collect(range(0., stop = 2., length = ne))
    eprob  = [2*mollifier_hetdsgegovdebt(egrid[i], 2., 0.) / ne for i=1:ne]
    eprob /= sum(eprob)

    ecdf = cumsum(eprob)
    eave = 0.5 * egrid[1:ne-1] + 0.5 * egrid[2:ne]
    es   = esample(ue, egrid, ecdf, ni, ne)

    return us, es
    #=
    JLD2.jldopen("$HETDSGEGOVDEBT/reference/us_es.jld2", true, true, true, IOStream) do file
        file["us"] = us
        file["es"] = es
    end
    =#
end

"""
```
CashOnHandError <: Exception
```
A `CashOnHandError` is thrown when:

1. The implied assets today `b` are all negative during the Euler iteration.
2. The consumption policy at the smallest `b` value is smaller than the constrained consumption policy.
3. Some assets today `b` for constrained agents are negative.
4. Consumption is not always less than cash on hand after mapping the consumption policy from `(b, s, e)` to `(a, s)`.
"""
mutable struct CashOnHandError <: Exception
    msg::String
end
CashOnHandError() = CashOnHandError("Consumption policy is not consistent with cash on hand.")
Base.showerror(io::IO, ex::CashOnHandError) = print(io, ex.msg)
