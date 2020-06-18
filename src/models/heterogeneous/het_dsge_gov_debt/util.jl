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

function cash_grid(sgrid::AbstractArray, ω::AbstractFloat, H::AbstractFloat,
                   r::AbstractFloat, η::AbstractFloat, γ::AbstractFloat,
                   T::AbstractFloat, zlo::AbstractFloat, na::Int)
    smin = minimum(sgrid)*zlo                                   # lowest possible skill
    xlo_ss = ω*smin*H - (1+r)*η*exp(-γ) + T + sgrid[1]*ω*H*0.05 # lowest SS possible cash on hand

    xlo = xlo_ss                        # lower bound on cash on hand - could be < xlo_ss
    xhi = max(xlo*2, xlo + 12.0)         # upper bound on cash on hand
    xscale = (xhi-xlo)                  # size of w grids

    # Make grids
    xgrid = collect(range(xlo,stop = xhi, length = na)) # Evenly spaced grid
    xwts  = (xscale/na)*ones(na)          # Quadrature weights
    return xgrid, xwts, xlo, xhi, xscale
end

"""
```
function interp_one(xpts::T, ypts::T, x::T) where {T<:Vector{Float64}}
```
Mimics behavior of interp1 in MATLAB. Takes set {(x_i, y_i)}, linearly interpolates between each (x_i, y_i) and (x_{i+1}, y_{i+1}), and then returns vector y for interpolated points in x.
"""
function interp_one(xpts::T, ypts::T, x::T) where {T<:Vector{Float64}}
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
function histc(points, grid)
    I, J, K = size(points)
    N    = length(grid[:, :, 1])

    ib_pol = zeros(Int64, I, J, K)
    for i=1:I, j=1:J, k=1:K
        @show i, j, k
        p = points[i,j,k]
        global min, max = 1, N
        global found = false

        if p > grid[max]
            found = true
            ib_pol[i,j,k] = max
        end

        while !found
            ind = Int(floor((min + max) / 2))
            if p >= grid[ind]
                if p < grid[ind+1]
                    found = true
                    ib_pol[i,j,k] = ind
                elseif p == grid[ind+1]
                    found = true
                    ib_pol[i,j,k] = ind+1
                else
                    global min = ind
                    if min == max - 1
                        found = true
                        ib_pol[i,j,k] = ind
                    end
                end
            else
                global max = ind
            end
        end
    end
    wei = (points - grid[ib_pol]) ./
        (points[ib_pol.+1] - grid[ib_pol])

    return ib_pol, wei
end


function histc_2d(points, grid)
    I, J = size(points)
    N    = size(grid, 1)

    ib_pol = zeros(Int64, I, J)
    for i=1:I, j=1:J
        p = points[i,j]
        global min, max = 1, N
        global found = false

        if p > grid[max]
            found = true
            ib_pol[i,j] = max
        end

        while !found
            ind = Int(floor((min + max) / 2))
            if p >= grid[ind]
                if p < grid[ind+1]
                    found = true
                    ib_pol[i,j] = ind
                elseif p == grid[ind+1]
                    found = true
                    ib_pol[i,j] = ind+1
                else
                    global min = ind
                    if min == max - 1
                        found = true
                        ib_pol[i,j] = ind
                    end
                end
            else
                global max = ind
            end
        end
    end
    wei = (points - grid[ib_pol]) ./
        (points[ib_pol.+1] - grid[ib_pol])

    return ib_pol, wei
end

function histc_2d_reca(points, grid)
    I, J = size(points)
    N    = length(grid)

    ib_pol = zeros(Int64, I, J)
    for i=1:I, j=1:J
        p = points[i,j]
        global min, max = 1, N
        global found = false

        if p > grid[max]
            found = true
            ib_pol[i,j] = max
        end

        while !found
            ind = Int(floor((min + max) / 2))
            if p >= grid[ind]
                if p < grid[ind+1]
                    found = true
                    ib_pol[i,j] = ind
                elseif p == grid[ind+1]
                    found = true
                    ib_pol[i,j] = ind+1
                else
                    global min = ind
                    if min == max - 1
                        found = true
                        ib_pol[i,j] = ind
                    end
                end
            else
                global max = ind
            end
        end
    end
    wei = (points - grid[ib_pol]) ./
        (points[ib_pol.+1] - grid[ib_pol])

    return ib_pol, wei
end

"""
```
generate_us_and_zs(ni, nz)
```

There is no need to recall this function, unless one wants to undo the seeding
in all past saved output.
"""
function generate_us_and_zs(ni, nz)
    us = rand(ni, 8)
    uz = rand(ni, 8)

    zgrid  = collect(range(0., stop = 2., length = nz))
    zprob  = [2*mollifier_hetdsgegovdebt(zgrid[i], 2., 0.) / nz for i=1:nz]
    zprob /= sum(zprob)

    zcdf = cumsum(zprob)
    zave = 0.5 * zgrid[1:nz-1] + 0.5 * zgrid[2:nz]
    zs   = zsample(uz, zgrid, zcdf, ni, nz)

    return us, zs
    #=
    JLD2.jldopen("$HETDSGEGOVDEBT/reference/us_zs.jld2", true, true, true, IOStream) do file
        file["us"] = us
        file["zs"] = zs
    end
    =#
end
