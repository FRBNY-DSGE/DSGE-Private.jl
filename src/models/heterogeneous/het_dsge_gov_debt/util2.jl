using Interpolations
# Find consumption for current assets b1 and next assets b2
#=
function find_cl(c, j, b1, b2, r)
    global theta z fac gameta

    n = maximum(0, 1 - fac[j] * c .^ gameta)
    return b1 - b2 / (1+r) - c + theta[j]*n + z[j]
end
=#
function max_select(val::Float64, v::Vector{Float64})
    for i=1:length(v)
        if v[i] < val
            v[i] = val
        end
    end
    return v
end

function max_select(val::Float64, arr::Matrix{Float64})
    for i=1:size(arr,1), j=1:size(arr,2)
        if arr[i,j] < val
            arr[i,j] = val
        end
    end
    return arr
end


function interp_one(xpts, ypts, x)
    intf = extrapolate(interpolate((xpts,), ypts, Gridded(Linear())), Line())
    y = [intf(v) for v in x]
    return y
end

function histc(points, grid)
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
