using Interpolations

"""
    copula_nodes_mass(nx)

Generate equal-mass copula nodes.
Returns a vector of nx linearly spaced values from eps to 1-eps.
"""
function copula_nodes_mass(nx::Int)
    epsu = 1e-6
    return collect(range(epsu, 1 - epsu, length=nx))
end

"""
    enforce_min_spacing(u, dmin)

Enforce minimum spacing between consecutive elements.
Modifies u in place and returns it.
"""
function enforce_min_spacing(u::Vector{Float64}, dmin::Float64)
    u[1] = max(u[1], 1e-9)
    for k in 2:length(u)
        if u[k] <= u[k-1] + dmin
            u[k] = u[k-1] + dmin
        end
    end
    u[end] = min(u[end], 1 - 1e-9)
    return u
end

"""
    copula_nodes_share_safe(distr_i, grid_i, nx; transform="none", alpha=1.0, pinPenultimate=false)

Equal-"share" knots but safe for negatives and heavy tails.

# Arguments
- `distr_i`: Distribution vector
- `grid_i`: Grid vector
- `nx`: Number of nodes (minimum 3)
- `transform`: "none", "pos", or "logshift"
- `alpha`: Blend parameter in [0,1]. 1=full share, 0=mass
- `pinPenultimate`: If true, pin the penultimate node to CDF(end-1)

# Returns
- `u`: Vector of copula nodes
"""
function copula_nodes_share_safe(distr_i::AbstractVector, grid_i::AbstractVector, nx::Int;
                                  transform::String="none", alpha::Float64=1.0,
                                  pinPenultimate::Bool=false)

    distr = vec(distr_i)
    distr = distr / sum(distr)
    CDF = cumsum(distr)
    nx = max(nx, 3)

    # ----- share integrand (nonnegative) -----
    g = vec(grid_i)

    if transform == "pos"
        wgrid = max.(g, 0)                          # drop negative contribution
    elseif transform == "logshift"
        gmin = minimum(g)
        wgrid = log1p.(g .- gmin .+ 1e-8)           # ensure >=0
    else  # "none" or default
        if any(g .< 0)
            # fallback if user forgot to set a transform
            wgrid = max.(g, 0)
        else
            wgrid = g
        end
    end

    w = wgrid .* distr
    s = sum(w)
    if s > 0
        w = w / s
    else
        w .= 0
    end
    FN = cumsum(w)

    # target equal shares
    ut = zeros(nx)
    ut[1] = max(1e-9, CDF[1])
    ut[end] = 1 - 1e-9

    if pinPenultimate
        ut[end-1] = max(ut[1] + 1e-6, CDF[end-1])   # y-axis trick
    end

    x2 = 1.0

    # MATLAB: for ii = 2:nx-(o.pinPenultimate*1)
    # If pinPenultimate is false: loop 2 to nx
    # If pinPenultimate is true: loop 2 to nx-1
    last_ii = pinPenultimate ? nx - 1 : nx

    for ii in 2:last_ii
        # invert FN at (nx-ii+1)/nx share from the right
        tgt = 1 - (ii - 1) / nx

        # monotone bracket in U
        x1 = max(ut[1], 1e-9)

        # safe bisection
        lo = x1
        hi = x2 - 1e-12

        for it in 1:50
            mid = 0.5 * (lo + hi)
            # Interpolate: find FN value at CDF position = mid
            interp_val = linear_interp_extrap(CDF, FN, mid)
            if interp_val - tgt > 0
                hi = mid
            else
                lo = mid
            end
        end

        ut[end - ii + 1] = 0.5 * (lo + hi)
        x2 = ut[end - ii + 1]
    end

    # blend with equal-mass
    ue = copula_nodes_mass(nx)
    u = sort((1 - alpha) * ue + alpha * ut)
    u = enforce_min_spacing(u, 1e-6)

    return u
end

"""
    linear_interp_extrap(x, y, xi)

Linear interpolation with extrapolation (similar to MATLAB's interp1 with 'extrap').
"""
function linear_interp_extrap(x::Vector{Float64}, y::Vector{Float64}, xi::Float64)
    n = length(x)

    # Handle extrapolation below
    if xi <= x[1]
        if n >= 2
            slope = (y[2] - y[1]) / (x[2] - x[1])
            return y[1] + slope * (xi - x[1])
        else
            return y[1]
        end
    end

    # Handle extrapolation above
    if xi >= x[end]
        if n >= 2
            slope = (y[end] - y[end-1]) / (x[end] - x[end-1])
            return y[end] + slope * (xi - x[end])
        else
            return y[end]
        end
    end

    # Find the interval
    idx = searchsortedlast(x, xi)
    if idx == 0
        idx = 1
    elseif idx >= n
        idx = n - 1
    end

    # Linear interpolation
    t = (xi - x[idx]) / (x[idx+1] - x[idx])
    return y[idx] + t * (y[idx+1] - y[idx])
end
