"""
```
tauchen86(ρ, N; σ = 1., μ_e = 0.)
```
Generate a discrete approximation to an AR(1) process, following Tauchen (1986).

Uses importance sampling: each bin has probability 1/N to realize

# Arguments
- `ρ`: autocorrelation coefficient
- `N`: number of gridpoints
- `σ`: long-run variance (short-run variance is `σ_e = σ * sqrt(1 - ρ^2)`)
- `μ_e`: mean of the AR(1) process

# Returns
- `grid_vec`: state vector grid
- `P`: transition matrix
- `bounds`: bin bounds
"""
function tauchen86(ρ::T, N::Int64; σ::T = 1.0, μ_e::T = 0.0) where {T <: Real}
#   Author: Christian Bayer, Uni Bonn, 03.05.2010
#   Modified by William Chen, April 6, 2021.
#   See https://github.com/BenjaminBorn/HANK_BusinessCycleAndInequality for original implementation as "Tauchen"

    dis = Normal()
    pr_ij(x, bound1, bound2, ρ, σ_e) = pdf.(dis, x) .*
        (cdf.(dis, (bound2 - ρ .* x) ./ σ_e) -
         cdf.(dis, (bound1 - ρ .* x) ./ σ_e) )

    grid_probs = range(0.0, stop = 1.0, length = N+1)   # generate equi-likely bins

    bounds = quantile.(dis, grid_probs[1:end])     # corresponding bin bounds

    # replace [-]Inf bounds, by finite numbers
    bounds[1] = bounds[2] - 1.0e2
    bounds[end] = bounds[end-1] + 1.0e2

    # Calculate grid() - centers
    grid_vec = N * (pdf.(dis, bounds[1:end-1]) - pdf.(dis, bounds[2:end]))

    # σ_e = sqrt(1 - ρ^2) # Calculate short run variance # NOTE: this was the original line of code, but I think this assumes σ = 1
    σ_e = sqrt(1 - ρ^2) * σ # Calculate short run variance
    P = fill(0.0, (N, N)) # Initialize Transition Probability Matrix

    for j = 1:N
        p(x) = pr_ij(x,bounds[j], bounds[j+1], ρ, σ_e)
        for i = 1:floor(Int64, (N-1)/2)+1 # Exploit Symmetrie to save running time
            P[i, j] = _bbl_gauss_chebyshev_integrate(p, bounds[i], bounds[i+1]) # Evaluate Integral
        end
    end

    # Exploit Symmetrie Part II
    P[floor(Int64, (N - 1) / 2) + 2:N, :] = P[(ceil(Int64, (N - 1) / 2):-1:1), end:-1:1]

    # Make sure P is a Probability Matrix
    P = P ./ sum(P, dims = 2)

    grid_vec   = grid_vec .* σ .+ μ_e
    lmul!(σ, bounds)
    # bounds = bounds .* σ

    return grid_vec, P, bounds
end

function ExTransition(rho::Number,bounds::Array{Float64,1},riskscale::Number)
#similar to TAUCHEN
N = length(bounds)-1
# Assume Importance Sampling
        sigma_e=riskscale*sqrt(1-rho^2)# Calculate short run variance
        P=zeros(typeof(riskscale),N,N) # Initialize Transition Probability Matrix

        for i=1:floor(Int,(N-1)/2)+1
            nodes, weights = my_qnwcheb(500, bounds[i], bounds[i+1])
            for j=1:N
            p(x) = pr_ij(x,bounds[j],bounds[j+1],rho,sigma_e)
             # Exploit Symmetrie to save running time
                P[i,j]=dot(weights, p.(nodes)) # _bbl_gauss_chebyshev_integrate(p,bounds[i],bounds[i+1]) # Evaluate Integral
            end
        end

       # Exploit Symmetrie Part II
        P[floor(Int,(N-1)/2)+2:N,:]=P[(ceil(Int,(N-1)/2):-1:1),end:-1:1]

# Make sure P is a Probability Matrix
P = P./ sum(P, dims = 2)

return P

end

function pr_ij(x,bound1,bound2,rho,sigma_e)
    mycdf(x) = 0.5 + 0.5 * erf.(x / sqrt(2.0))
    mypdf(x) =  1/sqrt(2*π).*exp.(-x.^2/2.0)
    p = mypdf.(x) .* (mycdf.((bound2 - rho.*x)./sigma_e) - mycdf.((bound1 - rho.*x)./sigma_e) )
return p
end
