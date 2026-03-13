"""
    createSparsebasis(grid, maxdim, Xss)

Creates a sparse basis, POLY, for polynomial approximation.
Assumes grid points coincide with Chebyshev nodes for a relevant transformation.

Returns:
- `Poly`: Sparse basis matrix for polynomial approximation
- `InvCheb`: Generalized inverse to estimate coefficients from function values (least squares)
- `Gamma`: Matrix such that Gamma*x is a perturbation of marginal distributions in Xss

Authors: Christian Bayer and Ralph C. Luetticke, Bonn and London.
"""
function createSparsebasis(grid::Dict, maxdim::Int, Xss::AbstractVector)

    nb = Int(grid["nb"])
    na = Int(grid["na"])
    nse = Int(grid["nse"])

    maxlevel = max(nb, na, nse)

    # Values of Chebyshev Polynomials (1st type) at Chebyshev Nodes
    # MATLAB: Tb = cos(pi* (0:maxlevel-1)' * (linspace(0.5/grid.nb/2,1-0.5/grid.nb*2,grid.nb)))'
    # In MATLAB: (0:maxlevel-1)' is column (maxlevel x 1), linspace is row (1 x nb)
    # Product is (maxlevel x nb), then transpose to (nb x maxlevel)
    orders = collect(0:maxlevel-1)  # column vector (maxlevel,)

    nodes_b = collect(range(0.5/nb/2, 1-0.5/nb*2, length=nb))
    Tb = transpose(cos.(π * orders * nodes_b'))  # (nb x maxlevel)

    nodes_a = collect(range(0.5/na*2, 1-0.5/na/2, length=na))
    Ta = transpose(cos.(π * orders * nodes_a'))  # (na x maxlevel)

    nodes_se = collect(range(0.5/(nse-1), 1-0.5/(nse-1), length=nse-1))
    Tse = transpose(cos.(π * orders * nodes_se'))  # (nse-1 x maxlevel)

    # Build Poly matrix
    Poly_cols = Vector{Float64}[]

    # First triple loop
    for j1 in 1:(nse-1)
        for j2 in 1:na
            for j3 in 1:nb
                if j1 + j2 + j3 < maxdim
                    # ndgrid in MATLAB: first arg varies fastest in linear indexing
                    # Julia's ndgrid equivalent
                    TT1, TT2, TT3 = ndgrid(Tb[:, j3], Ta[:, j2], vcat(Tse[:, j1], 0.0))
                    push!(Poly_cols, vec(TT1 .* TT2 .* TT3))
                end
            end
        end
    end

    # Second double loop
    for j2 in 1:nb
        for j3 in 1:na
            if j2 + j3 < maxdim - 1
                TT1, TT2, TT3 = ndgrid(Tb[:, j2], Ta[:, j3], vcat(zeros(nse-1), 1.0))
                push!(Poly_cols, vec(TT1 .* TT2 .* TT3))
            end
        end
    end

    # Concatenate columns into matrix
    Poly = hcat(Poly_cols...)

    # Compute generalized inverse
    InvCheb = (Poly' * Poly) \ Poly'

    # Mapping for Histogram
    Gamma = zeros(nb + na + nse, nb + na + nse - 3)

    # First block (b dimension)
    for i in 1:nb-1
        Gamma[1:nb, i] .= -Xss[1:nb]
        Gamma[i, i] = 1 - Xss[i]
        Gamma[i, i] = Gamma[i, i] - sum(Gamma[1:nb, i])
    end

    jj = nb

    # Second block (a dimension)
    for i in 1:na-1
        Gamma[jj .+ (1:na), jj + i - 1] .= -Xss[jj .+ (1:na)]
        Gamma[jj + i, jj - 1 + i] = 1 - Xss[jj + i]
        Gamma[jj + i, jj - 1 + i] = Gamma[jj + i, jj - 1 + i] - sum(Gamma[jj .+ (1:na), jj - 1 + i])
    end

    jj = nb + na

    # Third block (se dimension)
    for i in 1:nse-1
        Gamma[jj .+ (1:nse), jj + i - 2] .= -Xss[jj .+ (1:nse)]
        Gamma[jj + i, jj - 2 + i] = 1 - Xss[jj + i]
        Gamma[jj + i, jj - 2 + i] = Gamma[jj + i, jj - 2 + i] - sum(Gamma[jj .+ (1:nse), jj - 2 + i])
    end

    return Poly, InvCheb, Gamma
end

