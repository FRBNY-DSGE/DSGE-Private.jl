using DSGE, Test, ModelConstructors, Plots, BenchmarkTools

function check_kolmogorov(D_as::AbstractMatrix{S}, C_Final::AbstractMatrix{S},
                          ehi::S, elo::S, agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S},
                          ω::S, H::S, r::S, γ::S, T::S) where {S <: Real}
    # Check D(a, s) solves the KF equation
    m′_as  = zeros(S, size(D_as))
    na, ns = size(D_as)
    qfunc(x) = DSGE.mollifier_hetdsgegovdebt(x, ehi, elo)
    for isp in 1:ns
        for iap in 1:na
            for is in 1:ns
                for ia in 1:na
                    m′_as[iap, isp] += D_as[ia, is] * f[is, isp] / (ω * H * sgrid[isp]) *
                        qfunc((agrid[iap] - (1 + r) * exp(-γ) * (agrid[ia] - C_Final[ia, is]) - T) / (ω * H * sgrid[isp]))
                end
            end
        end
    end

    D′_as = m′_as ./ sum(m′_as) # map m′(a, s) -> D′(a, s) by ensuring sum(D′(a, s)) = 1
    return D′_as
end

function check_kolmogorov_eigen(D_as::AbstractMatrix{S}, C_Final::AbstractMatrix{S},
                                ehi::S, elo::S, agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S},
                                ω::S, H::S, r::S, γ::S, T::S, ι::S) where {S <: Real}
    # Check D(a, s) solves the KF equation by constructing the transition matrix A and solving the eigenvalue problem
    # vec(m′(a, s))     = A * ι * vec(m(a, s)), which reduces to
    # vec(D′(a, s)) / ι = A * vec(D(a, s))
    # vec(D′(a, s))     = A * ι * vec(D(a, s))
    N = length(D_as)
    transition_mat  = Matrix{S}(undef, N, N)
    na, ns = size(D_as)
    qfunc(x) = DSGE.mollifier_hetdsgegovdebt(x, ehi, elo)
    for is in 1:ns
        for ia in 1:na
            for isp in 1:ns
                for iap in 1:na
                    transition_mat[na * (isp - 1) + iap, na * (is - 1) + ia] = f[is, isp] / (ω * H * sgrid[isp]) *
                        qfunc((agrid[iap] - (1 + r) * exp(-γ) * (agrid[ia] - C_Final[ia, is]) - T) / (ω * H * sgrid[isp]))
                end
            end
        end
    end

    # Find largest eigenvalue
    transition_mat .*= ι
    (D, V) = (eigen(transition_mat)..., )
    max_D  = argmax(abs.(D))
    D      = D[max_D]

    # Pick eigenvector associated w/ largest eigenvalue and moving it back to values
    μ = real(V[:,max_D])
    μ ./= sum(μ) # normalize to 1

    return reshape(μ, na, ns), D
end

function check_kolmogorov_eigen!(transition_mat::AbstractMatrix{S}, D_as::AbstractMatrix{S}, C_Final::AbstractMatrix{S},
                                ehi::S, elo::S, agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S},
                                ω::S, H::S, r::S, γ::S, T::S, ι::S) where {S <: Real}
    # Check D(a, s) solves the KF equation by constructing the transition matrix A and solving the eigenvalue problem
    # vec(m′(a, s))     = A * ι * vec(m(a, s)), which reduces to
    # vec(D′(a, s)) / ι = A * vec(D(a, s))
    # vec(D′(a, s))     = A * ι * vec(D(a, s))
    na, ns = size(D_as)
    qfunc(x) = DSGE.mollifier_hetdsgegovdebt(x, ehi, elo)
    for is in 1:ns
        for ia in 1:na
            for isp in 1:ns
                for iap in 1:na
                    transition_mat[na * (isp - 1) + iap, na * (is - 1) + ia] = f[is, isp] / (ω * H * sgrid[isp]) *
                        qfunc((agrid[iap] - (1 + r) * exp(-γ) * (agrid[ia] - C_Final[ia, is]) - T) / (ω * H * sgrid[isp]))
                end
            end
        end
    end

    # Find largest eigenvalue
    transition_mat .*= ι
    (D, V) = (eigen(transition_mat)..., )
    max_D  = argmax(abs.(D))
    D      = D[max_D]

    # Pick eigenvector associated w/ largest eigenvalue and moving it back to values
    μ = real(V[:,max_D])
    μ ./= sum(μ) # normalize to 1

    return reshape(μ, na, ns), D
end

m = HetDSGEGovDebt(; ref_dir =

                   joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
# m <= Setting(:na, 600)
# m <= Setting(:ne, 10)
DSGE.method3_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none)

D_as = reshape(m[:μstar].value, get_setting(m, :na), get_setting(m, :ns))
C_as = reshape(m[:cstar].value, get_setting(m, :na), get_setting(m, :ns))
pLH = m[:pLH].value
pHL = m[:pHL].value
f = [[1-pLH pLH];[pHL 1-pHL]] # f1[i,j] is prob of going from i to j
D′_as = check_kolmogorov(D_as, C_as, m[:ehi].value, m[:elo].value, m.grids[:agrid].points,
                         m.grids[:sgrid].points, f, m[:ωstar].value, m[:H].value, m[:r].scaledvalue,
                         m[:γ].scaledvalue, m[:Tstar].value)
D′_as_eigen, λ = check_kolmogorov_eigen(D_as, C_as, m[:ehi].value, m[:elo].value, m.grids[:agrid].points,
                                        m.grids[:sgrid].points, f, m[:ωstar].value, m[:H].value, m[:r].scaledvalue,
                                        m[:γ].scaledvalue, m[:Tstar].value, m.grids[:agrid].weights[1])
println("One iteration")
@btime begin
    D′_as = check_kolmogorov(D_as, C_as, m[:ehi].value, m[:elo].value, m.grids[:agrid].points,
                             m.grids[:sgrid].points, f, m[:ωstar].value, m[:H].value, m[:r].scaledvalue,
                             m[:γ].scaledvalue, m[:Tstar].value)
end

println("Direct")
@btime begin
    D_as0 = deepcopy(D_as)
end
@btime begin
    D_as0 = deepcopy(D_as)
    for i in 1:100
        D′_as = check_kolmogorov(D_as0, C_as, m[:ehi].value, m[:elo].value, m.grids[:agrid].points,
                                 m.grids[:sgrid].points, f, m[:ωstar].value, m[:H].value, m[:r].scaledvalue,
                                 m[:γ].scaledvalue, m[:Tstar].value)
        if maximum(abs.(D′_as - D_as0)) < 1e-6
            break
        else
            D_as0 .= D′_as
        end
    end
end
for i in 1:100
    D′_as = check_kolmogorov(D_as, C_as, m[:ehi].value, m[:elo].value, m.grids[:agrid].points,
                             m.grids[:sgrid].points, f, m[:ωstar].value, m[:H].value, m[:r].scaledvalue,
                             m[:γ].scaledvalue, m[:Tstar].value)
    if maximum(abs.(D′_as - D_as)) < 1e-6
        println("The number of fixed point iterations using direct adding is $(i)")
        break
    else
        D_as .= D′_as
    end
end

println("Eigenvalue solution")
@btime begin
    D′_as_eigen, λ = check_kolmogorov_eigen(D_as, C_as, m[:ehi].value, m[:elo].value, m.grids[:agrid].points,
                                            m.grids[:sgrid].points, f, m[:ωstar].value, m[:H].value, m[:r].scaledvalue,
                                            m[:γ].scaledvalue, m[:Tstar].value, m.grids[:agrid].weights[1])
end

println("Eigenvalue solution, in place for transition matrix")
N = length(D_as)
transition_mat  = Matrix{Float64}(undef, N, N)
@btime begin
    D′_as_eigen, λ = check_kolmogorov_eigen!(transition_mat, D_as, C_as, m[:ehi].value, m[:elo].value, m.grids[:agrid].points,
                                            m.grids[:sgrid].points, f, m[:ωstar].value, m[:H].value, m[:r].scaledvalue,
                                            m[:γ].scaledvalue, m[:Tstar].value, m.grids[:agrid].weights[1])
end

agrid = m.grids[:agrid].points
fig1 = plot(agrid, D_as[:, 1], label = "D(a, slo)")
plot!(agrid, D′_as[:, 1], label = "D'(a, slo) Direct")
plot!(agrid, D′_as_eigen[:, 1], label = "D'(a, slo) Eigen")
fig2 = plot(agrid, D_as[:, 2], label = "D(a, shi)")
plot!(agrid, D′_as[:, 2], label = "D'(a, shi) Direct")
plot!(agrid, D′_as_eigen[:, 2], label = "D'(a, shi) Eigen")

fig3 = plot(agrid, D_as[:, 1] - D′_as[:, 1], label = "D(a, slo) - D'(a,slo) Direct")
plot!(agrid, D_as[:, 1] - D′_as_eigen[:, 1], label = "D(a, slo) - D'(a, slo) Eigen")
fig4 = plot(agrid, D_as[:, 2] - D′_as[:, 2], label = "D(a, slo) - D'(a,slo) Direct")
plot!(agrid, D_as[:, 2] - D′_as_eigen[:, 2], label = "D(a, slo) - D'(a, slo) Eigen")

nothing
