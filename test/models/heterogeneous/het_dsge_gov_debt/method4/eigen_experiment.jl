using DSGE, Plots, BenchmarkTools, ModelConstructors, Arpack, ArnoldiMethod, KrylovKit
using IterativeSolvers, SparseArrays, LinearAlgebra

m = HetDSGEGovDebt(; ref_dir =
                   joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
DSGE.method4_steadystate!(m; tol = 5e-4, doplots = false, verbose = :none, kf_eigen = true, kf_anderson = false)

na = get_setting(m, :na)
ns = get_setting(m, :ns)
agrid = m.grids[:agrid].points
sgrid = m.grids[:sgrid].points
ω = m[:ωstar].value
H = m[:H].value
T = m[:Tstar].value
γ = m[:γ].scaledvalue
R  = 1 + m[:r].scaledvalue
η  = m[:η].value
bg = m[:bg].value

# Construct Markov transition matrix for skill
pLH = m[:pLH].value
pHL = m[:pHL].value
f = [[1-pLH pLH];[pHL 1-pHL]] # f1[i,j] is prob of going from i to j

# Initialize mollifier function
qfunc(x) = DSGE.mollifier_hetdsgegovdebt(x, m[:ehi].value, m[:elo].value)
ι = (agrid[end] - agrid[1]) / length(agrid)
transition_mat = Matrix{Float64}(undef, na * ns, na * ns) # This is typically sparse
C_as = reshape(m[:cstar].value, na, ns)

function construct_transition!(transition_mat::AbstractMatrix{S}, C_as::AbstractMatrix{S},
                               agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S},
                               qfunc, ω::S, H::S, R::S, γ::S, T::S) where {S <: Real}
    na = length(agrid)
    ns = length(sgrid)
    @inbounds @simd for is in 1:ns
        @inbounds @simd for ia in 1:na
            @inbounds @simd for isp in 1:ns
                @inbounds @simd for iap in 1:na
                    transition_mat[na * (isp - 1) + iap, na * (is - 1) + ia] = f[is, isp] / (ω * H * sgrid[isp]) *
                        qfunc((agrid[iap] - R * exp(-γ) * (agrid[ia] - C_as[ia, is]) - T) / (ω * H * sgrid[isp]))
                end
            end
        end
    end
end

function construct_transition!(transition_mat::SparseMatrixCSC{S}, C_as::AbstractMatrix{S},
                               agrid::AbstractVector{S}, sgrid::AbstractVector{S}, f::AbstractMatrix{S},
                               qfunc, ω::S, H::S, R::S, γ::S, T::S) where {S <: Real}
    na = length(agrid)
    ns = length(sgrid)
    @inbounds @simd for is in 1:ns
        @inbounds @simd for ia in 1:na
            @inbounds @simd for isp in 1:ns
                @inbounds @simd for iap in 1:na
                    ele = f[is, isp] / (ω * H * sgrid[isp]) *
                        qfunc((agrid[iap] - R * exp(-γ) * (agrid[ia] - C_as[ia, is]) - T) / (ω * H * sgrid[isp]))
                    if ele != 0.
                        transition_mat[na * (isp - 1) + iap, na * (is - 1) + ia] = ele
                    end
                end
            end
        end
    end
end

sparse_transition_mat = spzeros(600, 600)
construct_transition!(transition_mat, C_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T)
construct_transition!(sparse_transition_mat, C_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T)

# Sparse solvers
println("Arpack (sparse)") # third fastest
@btime begin
    eigs(sparse_transition_mat, nev = 1)
    nothing
end
println("IterativeSolvers (sparse)") # fewest allocations
@btime begin
    powm(sparse_transition_mat)
    nothing
end
println("KrylovKit (sparse)") # fastest
@btime begin
    eigsolve(sparse_transition_mat, 1, :LM)
    nothing
end
println("Arnoldi (sparse)") # second fastest
@btime begin
    partialeigen(partialschur(sparse_transition_mat; nev = 1, which = LM())[1])
    nothing
end

# Full
println("Arpack (full)") # third fastest
@btime begin
    eigs(transition_mat, nev = 1)
    nothing
end
println("IterativeSolvers (full)") # fewest allocations
@btime begin
    powm(transition_mat)
    nothing
end
println("KrylovKit (full)") # second fastest
@btime begin
    eigsolve(transition_mat, 1, :LM)
    nothing
end
println("Arnoldi (full)") # fastest
@btime begin
    partialeigen(partialschur(transition_mat; nev = 1, which = LM())[1])
    nothing
end
println("eigen (full)") # slowest
@btime begin
    eigen(transition_mat)
    nothing
end

# Now testing the whole update w/select solvers
println("KrylovKit (sparse)") # fastest
@btime begin
    construct_transition!(sparse_transition_mat, C_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T)
    eigsolve(sparse_transition_mat, 1, :LM)
    nothing
end
println("Arnoldi (sparse)") # second fastest
@btime begin
    construct_transition!(sparse_transition_mat, C_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T)
    partialeigen(partialschur(sparse_transition_mat; nev = 1, which = LM())[1])
    nothing
end

println("KrylovKit (full)") # second fastest
@btime begin
    construct_transition!(transition_mat, C_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T)
    eigsolve(transition_mat, 1, :LM)
    nothing
end
println("Arnoldi (full)") # fastest
@btime begin
    construct_transition!(transition_mat, C_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T)
    partialeigen(partialschur(transition_mat; nev = 1, which = LM())[1])
    nothing
end
println("KrylovKit (sparse from full)") # second fastest
@btime begin
    construct_transition!(transition_mat, C_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T)
    eigsolve(sparse(transition_mat), 1, :LM)
    nothing
end
println("Arnoldi ((sparse from full)") # fastest
@btime begin
    construct_transition!(transition_mat, C_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T)
    partialeigen(partialschur(sparse(transition_mat); nev = 1, which = LM())[1])
    nothing
end
println("eigen (full)") # slowest
@btime begin
    construct_transition!(transition_mat, C_as, agrid, sgrid, f, qfunc, ω, H, R, γ, T)
    eigen(transition_mat)
    nothing
end
