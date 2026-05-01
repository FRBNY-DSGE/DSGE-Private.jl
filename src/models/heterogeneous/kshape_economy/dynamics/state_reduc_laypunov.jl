using LinearAlgebra

"""
    state_reduc_laypunov(hx, gx, grid, P_ref, Q_ref;
                         Dindex=nothing, Vindex=nothing,
                         further_compress_critS=1e-9,
                         further_compress_critC=1e-9,
                         shock_sigmas=nothing)

Combined translation of MATLAB `compute_reduction.m` and
`state_reduc_Laypunov.m`, adapted to this codebase's SGU outputs:

- `hx`, `gx` are from `SGU_solver(...)`
- `grid` is a Dict-like object used across kshape code
- `P_ref`, `Q_ref` are from the linearized system assembly

Returns a named tuple containing projection matrices and reduced objects.
"""
function state_reduc_laypunov(
    hx::AbstractMatrix,
    gx::AbstractMatrix,
    grid,
    P_ref::AbstractMatrix,
    Q_ref::AbstractMatrix;
    Dindex=nothing,
    Vindex=nothing,
    further_compress_critS::Real=1e-9,
    further_compress_critC::Real=1e-9,
    shock_sigmas=nothing,
)
    getgrid(g, k::Symbol) =
        g isa AbstractDict ? (
            haskey(g, k) ? g[k] :
            haskey(g, String(k)) ? g[String(k)] :
            error("Missing grid key: $(k)")
        ) : getproperty(g, k)

    n_states = Int(getgrid(grid, :numstates))
    n_states_endo = Int(getgrid(grid, :numstates_endo))
    n_shocks = Int(getgrid(grid, :numstates_shocks))
    n_controls = Int(getgrid(grid, :numcontrols))

    Didx = isnothing(Dindex) ? getgrid(grid, :Dindex) : Dindex
    Vidx = isnothing(Vindex) ? getgrid(grid, :Vindex) : Vindex
    Didx = collect(Int.(vec(Didx)))
    Vidx = collect(Int.(vec(Vidx)))

    if isnothing(shock_sigmas)
        shock_sigmas = ones(n_shocks)
    end
    shock_sigmas = vec(Float64.(shock_sigmas))
    length(shock_sigmas) == n_shocks || error("shock_sigmas length must equal numstates_shocks")

    # Structural shock covariance in full state space (shocks are at the tail).
    SCov = zeros(Float64, n_states, n_states)
    shock_idx = (n_states - n_shocks + 1):n_states
    @inbounds for i in 1:n_shocks
        SCov[shock_idx[i], shock_idx[i]] = shock_sigmas[i]^2
    end

    # Lyapunov: hx*X*hx' - X + SCov = 0
    StateCOVAR = Matrix(solve_discrete_lyapunov(Matrix(hx), SCov))
    StateCOVAR = (StateCOVAR + StateCOVAR') / 2

    ControlCOVAR = Matrix(gx) * StateCOVAR * Matrix(gx)'
    ControlCOVAR = (ControlCOVAR + ControlCOVAR') / 2

    # MATLAB covariance blocks are symmetric; enforce that here before eig.
    CovD = Symmetric((StateCOVAR[Didx, Didx] + StateCOVAR[Didx, Didx]') / 2)
    eS = eigen(CovD)
    evalS = real.(eS.values)
    evecS = eS.vectors
    maxS = maximum(abs.(evalS))
    keepD = maxS == 0 ? trues(length(evalS)) : abs.(evalS) .> maxS * float(further_compress_critS)
    any(keepD) || (keepD[argmax(abs.(evalS))] = true)

    CovV = Symmetric((ControlCOVAR[Vidx, Vidx] + ControlCOVAR[Vidx, Vidx]') / 2)
    eV = eigen(CovV)
    evalV = real.(eV.values)
    evecV = eV.vectors
    maxV = maximum(abs.(evalV))
    keepV = maxV == 0 ? trues(length(evalV)) : abs.(evalV) .> maxV * float(further_compress_critC)
    any(keepV) || (keepV[argmax(abs.(evalV))] = true)

    PRightStates_aux = Matrix{Float64}(I, n_states, n_states)
    PRightStates_aux[Didx, Didx] .= evecS
    keep_states = trues(n_states)
    keep_states[Didx[.!keepD]] .= false
    PRightStates = PRightStates_aux[:, keep_states]
    n_states_r = size(PRightStates, 2)

    n_total = n_states + n_controls
    Vindex_full = n_states .+ Vidx
    PRightAll_aux = Matrix{Float64}(I, n_total, n_total)
    PRightAll_aux[Didx, Didx] .= evecS
    PRightAll_aux[Vindex_full, Vindex_full] .= evecV

    keep_all = trues(n_total)
    keep_all[Didx[.!keepD]] .= false
    keep_all[Vindex_full[.!keepV]] .= false
    PRightAll = PRightAll_aux[:, keep_all]
    n_controls_r = n_controls - count(.!keepV)

    # Match MATLAB state_reduc_Laypunov.m row selection for P_ref compatibility:
    # keep endogenous-state rows + control rows (drop shock-state rows).
    keep_rows = vcat(collect(1:n_states_endo), collect((n_states + 1):(n_states + n_controls)))
    PRightAll_Pref = PRightAll[keep_rows, :]

    hx_r = PRightStates' * Matrix(hx) * PRightStates
    gx_r = Matrix(gx) * PRightStates
    P_r = PRightAll_Pref' * Matrix(P_ref) * PRightAll_Pref
    Q_r = PRightAll_Pref' * Matrix(Q_ref)

    return PRightStates, PRightAll, PRightAll_Pref, n_states_r, n_controls_r, StateCOVAR, ControlCOVAR, hx_r, gx_r, P_r, Q_r, keepD, keepV
end

