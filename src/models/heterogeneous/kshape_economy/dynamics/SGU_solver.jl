using LinearAlgebra

"""
    SGU_solver(F, param, grid, p)

Solves for a competitive equilibrium defined as a zero of the function F which is
written in Schmitt-Grohé Uribe form using the algorithm suggested in
Schmitt-Grohé and Uribe (2004): "Solving dynamic general equilibrium models
using a second-order approximation to the policy function"

# Arguments
- `F`: Function that computes the system of equations F(State, State_m, Contr, Contr_m)
- `param`: Parameter dictionary (will be modified in place)
- `grid`: Grid structure dictionary with fields `numstates`, `numcontrols`, `nb`, `na`, `oc`, `os`
- `p`: Parameter dictionary with field `NumWorkers` for parallel processing

# Returns
Tuple of:
- `hx`: Policy function for states (nk × nk)
- `gx`: Policy function for controls (numcontrols × nk)
- `F1`: Jacobian w.r.t. tomorrow's states (numstates+numcontrols × numstates)
- `F2`: Jacobian w.r.t. tomorrow's controls (numstates+numcontrols × numcontrols)
- `F3`: Jacobian w.r.t. today's states (numstates+numcontrols × numstates)
- `F4`: Jacobian w.r.t. today's controls (numstates+numcontrols × numcontrols)
- `param`: Updated parameter dictionary

# Notes
- Uses numerical differentiation to compute Jacobians (many F evaluations).
- For speed, start Julia with multiple threads: `julia -t N` (e.g. `julia -t 4`).
- Implements Schmitt-Grohé-Uribe algorithm based on QZ decomposition.
"""
function SGU_solver(F, param, grid, p, parallel = false)

    # Ensure numstates and numcontrols are integers (defensive conversion)
    numstates = Int(grid["numstates"])
    numcontrols = Int(grid["numcontrols"])

    # Absolute deviations (set before Fb so all F calls see identical param state)
    # TODO: SS/transition matrix object — handle separately (scaleval1, scaleval2 are numerical step sizes)
    param["scaleval1"] = 1e-5  # Numerical differentiation step size
    param["scaleval2"] = 1e-5  # Numerical differentiation step size

    # Initialize state and control vectors
    State = zeros(numstates)
    State_m = State
    Contr = zeros(numcontrols)
    Contr_m = Contr

    # Evaluate function at steady state
    @info "SGU_solver: evaluating F at steady state..."
    Fb, ~, ~, ~ = F(State, State_m, Contr, Contr_m)
    # Use Schmitt Grohe Uribe Algorithm
    # E[x' u']' = inv(A)*B*[x u]'
    # A = [dF/dx' dF/du'], B = [dF/dx dF/du]
    # A = [F1 F2]; B = [F3 F4]

    # numstates and numcontrols already defined above

    # Initialize Jacobian matrices
    F1 = zeros(numstates + numcontrols, numstates)   # Tomorrow's states
    F2 = zeros(numstates + numcontrols, numcontrols) # Tomorrow's controls
    F3 = zeros(numstates + numcontrols, numstates)   # Today's states
    # Today's controls: marginal distribution in h should not be moved by controls
    # and have unit effect on Value function error (LAST TWO COLUMNS TO BE FILLED: Aggregate Prices)
    F4 = [zeros(numstates, numcontrols); Matrix{Float64}(I, numcontrols, numcontrols)]

    # Parameters which control numerical differentiation and Reiter solution
    # Handle nothing, Missing p, or missing NumWorkers field

    if isnothing(p) || ismissing(p) || !haskey(p, "NumWorkers")
        # Default to number of threads if p is missing
        pnum = Threads.nthreads()
    else
        pnum = p["NumWorkers"]
    end
    # parallel = false  # set true to use Threads.@threads for Jacobian blocks
    if parallel && pnum == 1
        @info "SGU_solver: running with 1 thread. For faster Jacobians, start Julia with multiple threads: julia -t N"
    end

    # Compute Jacobian F1 = DF/DXprime and F3 = DF/DX
    #Hard-coded for now: 
    pnum = 3 #TO-DO fix pnum stuff
    packagesize = Int(ceil(numstates / (3 * pnum)))
    blocks = Int(ceil(numstates / packagesize))
    @info "SGU_solver: steady state done. Computing Jacobians F1, F3 (2×$numstates F-calls in $blocks blocks, parallel=$parallel)..."

    # Absolute deviations
    # TODO: SS/transition matrix object — handle separately (scaleval1, scaleval2 are numerical step sizes)
    param["scaleval1"] = 1e-5  # Numerical differentiation step size
    param["scaleval2"] = 1e-5  # Numerical differentiation step size

    # Pre-allocate storage for parallel blocks
    FF1 = Vector{Matrix{Float64}}(undef, blocks)
    FF3 = Vector{Matrix{Float64}}(undef, blocks)

    # Compute Jacobian w.r.t. X', X
    if parallel
        Threads.@threads for bl in 1:blocks
            range_start = (bl - 1) * packagesize + 1
            range_end = min(packagesize * bl, numstates)
            range = range_start:range_end
            range_len = length(range)

            DF1 = zeros(length(Fb), range_len)
            DF3 = zeros(length(Fb), range_len)
            cc = zeros(numcontrols)
            ss = zeros(numstates)

            for (idx, Xct) in enumerate(range)
                h = param["scaleval1"]  # TODO: SS/transition matrix object — handle separately
                X = zeros(Float64, numstates)
                X[Xct] = h

                # F3: derivative w.r.t. today's state
                Fx, ~, ~, ~ = F(ss, X, cc, cc)
                DF3[:, idx] = (Fx - Fb) / h

                # F1: derivative w.r.t. tomorrow's state
                Fx, ~, ~, ~ = F(X, ss, cc, cc)
                DF1[:, idx] = (Fx - Fb) / h
            end

            FF1[bl] = DF1
            FF3[bl] = DF3
        end
    else
        for bl in 1:blocks
            range_start = (bl - 1) * packagesize + 1
            range_end = min(packagesize * bl, numstates)
            range = range_start:range_end
            range_len = length(range)

            DF1 = zeros(length(Fb), range_len)
            DF3 = zeros(length(Fb), range_len)
            cc = zeros(numcontrols)
            ss = zeros(numstates)

            for (idx, Xct) in enumerate(range)
                X = zeros(Float64, numstates)
                h = param["scaleval1"]  # TODO: SS/transition matrix object — handle separately
                X[Xct] = h

                # F3: derivative w.r.t. today's state
                Fx, _, _, _ = F(ss, X, cc, cc)
                DF3[:, idx] = (Fx - Fb) / h
                
                # F1: derivative w.r.t. tomorrow's state
                Fx, _, _, _ = F(X, ss, cc, cc)
                DF1[:, idx] = (Fx - Fb) / h
            end

            FF1[bl] = DF1
            FF3[bl] = DF3
        end
    end
    Main.xx[][:blocks] = blocks
    Main.xx[][:packagesize] = packagesize
    Main.xx[][:Fb] = Fb
    Main.xx[][:FF1] = FF1
    Main.xx[][:FF3] = FF3
    #foo()

    # Assemble F1 and F3 from blocks
    for i in 1:blocks
        range_start = (i - 1) * packagesize + 1
        range_end = min(packagesize * i, numstates)
        range = range_start:range_end
        F1[:, range] = FF1[i]
        F3[:, range] = FF3[i]
    end

    # Compute Jacobian F2 = DF/DYprime
    packagesize = Int(ceil(numcontrols / (3 * pnum)))
    blocks = Int(ceil(numcontrols / packagesize))
    @info "SGU_solver: F1/F3 done. Computing F2 ($numcontrols F-calls in $blocks blocks)..."

    FF = Vector{Matrix{Float64}}(undef, blocks)

    if parallel
        Threads.@threads for bl in 1:blocks
            range_start = (bl - 1) * packagesize + 1
            range_end = min(packagesize * bl, numcontrols)
            range = range_start:range_end
            range_len = length(range)

            DF2 = zeros(length(Fb), range_len)
            cc = zeros(numcontrols)
            ss = zeros(numstates)

            Y = zeros(numcontrols)
            for (idx, Yct) in enumerate(range)
                h = param["scaleval2"]  # TODO: SS/transition matrix object — handle separately
                fill!(Y, 0.0)
                Y[Yct] = h

                Fx, ~, ~, ~ = F(ss, ss, Y, cc)
                DF2[:, idx] = (Fx - Fb) / h
            end

            FF[bl] = DF2
        end
    else
        for bl in 1:blocks
            range_start = (bl - 1) * packagesize + 1
            range_end = min(packagesize * bl, numcontrols)
            range = range_start:range_end
            range_len = length(range)

            DF2 = zeros(length(Fb), range_len)
            cc = zeros(numcontrols)
            ss = zeros(numstates)

            Y = zeros(numcontrols)
            for (idx, Yct) in enumerate(range)
                h = param["scaleval2"]  # TODO: SS/transition matrix object — handle separately
                fill!(Y, 0.0)
                Y[Yct] = h

                Fx, ~, ~, ~ = F(ss, ss, Y, cc)
                DF2[:, idx] = (Fx - Fb) / h
            end

            FF[bl] = DF2
        end
    end

    # Assemble F2 from blocks
    for i in 1:blocks
        range_start = (i - 1) * packagesize + 1
        range_end = min(packagesize * i, numcontrols)
        range = range_start:range_end
        F2[:, range] = FF[i]
    end

    # Derivative w.r.t. Y (value functions today change only themselves)
    cc = zeros(numcontrols)
    ss = zeros(numstates)
    oc = Int(grid["oc"])
    @info "SGU_solver: F2 done. Computing F4 ($oc F-calls)..."

    Y = zeros(numcontrols)
    for Yct in 0:(oc - 1)
        h = param["scaleval2"]
        fill!(Y, 0.0)
        Y[end - Yct] = h

        Fx, ~, ~, ~ = F(ss, ss, cc, Y)
        F4[:, end - Yct] = (Fx - Fb) / h
    end

    # Remove numerical garbage from F2: marginal distribution in h should not be moved by controls
    nb = Int(grid["nb"])
    na = Int(grid["na"])
    os = Int(grid["os"])
    F2[(nb + na - 2):(numstates - os), :] .= 0

    # Schmidt-Grohé and Uribe code based on QZ decomposition of the system
    # Code adapted from SGU's online resources
    @info "SGU_solver: F4 done. Running QZ decomposition..."

    # QZ decomposition: [F1, F2] * E[x', u'] = -[F3, F4] * [x, u]
    # In MATLAB: [s, t, Q, Z] = qz(full([F1,F2]), full(-[F3, F4]))
    # In Julia, we use schur for generalized Schur decomposition
    A = [F1 F2]
    B = -[F3 F4]
    # Generalized Schur decomposition: A = Q*S*Z', B = Q*T*Z'
    schur_result = schur(A, B)
    s = schur_result.S  # S matrix
    t = schur_result.T  # T matrix
    Q = schur_result.left  # Q matrix
    Z = schur_result.right  # Z matrix

    # Compute generalized eigenvalues: |s|/|t|
    relev = abs.(diag(s)) ./ abs.(diag(t))  # Match MATLAB exactly
    ll = sort(relev)
    slt = relev .>= 1.0
    Sm = Matrix(s)
    nk = sum(slt)  # Number of state variables based on eigenvalues

    if nk > numstates
        if haskey(param, "overrideEigen") && param["overrideEigen"]  # TODO: SS/transition matrix object — handle separately
            @warn "The Equilibrium is Locally Indeterminate, critical eigenvalue shifted to: $(ll[end - numstates])"
            slt = relev .> ll[end - numstates]
            nk = sum(slt)
        else
            error("No Local Equilibrium Exists, last eigenvalue: $(ll[end - numstates])")
        end
    elseif nk < numstates
        if haskey(param, "overrideEigen") && param["overrideEigen"]  # TODO: SS/transition matrix object — handle separately
            threshold = ll[end - numstates]
            @warn "No Local Equilibrium Exists, critical eigenvalue shifted to: $threshold"
            slt = relev .> threshold
            nk = sum(slt)
            @info "SGU_solver: After shift: nk=$nk, numstates=$numstates, threshold=$threshold"
            # If still not equal, try >= instead of >
            if nk != numstates
                @info "SGU_solver: nk != numstates, trying >= comparison"
                slt = relev .>= threshold
                nk = sum(slt)
                @info "SGU_solver: After >= shift: nk=$nk"
            end
        else
            error("No Local Equilibrium Exists, last eigenvalue: $(ll[end - numstates])")
        end
    end

    # Reorder QZ decomposition based on eigenvalue selection
    # In MATLAB: [s, t, ~, Z] = ordqz(s, t, Q, Z, slt)
    # LAPACK.tgsen! has been in Julia since 1.0 (added Jan 2015, PR #9701). Use it for
    # proper orthogonal reordering; a simple permutation breaks quasitriangular structure.
    S = copy(Matrix(schur_result.S))
    T = copy(Matrix(schur_result.T))
    Qmat = copy(Matrix(schur_result.left))
    Zmat = copy(Matrix(schur_result.right))
    # Julia 1.5 tgsen! signature: tgsen!(select::Vector{Int64}, S, T, Q, Z)
    # Convert Bool to Int64 for the select vector
    selectInt = Vector{Int64}(slt)
    z11 = Matrix{Float64}(undef, nk, nk)
    z21 = Matrix{Float64}(undef, size(Zmat, 1) - nk, nk)
    s11 = Matrix{Float64}(undef, nk, nk)
    t11 = Matrix{Float64}(undef, nk, nk)

    # Check for eigenvalues near the threshold
    thresh = ll[end - numstates]
    near_thresh = findall(x -> abs(x - thresh) < 1e-8, relev)
    @info "SGU_solver: Eigenvalues near threshold ($thresh): $(length(near_thresh)) at indices $near_thresh"
    if length(near_thresh) > 0
        @info "SGU_solver: Values near threshold: $(relev[near_thresh])"
    end
    @info "SGU_solver: Before tgsen: rank(Zmat)=$(rank(Zmat)), sum(selectInt)=$(sum(selectInt))"

    try
        # Reorder QZ decomposition: selected eigenvalues (selectInt==1) move to top-left
        LinearAlgebra.LAPACK.tgsen!(selectInt, S, T, Qmat, Zmat)
        @info "SGU_solver: After tgsen: rank(Zmat)=$(rank(Zmat))"
        # Check for 2x2 blocks in S (complex eigenvalue pairs)
        n2x2 = 0
        for i in 1:(size(S,1)-1)
            if abs(S[i+1, i]) > 1e-10
                n2x2 += 1
            end
        end
        @info "SGU_solver: Number of 2x2 blocks in S: $n2x2"

        # Check SVD of first nk columns of Z
        Z_first_nk = Zmat[:, 1:nk]
        sv = svdvals(Z_first_nk)
        @info "SGU_solver: Smallest 5 singular values of Z[:, 1:nk]: $(sv[end-4:end])"

        z11 .= Zmat[1:nk, 1:nk]
        z21 .= Zmat[(nk + 1):end, 1:nk]
        s11 .= S[1:nk, 1:nk]
        t11 .= T[1:nk, 1:nk]
    catch e
        # Fallback if tgsen! fails (e.g. signature change); result may not match MATLAB
        @warn "SGU_solver: LAPACK.tgsen! failed, using permutation fallback (hx/gx may differ from MATLAB). Error: $e"
        idx_stable = findall(slt)
        idx_unstable = findall(.!slt)
        perm = [idx_stable; idx_unstable]
        s_ordered = Matrix(s)[perm, perm]
        t_ordered = Matrix(t)[perm, perm]
        Z_ordered = Matrix(Z)[:, perm]
        z11 .= Z_ordered[1:nk, 1:nk]
        z21 .= Z_ordered[(nk + 1):end, 1:nk]
        s11 .= s_ordered[1:nk, 1:nk]
        t11 .= t_ordered[1:nk, 1:nk]
    end

    # Checks
    z11_rank = rank(z11)
    @info "SGU_solver: z11 size=$(size(z11)), rank=$(z11_rank), nk=$nk"
    if z11_rank < nk
        @warn "invertibility condition violated: rank(z11)=$(z11_rank) < nk=$nk, using pinv"
        z11i = pinv(z11)
    else
        z11i = z11 \ Matrix{Float64}(I, nk, nk)
    end
    gx = real(z21 * z11i)
    hx = real(z11 * (s11 \ t11) * z11i)

    @info "SGU_solver: done."
    return hx, gx, F1, F2, F3, F4, param
end
