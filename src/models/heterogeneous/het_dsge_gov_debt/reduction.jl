function normalize(m::HetDSGEGovDebt, JJ::Matrix{Float64})

    Qx, Qy, Qleft, Qright  = compose_normalization_matrices(m)

    m <= Setting(:n_predetermined_variables, size(Qx, 1))
    m <= Setting(:Qx, Qx)
    m <= Setting(:Qy, Qy)

    Jac1 = Qleft*JJ*Qright

    return Jac1
end

# use proxy distributions method - now `averaging' so that Qx'*Qx should not "add mass"
# Basically, maps an `n` length vector to a smaller vector by binning grid points into new bins of size binsize
# (with a size of 1 being an `n` to `n` mapping)
# TODO: Refactor to use Kronecker and Diagonal, and BlockBandedMatrix
function avg_prox(n::Int, binsize::Int = 3, minn_noag::Int = 0)
    minn_noag_use = min(minn_noag,n)
    n_gps = Int(floor((n-minn_noag_use)/binsize)) # number of groups, binsize is the size of the bins we reallocate distribution to
    n_noag = n - binsize*n_gps # Leftover points after rebinning distribution, assumed these are at the left end of the grid
    Prox = cat(eye(n_noag), kron(eye(n_gps),fill(1 ./ sqrt(binsize), 1, binsize)), dims = [1 2])
    n_new = n_noag + n_gps
    return (Prox, n_new, n_noag, n_gps)
end
# note: setting minn_noag = 0 seems to work the best

# TODO: use SparseArrays or something like that here; can use QR on it
function make_S(n::Int, n_noag::Int = 0, binsize::Int = 1)
    P1 = ones(n,1)*sqrt(binsize)
    P1[1:n_noag] .= 1. # set unbinned points to 1
    Ptemp = eye(n)
    Ptemp = Ptemp[:,2:end] # Remove the first column of the identity matrix, and replace w/ P1
    P2 = Ptemp
    P = [P1 P2]
    (QQQ,Rjunk)=qr(P) # Get eigenvalues
    S         = QQQ[:,2:end]'
    return S
end

# TODO: USE SPARSE MATRICES/BANDED MATRICES TO IMPLEMENT THE REDUCTION, KRONECKER
function compose_normalization_matrices(m::HetDSGEGovDebt)
    if get_setting(m, :poor_man_reduc)
        na = get_setting(m, :na1_state) #:na) # Want the na1_state, not na, b/c want dimension of low skill cash on hand
        ns = get_setting(m, :ns)
        n = na * ns #get_setting(m, :na1) + get_setting(m, :na2) #
        nscalars = get_setting(m, :nscalars)
        nyscalars = get_setting(m, :nyscalars)
        nascalars = get_setting(m, :nascalars)
        mindens = get_setting(m, :mindens)
        D = m[:Dstar].value

        na1 = maximum(findall(D[1:na] .> mindens)) # Chop off unneeded cash-on-hand grid points for low-skill workers b/c no one there
        m <= Setting(:na1_state, na1)
        m <= Setting(:na1_jump, na1)

        update_reduced_indices!(m)

        Reduc = eye(n) # use Diagonal or sparse matrix?
        Reduc = Reduc[[1:na1;na+1:n],:] # Remove eliminated cash-on-hand grid points for low skill workers

        minn_noag = 0
        binsize = get_setting(m, :binsize) # this is the maximum binsize which seemed to leave the IRFs unchanged, you can experiment with this
        # Note that setting binsize = 1 should return what we had before
        (ProxL, n_newL, n_noagL, n_gpsL) = avg_prox(na1, binsize, minn_noag) # Get new indices corresponding to binning
        (ProxH, n_newH, n_noagH, n_gpsH) = avg_prox(na, binsize, minn_noag)   # reduction of distributions
        m <= Setting(:na1_state, n_newL)
        m <= Setting(:na2_state, n_newH)
        if get_setting(m, :reduce_ell) # Reduce ell by using binning
            m <= Setting(:na1_jump, n_newL)
            m <= Setting(:na2_jump, n_newH)
        end
        update_reduced_indices!(m)

        # S is made from an orthogonal matrix, so further reduction by projecting onto an orthogonal basis?
        Prox = cat(ProxL, ProxH, dims = [1 2]) # TODO: Create a BlockBanded Matrix here rather than cat
        SL = make_S(n_newL, n_noagL, binsize)
        SH = make_S(n_newH, n_noagH, binsize)
        S = cat(SL, SH, dims = [1 2]) # TODO: Create a BlockBanded Matrix here rather than cat

        # TODO: Create a BlockBanded Matrix here rather than cat
        if get_setting(m, :reduce_ell)
            Qleft     = cat(Prox*Reduc,S*Prox*Reduc,eye(nscalars), dims = [1 2]) # ell is in the first Reduc, so we add
            Qx        = cat(S*Prox*Reduc,eye(nascalars), dims = [1 2])           # an Prox * Reduc to further reduce
            Qy        = cat(Prox*Reduc,eye(nyscalars), dims = [1 2])
        else
            Qleft     = cat(Reduc,S*Prox*Reduc,eye(nscalars), dims = [1 2])
            Qx        = cat(S*Prox*Reduc,eye(nascalars), dims = [1 2]) # same as if reducing ell b/c just applies to distribution
            Qy        = cat(Reduc,eye(nyscalars), dims = [1 2])
        end

        Qright    = cat(Qx',Qy',Qx',Qy', dims = [1,2])

        return Qx, Qy, Qleft, Qright # Qx, Qy are the individual components of Qleft, Qright
    else
        na = get_setting(m, :na)
        ns = get_setting(m, :ns)
        nscalars = get_setting(m, :nscalars)
        nyscalars = get_setting(m, :nyscalars)
        nascalars = get_setting(m, :nascalars)

        # Create PPP matrix
        P1 = kron(Matrix{Float64}(I, ns,ns),ones(na,1))
        Ptemp = Matrix{Float64}(I, na, na)
        Ptemp = Ptemp[:, 2:end]
        P2 = kron(Matrix{Float64}(I, ns, ns), Ptemp)
        P  = hcat(P1, P2)

        Q,R = qr(P)
        Q = Array(Q)
        S         = Q[:, ns+1:end]'

        nans = na*ns

        Qleft     = cat(Matrix{Float64}(I, nans, nans),S,Matrix{Float64}(I, nscalars, nscalars), dims = [1 2])
        Qx        = cat(S,Matrix{Float64}(I, nascalars, nascalars), dims = [1 2])
        Qy        = cat(Matrix{Float64}(I, nans, nans),Matrix{Float64}(I, nyscalars, nyscalars), dims = [1 2])
        Qright    = cat(Qx',Qy',Qx',Qy', dims = [1,2])

        return Qx, Qy, Qleft, Qright

    end
end

function truncate_distribution!(m::HetDSGEGovDebt) #, nt′::NamedTuple = NamedTuple(), nt::NamedTuple = NamedTuple())
#=    @assert ((isempty(nt′) && isempty(nt)) || (!isempty(nt′) && !isempty(nt))) "NamedTuple inputs to truncate_distribution! " *
             "must either both be empty or both nonempty"=#
    if get_setting(m, :trunc_distr)
        mindens = get_setting(m, :mindens)
        rescale_weights = get_setting(m, :rescale_weights)

        na = get_setting(m, :na)
        D = m[:Dstar].value
        ell = m[:lstar].value
        c = m[:cstar].value
        agrid = m.grids[:agrid].points
        swts::Vector{Float64}  = m.grids[:sgrid].weights

        oldna = na
        na = maximum(findall(D[1:na]+D[na+1:2*na] .> mindens)) # used to be 1e-8
        m[:Dstar] = D[[1:na; oldna+1:oldna+na]]
        m[:lstar] = ell[[1:na; oldna+1:oldna+na]]
        m[:cstar] = c[[1:na; oldna+1:oldna+na]]
        if rescale_weights
            ahi = agrid[na]
            alo = agrid[1]
            ascale = ahi-alo
        end
        m <= Setting(:na, na)
        m <= Setting(:ahi, ahi)
        m <= Setting(:ascale, ascale)
        m.grids[:agrid] = Grid(uniform_quadrature(ascale), alo, ahi, na, scale = ascale)
        m.grids[:weights_total] = kron(swts, m.grids[:agrid].weights)
        nans = na*get_setting(m, :ns)
        m <= Setting(:n, nans)
        m <= Setting(:na, na)

        update_reduced_indices!(m)
    end
end

function update_reduced_indices!(m::HetDSGEGovDebt)
    setup_indices!(m) # Update the indices
    init_states_and_jumps!(m, get_setting(m, :states), get_setting(m, :jumps)) # Update mapping from states/jumps to indices
    normalize_model_state_indices!(m)
    endogenous_states_augmented = [:C_t1]
    for (i,k) in enumerate(endogenous_states_augmented) # Augment the model w/post-steady-state states
        m.endogenous_states_augmented[k] = i +
            first(m.endogenous_states[get_setting(m, :jumps)[end]]) # first(collect(values(m.endogenous_states))[end])
    end
    m <= Setting(:n_model_states_augmented, get_setting(m, :n_model_states) +
                 length(m.endogenous_states_augmented)) # Update number of states (incl. augmented states)
end
