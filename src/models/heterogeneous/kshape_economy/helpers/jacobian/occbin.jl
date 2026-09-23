function occbin(m::mBBQ)
    Tmax = 3 #setting for n periods for ZLB duration
    nvars = #system matrices
    Astarbar

    return occbin(m, Tmax, nvars, Astar, Bstar, Cstar, D_ZLB, J_ZLB, decrulea)
end


function occbin( 
        Tmax::Int64, nvars::Int64, Astar, Bstar, Cstar, D_ZLB, J_ZLB, decrulea)

    Ps = zeros(nvars, nvars, Tmax)
    Ds = zeros(nvars, Tmax)

    # Terminal condition at Tmax (model exits to reference regime at Tmax+1)
    invmat          = inv(Astarbarmat * decrulea + Bstarbarmat)
    Ps[:, :, Tmax]  = -invmat * Cstarbarmat
    Ds[:, Tmax]     = -invmat * D_ZLB

    # Backward recursion from Tmax-1 to 1
    for ii = Tmax-1:-1:1
        invmat         = inv(Astarbarmat * Ps[:, :, ii+1] + Bstarbarmat)
        Ps[:, :, ii]   = -invmat * Cstarbarmat
        Ds[:, ii]      = -invmat * (Astarbarmat * Ds[:, ii+1] + D_ZLB)
    end

    E   = -invmat * J_ZLB
    P_1 = Ps[:, :, 1]
    D_1 = Ds[:, 1]
    E_1 = E




    return Ps, Ds, E, P_1, D_1, E_1


end
