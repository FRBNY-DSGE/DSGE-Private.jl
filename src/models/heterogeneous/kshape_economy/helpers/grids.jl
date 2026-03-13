#TO-DO: FIGURE OUT WHAT'S UP W/ BMIN
function _construct_liquid_asset_grid_mbbq(bmin::Float64, bmax::T, nb::Int) where {T <: Real}
    return exp.(range(0, stop = log(bmax - bmin + 1.), length = nb)) .+ bmin .- 1.
end

function _construct_illiquid_asset_grid_mbbq(amin::T, amax::T, na::Int) where {T <: Real}
    return exp.(range(log(amin + 1.), stop = log(amax + 1.), length = na)) .- 1.
end
function _make_PSS_mbbq(ρ_S::T, σ_S::T, ns::Int) where {T <: Real}
    sgrid, P_SS, bounds = tauchen86(ρ_S, ns-2,; σ=1.0, μ_e=0.0)
    sgrid = sgrid*σ_S / sqrt(1-ρ_S)
    sgrid = exp.(sgrid)
    return P_SS, sgrid
end


function init_grids!(m::mBBQ{T}; coarse::Bool=true) where (T<: Real)
    grids = m.grids
   #this stuff is outdated now 
    bmin = coarse ? get_setting(m, :coarse_bmin) : get_setting(m, :bmin) #this is wrong, bmin shld be a param i think
    bmax = coarse ? get_setting(m, :coarse_bmax) : get_setting(m, :bmax)
    nb   = coarse ? get_setting(m, :coarse_nb)   : get_setting(m, :nb)
    amin = coarse ? get_setting(m, :coarse_amin) : get_setting(m, :amin)
    amax = coarse ? get_setting(m, :coarse_amax) : get_setting(m, :amax)
    na   = coarse ? get_setting(m, :coarse_na)   : get_setting(m, :na)
    ns   = coarse ? get_setting(m, :coarse_ns)   : get_setting(m, :ns)

    # Liquid asset grid
    bgrid = _construct_liquid_asset_grid_mbbq(bmin, bmax, nb)
    bgrid = sort([bgrid; 0])
    grids[:b_grid] = Grid(bgrid, uniform_quadrature(bmin, Float64(bmax), nb; scale=bmax-bmin)[2],
                          Float64(bmax-bmin))
    # Illiquid asset grid
    agrid = _construct_illiquid_asset_grid_mbbq(amin, amax, na)
    #uniform quadrature might be wrong
    grids[:a_grid] = Grid(agrid, uniform_quadrature(amin, amax, na; scale=amax-amin)[2], 
                          Float64(amax-amin)) #maybe amax/amin shld be initialized as floats

    ## Idiosyncratic Income States
    @show m[:ρ_S].value
    @show m[:σ_S].value
    P_SS, sgrid = _make_PSS_mbbq(m[:ρ_S].value, m[:σ_S].value, ns)
    sgrid = sgrid/get_setting(m, :n)

    # Build transition matrix for skill states
    aux1 = [1-get_setting(m, :out_l), get_setting(m, :out_l)-0.002, 0.002] #transition row for low-skill
    aux2 = [get_setting(m, :in_l); 0.002; zeros(ns-4)] #transition col for becoming low-skill
    aux3 = [aux1, zeros(ns-4)] #first row of P_SS, i.e. transition row for low-skill

    P_SS = [aux3; [aux2; P_SS]]
    P_SS = [(P_SS ./ sum(P_SS, dims=2)) .* (1-get_setting(m, :in_s))  get_setting(m, :in_s) .* ones(ns-1, 1);
            get_setting(m, :out_s) .* ones(1, ns-1) ./ (ns-1)  1 - get_setting(m, :out_s)]
    P_SS = P_SS ./ sum(P_SS, dims=2)

    s_dist = ones(length(sgrid)) ./ length(sgrid)

    # Compute stationsry distribution of MC
    for i in 1:1000
        s_dist = P_SS' * s_dist
    end

    s_bar = sum(sgrid .* s_dist)
    s2_bar = sum(sgrid .^ m[:b_aux].value .* s_dist)

    se_grid = hcat(sgrid, (1-m[:ξ].value)/m[:ξ].value * min(get_setting(m, :b_ratio)*sgrid, s_bar), 1) #last col is entrepreneurs
    P_SS2 = kron(I(2), P_SS) #creating the block diagonal
    P_SS2 =  [(P_SS2 * (1-get_setting(m, :in))) repeat(fill(get_setting(m, :in), 1), [2*ns])] #scaling down by prob of becoming entrepreneur
    lastrow = [zeros(ns) ones(ns)*(get_setting(m, :out) / ns) 1-get_setting(m, :out)] #add entrepreneur column
    P_SS2 = [P_SS2; lastrow];

    # Get stationary dist
    s2_dist = ones(2*ns+1) ./ (2*ns+1)
    # Productivity grid
    for i in 1:1000
        s2_dist = P_SS2' * s2_dist
    end
    
    grids[:P_SS2] = P_SS2

end
