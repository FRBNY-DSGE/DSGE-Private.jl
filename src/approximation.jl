"""
Approximation

The `Approximation` type defines the structure of the approximation methods used in global solutions nonlinear (rather than log-linearized) DSGE models. It includes methods to approximate functions with efficient, anistropic Smoylak sparse grids as well as methods to approximate integrals using Gauss-Hermite quadrature. The domain on which to approximate functions and exogenous shock values is determined by solving the linearized version of the model and simulating many paths starting from the steady state. This determines the high-probability region of the domain, which is scaled to a unit hypercube to perform the approximation. The object also holds the necessary information to perform these operations.

### Fields

#### Function Approximation
* `nfunc::Int`: Number of functions to approximate
* `nmsv::Int`: Number of endogenous variables in the minimum state variable representation (we assume model is solved using a projection method--see page 10 of Gust et. al (2017)--otherwise set equal to number of endogenous variables)
* `nendogvars::Int`: Number of endogenous variables
* `ngridpoints::Int`: Number of grid points on the Smolyak grid (see Judd et. al (2014), especially section 2)
* `nindplus::Int`: Number of functions with higher approximation level
* `indplus::Array{Int, 1}`: Index of functions that get hhigher approximation level
* `nshockgrid::Array{Int ,1}`: Number of possible values each shock can take on
* `xgrid::Array{Float64, 2}`: Value of each minimum state variable (transformed to [-1, 1] domain) at each Smolyak grid point
* `nexogvars::Int`: Number of enxogenous variables
* `nexogshocks::Int`: Number of exogenous shocks
* `ns::Int`: Number of exogenous states, equal to the number of distinct combinations of shock values
* `ninter::Int`: Number of points used to calculate interpolated value of shock
* `bbt::Array{Float64, 2}`: Transposed matrix of Smolyak basis functions evaluated at each Smolyak grid point
* `bbtinv::Array{Float64, 2}`: Inverse of bbt
* `slopeconxx::Array{Float64, 1}`: Slope and constant to convert from [-1, 1] domain of endogenous variables to normal domain
* `slopeconmsv::Array{Float64, 1}`: Slope to convert from normal domain of endogenous variables to [-1, 1] domain
* `shockbounds::Array{Float64, 2}`: Minimum and maximum values we impose on shocks
* `shockdistance::Array{Float64, 1}`: Distance between shock values on the exogenous shock grid
* `exoggrid :: Array{Float64, 2}`: Grid containing the values of each shock at each exogenous state
* `interpolatemat :: Array{Int, 2}`: Grid to contain combinations of exogenous states to use in interpolation of shock values

#### Integral Approximation
* `nquad::Int`: Number of quadrature points
* `ghnodes::Array{Float64, 2}`: Gauss-Hermite nodes for integrating over shocks
* `ghweights::Array{Float64, 1}`: Gauss-Hermite weights for integrating over shocks

#### Linear Simulation
* `endog_emean::Array{Float64, 1}`: Ergodic mean of endogneous variables resulting from linear simulation
* `zlbfrequency::Float64`: Frequency of reaching zero lower bound during linear simulation
* `statezlbinfo::Array{Int64, 1}`: Holds whether a given exogenous state was at zero lower bound
"""
mutable struct Approximation

    nfunc::Int
    nmsv :: Int
    nindplus :: Int
    indplus :: Array{Int, 1}
    nshockgrid :: Array{Int, 1}
    xgrid::Array{Float64, 2}

    nendogvars::Int
    ngridpoints::Int
    nexogvars::Int
    nexogshocks::Int
    ns::Int
    ninter::Int

    bbt::Array{Float64, 2}
    bbtinv::Array{Float64, 2}

    slopeconxx::Array{Float64, 1}
    slopeconmsv::Array{Float64, 1}
    shockbounds::Array{Float64, 2}
    shockdistance::Array{Float64, 1}
    exoggrid::Array{Float64, 2}
    interpolatemat::Array{Int, 2}

    nquad::Int
    ghnodes::Array{Float64, 2}
    ghweights::Array{Float64, 1}

    endog_emean::Array{Float64, 1}
    zlbfrequency::Float64
    msvbounds::Array{Float64, 1}
    statezlbinfo::Array{Int, 1}

end

"""
    Approximation()
Arguments:

Description:
Initializes a default Apprxomation object for working with the model in Gust et. al (2017)
"""
function Approximation()

    #Initialize empty approximation object
    approx = Approximation(0,0,0,[0],[0],[0. 0.],
                                    0,0,0,0,0,0,
                                    [0. 0.], [0. 0.],
                                    [0.],[0.],[0. 0.],[0.],[0. 0.],[0. 0.],
                                    0, [0. 0.], [0.],
                                    [0.],0.,[0.],[0])
    init_settings!(approx)
    init_solution!(approx)
    return approx
end

"""
    init_settings!(approx::Approximation)

# Arguments:
- `approx::Approximation`: Approximation object

# Description:
Initializes number of exogenous, endogenous, and minimum state variables as well as which functions receive higher order apprxomations and how many values each shock will take on in teh shock grid to defaults (those in Gust et. al (2017)).
"""
function init_settings!(approx::Approximation)

    approx.nexogvars = 6
    approx.nendogvars = 22
    approx.nmsv = 7
    approx.nfunc = 7
    approx.nindplus = 1
    approx.nshockgrid = [7,2,2,2,2,1]# CHANGE TO THIS AFTER DONE TESTING: [7,3,3,3,3,1]
    approx.indplus = [3]
end

"""
setgridsize(nexogvars::Int,nshockgrid::Array{Int})

# Arguments:
- `nexogvars::Int`: Number of exogenous variables including those shocks held constant.
- `nshockgrid::Array{Int,1}`: Vector containing number of values each shock can take on in the exogenous variables grid

# Description:
Sets grid size for exogenous shocks.

# Returns:
- `nexogshocks`: Number of non-constant shocks.
- `ninter`:  Number of points used to calculate interpolated value of shock
- `ns`: Total number of exogenous states.
...
"""
function setgridsize(nexogvars::Int,nshockgrid::Array{Int,1})

    # Counts how many elements of nshockgrid are strictly greater than one
    nexogshocks = count(x -> x > 1, nshockgrid)

    # Each shock can be interpolated using two points, so the total number of ways to interpolate  is  2^nexogshocks
    ninter = 2^nexogshocks

    # The number of possible shock combinations is equal to the product of the number of values each shock takes on
    ns = prod(nshockgrid)

    return nexogshocks,ninter,ns
end

"""
    gen_exoggrid_indices(nshockgrid::Array{Int, 1},nexogvars::Int,ns::Int)

# Arguments:
- `nshockgrid::Array{Int, 1}`: Array containing number of shock values in the grid for each of the nexogshocks shocks.
- `nexogvars::Int`: Number of exogenous variables including shocks -- must be greater than nexogshocks.
- `ns::Int`: Total number of grid points, equals the product of elements in nshockgrid since this is the number of possible shock value combinations.

# Description:
For each exogenous shock, this associates a grid point with the index of the value that that shock takes on at that grid point. In total there are 'ns' such points, with each point representing a distinct combination of shock values.
"""
function gen_exoggrid_indices(nshockgrid::Array{Int,1},nexogvars::Int,ns::Int)

    #Initialize Variables
    exoggridindex = zeros(Int,nexogvars,ns)
    blocksize = 1 #This will record the number of possible shock value combinations

    # For each exogenous shock
    for i in nexogvars:-1:1

        # Split up the ns possible shock value combinations into groups. A group contains all distinct combination of shock values among the shocks that have already been looped through (inclusive of the current shock)
        ngroups = ns ÷ (blocksize*nshockgrid[i])

        # For each group
        for j in 1:ngroups

            # Break into 'nshockgrid[i]' blocks of size 'blocksize', with each block containing all distinct combinations of shock values among previous shocks (exclusive of current shock)
            # Each block is assigned a distinct value from the possible values of the current shock, so that the end result is that each column represents a distinct combination of shock values and taken together all columns represent all possible distinct combinations
            for k in 1:nshockgrid[i]
                for l in blocksize*(k-1)+1:blocksize*k
                    exoggridindex[i,nshockgrid[i]*blocksize*(j-1)+l] .= k
                end
            end
        end

        #The number of possible shock value combinations increases multiplicatively by the number of shock values a shock can take on
        blocksize *= nshockgrid[i]
    end

    return exoggridindex

end

"""
    ghquadrature(nquadsingle::Int64,nexogvars::Int64)

# Arguments:
- `nquadsingle::Int64`: Number of univariate quadrature points (must be 3,5 or 7).
- `nexogvars::Int64`: Number of exogenous variables.

# Description:
Produces gauss-hermite quadrature weights and nodes for mulivariate case from univariate case.
Returns total number of quadrature nodes, multivariate quadrature nodes, and multivariate quadrature weights.
"""
function ghquadrature(nquadsingle::Int64,nexogvars::Int64)

    # Initialize Variables
    quadnodes_s=zeros(nquadsingle)
    quadweights_s=zeros(nquadsingle)
    ghnodes=zeros(nexogvars,nquadsingle^nexogvars)
    ghweights_mat=zeros(nexogvars,nquadsingle^nexogvars)
    ghweights=Array{Float64, 1}(undef,nquadsingle^nexogvars)
    const_pi = 3.14159265358979323846
    quadnodes_s,quadweights_s=gausshermite(nquadsingle)

    nquad = nquadsingle^nexogvars
    blocksize = 1
    for ie = nexogvars:-1:1
        if (ie == nexogvars)
            blocksize = 1
        else
            blocksize *= nquadsingle
        end
        ncall = div(nquad,nquadsingle*blocksize)
        for ic in 1:ncall
            for ib in 1:nquadsingle
                left=nquadsingle*blocksize*(ic-1)+blocksize*(ib-1)+1
                right=nquadsingle*blocksize*(ic-1)+blocksize*ib
                ghnodes[ie,left:right] .= sqrt(2)*quadnodes_s[ib]
                ghweights_mat[ie,left:right] .= quadweights_s[ib]
            end
        end
    end

    ghweights = vec((1.0/const_pi)^(nexogvars/2.0)*prod(ghweights_mat,dims=1)) # product of ghweights_mat along the first dimension

    return nquad,ghnodes,ghweights

end

"""
    smolyakpoly(nmsv::Int,ngridpoints::Int,nindplus::Int,indplus::Array{Int},xx::Array{Float64})

# Arguments:
- `nmsv::Int`: Minimum number of state variables.
- `ngridpoints::Int`: Number of grid points.
- `nindplus::Int`: Number of variables to include up to fourth order in polynomial
- `indplus::Array{Int, 1}1: Indicator array for variables that go to fourth order.
- `xx::Array{Float64, 1}`: Endogenous state variables defined over [-1,1] domain.

# Description:
Returns vector of Smolyak basis polynomials evaluated at given values of endogenous variables.
"""
function smolyakpoly(nmsv::Int,ngridpoints::Int,nindplus::Int,indplus::Array{Int,1},xx::Array{Float64,1})

    # Initialize variables
    smolyakpoly=Array{Float64}(undef,ngridpoints)

    # First polynomial is a constant
    smolyakpoly[1] = 1.0

    # Every function approximation has at least two basis functions (first/second order Chebyshev polynomials)
    for i in 1:nmsv
	smolyakpoly_aux = xx[i]
        smolyakpoly[2*i] = smolyakpoly_aux
        smolyakpoly[2*i+1] = 2.0*(smolyakpoly_aux)^2-1.0
    end

    # Some functions get two additional basis functions (third/fourth order Chebyshev polynomials)
    for i in 1:nindplus
	xx_aux =xx[indplus[i]]
        smolyakpoly[2*nmsv+2*(i-1)+2] = 4.0*xx_aux^3-3.0*xx_aux
        smolyakpoly[2*nmsv+2*(i-1)+3] = 8.0*xx_aux^4-8.0*xx_aux^2+1.0
    end

    return smolyakpoly

end

"""
    sparsegrid(nmsv::Int,nindplus::Int,ngridpoints::Int,indplus::Array{Int})

# Arguments:
- `nmsv::Int`: Minimum number of state variables.
- `nindplus::Int`: Number of variables to include up to fourth order in polynomial
- `ngridpoints::Int`: Number of grid points.
- `indplus::Array{Int, 1}`: Indicator array for variables that go to fourth order.

# Description:
Returns Smolyak grid points and Smolyak matrix bbt and its inverse. bbt is the matrix of Smolyak basis functions and xgrid the matrix of grid points.

The bb matrix contains the basis functions evaluated at one point along its rows.
We compute its transpose bbt since we will use it to grab points in column major form later.
We also need the inverse of bbt to update the polynomial coefficients.
"""
function sparsegrid(nmsv::Int,nindplus::Int,ngridpoints::Int,indplus::Array{Int,1})

    #Initilize Variables
    xgrid = zeros(nmsv,ngridpoints)
    bbt = zeros(ngridpoints,ngridpoints)

    for i in 1:nmsv
        xgrid[i,2*i] = -1.0
        xgrid[i,2*i+1] = 1.0
    end

    for i in 1:nindplus
        xgrid[indplus[i],2*nmsv+2*(i-1)+2] = -1.0/sqrt(2.0)
        xgrid[indplus[i],2*nmsv+2*(i-1)+3] = 1.0/sqrt(2.0)
    end

    #form bbt matrix
    for i in 1:ngridpoints
	bbt_aux = smolyakpoly(nmsv,ngridpoints,nindplus,indplus,xgrid[:,i])
        bbt[:,i] = bbt_aux
    end


    # find bbt inverse matrix
    bbtinv = copy(bbt)
    bbtinv,ipiv,info=LinearAlgebra.LAPACK.getrf!(bbtinv)
    if (info == 0)
        LinearAlgebra.LAPACK.getri!(bbtinv,ipiv)
    else
        println("something went wrong with getrf! (sparsegrid)")
        println("info = ", info)
    end

    return xgrid,bbt,bbtinv

end

"""
    init_solution!(approx::Approximation)

# Arguments:
`approx::Approximation`: Approximation object

# Description:
Calculates the inital values for most of the values in the Approximation object.
"""
function init_solution!(approx::Approximation)

    nquadsingle = 3
    approx.ngridpoints = 2*(approx.nmsv)+2*approx.nindplus+1

    #set nexogshocks,ns
    approx.nexogshocks,approx.ninter,approx.ns=setgridsize(approx.nexogvars,approx.nshockgrid)
    approx.nquad = nquadsingle^(approx.nexogshocks)

    #get matrix used for interpolating the shocks
    interpolatemat=Array{Int64}(undef,approx.nexogshocks,2^approx.nexogshocks)
    blocksize = 1
    for i in approx.nexogshocks:-1:1
        ngroups = 2^(approx.nexogshocks-1) ÷ blocksize
        for j in 1:ngroups
            for k = 1:2
                for l in 2*blocksize*(j-1)+blocksize*(k-1)+1:2*blocksize*(j-1)+blocksize*k
                    interpolatemat[i,l] = k-1
                end
            end
        end
        blocksize *= 2
    end
    approx.interpolatemat = interpolatemat

    #get quadrature nodes and weights
    approx.nquad,approx.ghnodes,approx.ghweights=ghquadrature(nquadsingle,approx.nexogshocks)

    #construct sparse grid, bb matrix and its inverse
    approx.xgrid, approx.bbt, approx.bbtinv=sparsegrid(approx.nmsv,approx.nindplus,approx.ngridpoints,approx.indplus)

    approx.slopeconmsv = Array{Float64}(undef,2*approx.nmsv)
    approx.shockbounds = Array{Float64}(undef,approx.nexogshocks,2)
    approx.shockdistance = Array{Float64}(undef,approx.nexogshocks)
    approx.exoggrid = Array{Float64}(undef,approx.nexogvars,approx.ns)

    return nothing

end
