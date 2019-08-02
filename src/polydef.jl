#=module polydef
    import FastGaussQuadrature: gausshermite
    export SmolyakApproximation, init_solution!, init_settings!, setgridsize,
        exoggridindex, ghquadrature,sparsegrid, smolyakpoly, initializelinearsolution!
using LinearAlgebra
=#
mutable struct SmolyakApproximation{T}

 # These should be settings
    number_shock_values :: Int
    nfunc::Int
    nmsv :: Int
    nvars :: Int
    ngrid :: Int
    nparams :: Int
    nindplus :: Int
    nexog :: Int
    nexogshock :: Int
    nexogcont :: Int
    ns :: Int
    nsexog :: Int
    ninter :: Int
    nquad :: Int
    zlbswitch :: Bool
    indplus :: Array{Int, 1}
    nshockgrid :: Array{Int, 1}

    exogvarinfo::Array{Int64, 2}
    xgrid::Array{Float64, 2}
    bbt::Array{Float64, 2}
    bbtinv::Array{Float64, 2}
    startingguess::Bool

    interpolatemat :: Array{Int, 2}
    slopeconmsv :: Array{Float64, 1}
    shockbounds :: Array{Float64, 2}
    shockdistance :: Array{Float64, 1}
    exoggrid :: Array{Float64, 2}
    ghnodes :: Array{Float64, 2}
    ghweights :: Array{Float64, 1}

    # simulate_linear
    endog_emean :: Array{Float64, 1}
    zlbfrequency :: Float64
    msvbounds :: Array{Float64, 1}
    statezlbinfo :: Array{Int64, 1}
    convergence :: Bool
    slopeconxx :: Array{Float64, 1}
end

"""
    SmolyakApproximation()
Initializes the values in the SmolyakApproximation object to defaults
"""
function SmolyakApproximation()
    #Initialize empty approximation object
    approx = SmolyakApproximation{Float64}(0,0,0,0,0,0,0,0,0,0,0,0,0,0,false,[0],[0],
                                           [0 0], [0. 0.], [0. 0.], [0. 0.], false, [0. 0.],
                                           [0 0],[0.],[0. 0.],[0.],[0. 0.],[0. 0.],[0.],
                                           [0.],0.,[0.],[0.],false,[0.])
    init_settings!(approx)
    init_solution!(approx)
    return approx
end

"""
    init_settings!(approx::SmolyakApproximation)
Initializes some of the values in the model's SmolyakApprox type to desired values.
"""
function init_settings!(approx::SmolyakApproximation)

    approx.nexog = 6
    approx.nexogcont = 0
    approx.nvars = 22
    approx.nmsv = 7
    approx.nfunc = 7
    approx.nindplus = 1
    approx.nshockgrid = [7,2,2,2,2,1]# CHANGE TO THIS AFTER DONE TESTING: [7,3,3,3,3,1]
    approx.indplus = [3]
end

"""
setgridsize(nexog::Int,nshockgrid::Array{Int})

Sets grid size for exogenous shocks.
# Returns:
    nexogshock: Number of active shocks (with states greater than one).
    ns: Total number of grid points.
    number_shock_values: Values of shocks at the gridpoints.
...
# Arguments
- `nexog::Int`: Number of exogenous variables including those shocks held constant.
- `nshockgrid::Array{Int,1}`: Vector containing grid size for each shock.
...
"""
function setgridsize(nexog::Int,nshockgrid::Array{Int,1})

    nexogshock = 0
    for i in 1:nexog
        if (nshockgrid[i] > 1)
            nexogshock = nexogshock + 1
        else
            break
        end
    end
    ninter = 2^nexogshock

    #allocate matrices for shock processes
    ns = 1
    number_shock_values = 0
    for i in 1:nexogshock
        ns = ns*nshockgrid[i]
        number_shock_values = number_shock_values + nshockgrid[i]
    end

    return nexogshock,ninter,ns,number_shock_values

end

"""
    exoggridindex(nshockgrid::Array{Int},nexog::Int,ns::Int)

For each exogenous shock, this associates a given grid point with the value that that shock takes on at that grid point.
...
# Arguments
- `nshockgrid::Array{Int, 1}`: Array containing number of shock values in the grid for each of the nexogshock shocks.
- `nexog::Int`: Number of exogenous variables including shocks -- must be greater than nexogshock.
- `ns::Int`: Total number of grid points, equals the product of elements in nshockgrid since this is the number of possible shock value combinations.
...
"""
function exoggridindex(nshockgrid::Array{Int,1},nexog::Int,ns::Int)

    #Initilize Variables
    exoggridindex = zeros(Int,nexog,ns)
    blocksize = 1 #This will record the number of possible shock value combinations

    # For each exogenous shock
    for i in nexog:-1:1

        # Split up the ns possible shock value combinations into groups, with each group
        # representing a distinct combination of shock values among the shocks that have already
        # been looped through and the current shock
        ngroups = ns/(blocksize*nshockgrid[i])

        for j in 1:ngroups

            #For each value of the shock
            for k in 1:nshockgrid[i]

                #START HERE NEXT TIME
                exoggridindex[i,nshockgrid[i]*blocksize*(j-1)+blocksize*(k-1)+1:nshockgrid[i]*blocksize*(j-1)+blocksize*k] .= k
            end
        end

        blocksize *= nshockgrid[i] #The number of possible shock value combinations increases multiplicatively by the number of shock values this shock can take on
    end

    return exoggridindex

end

"""
    ghquadrature(nquadsingle::Int64,nexog::Int64)

Produces gauss-hermite quadrature weights and nodes for mulivariate case from univariate case.
Returns total number of quadrature ndoes, multivariate quadrature nodes, and multivariate quadrature weights.
...
# Arguments
- `nquadsingle::Int64`: Number of univariate quadrature points (must be 3,5 or 7).
- `nexog::Int64`: Number of shocks.
...
"""
function ghquadrature(nquadsingle::Int64,nexog::Int64)

    # Initilize Variables
    quadnodes_s=zeros(nquadsingle)
    quadweights_s=zeros(nquadsingle)
    ghnodes=zeros(nexog,nquadsingle^nexog)#Not sure I should make this zeros
    ghweights_mat=zeros(nexog,nquadsingle^nexog)#Not sure I should make this zeros
    ghweights=Array{Float64, 1}(undef,nquadsingle^nexog)#Not sure I should make this zeros
    #const const_pi = 3.14159265358979323846
    const_pi = 3.14159265358979323846

    quadnodes_s,quadweights_s=gausshermite(nquadsingle) ##

    nquad = nquadsingle^nexog
    blocksize = 1 #Must Initilize blocksize
    for ie = nexog:-1:1
        if (ie == nexog)
            blocksize = 1
        else
            blocksize = blocksize*nquadsingle # it was nquadsingle*blocksize ??
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

    ghweights = vec((1.0/const_pi)^(nexog/2.0)*prod(ghweights_mat,dims=1)) # product of ghweights_mat along the first dimension

    return nquad,ghnodes,ghweights

end

"""
    smolyakpoly(nmsv::Int,ngrid::Int,nindplus::Int,indplus::Array{Int},xx::Array{Float64})

Returns vector of Smolyak polynomials.
...
# Arguments
- `nmsv::Int`: Minimum number of state variables.
- `ngrid::Int`: Number of grid points.
- `nindplus::Int`: Number of variables to include up to fourth order in polynomial
- `indplus::Array{Int, 1}1: Indicator array for variables that go to fourth order.
- `xx::Array{Float64, 1}`: Endogenous state variables defined over [-1,1] domain.
...
"""
function smolyakpoly(nmsv::Int,ngrid::Int,nindplus::Int,indplus::Array{Int,1},xx::Array{Float64,1})

    # Initilize Variables
    smolyakpoly=Array{Float64}(undef,ngrid)

    smolyakpoly[1] = 1.0
    for i in 1:nmsv
	smolyakpoly_aux = xx[i]
        smolyakpoly[2*i] = smolyakpoly_aux
        smolyakpoly[2*i+1] = 2.0*(smolyakpoly_aux)^2-1.0
    end

    for i in 1:nindplus
	xx_aux =xx[indplus[i]]
        smolyakpoly[2*nmsv+2*(i-1)+2] = 4.0*xx_aux^3-3.0*xx_aux
        smolyakpoly[2*nmsv+2*(i-1)+3] = 8.0*xx_aux^4-8.0*xx_aux^2+1.0
    end

    return smolyakpoly

end

"""
    sparsegrid(nmsv::Int,nindplus::Int,ngrid::Int,indplus::Array{Int})

Returns Smolyak grid points and Smolyak matrix bbt and its inverse. bbt is the matrix of Smolyak basis functions and xgrid the matrix of grid points.

The bb matrix (in JMM 2013) contains the basis functions evaluated at one point along its rows.
We compute its transpose (bbt) since we will use it to grab points in column major form later.
We also need the inverse of bbt to update the polynomial coefficients.
...
# Arguments
- `nmsv::Int`: Minimum number of state variables.
- `nindplus::Int`: Number of variables to include up to fourth order in polynomial
- `ngrid::Int`: Number of grid points.
- `indplus::Array{Int, 1}`: Indicator array for variables that go to fourth order.
...
"""
function sparsegrid(nmsv::Int,nindplus::Int,ngrid::Int,indplus::Array{Int,1})

    #Initilize Variables
    xgrid = zeros(nmsv,ngrid)
    bbt = zeros(ngrid,ngrid)

    for i in 1:nmsv
        xgrid[i,2*i] = -1.0
        xgrid[i,2*i+1] = 1.0
    end

    for i in 1:nindplus
        xgrid[indplus[i],2*nmsv+2*(i-1)+2] = -1.0/sqrt(2.0)
        xgrid[indplus[i],2*nmsv+2*(i-1)+3] = 1.0/sqrt(2.0)
    end

    #form bbt matrix
    for i in 1:ngrid
	bbt_aux = smolyakpoly(nmsv,ngrid,nindplus,indplus,xgrid[:,i])
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
    init_solution!(approx::SmolyakApproximation)
Calculates the inital values for most of the values in the SmolyakApproximation object.
"""
function init_solution!(approx::SmolyakApproximation) # ! to indicate that this function mutates and input

    #I don't think we need to declare types of these in the future (may not even work)
    nquadsingle = 3

    #put shocks into polynomial approximation if necessary
    nexogadj = approx.nexog - approx.nexogcont
    nmsvadj = approx.nmsv + approx.nexogcont

    approx.ngrid = 2*(approx.nmsv+approx.nexogcont)+2*approx.nindplus+1

    #set nexogshock,ns, and number_shock_values
    nexogshock,ninter,ns,number_shock_values=setgridsize(nexogadj,approx.nshockgrid)
    approx.nquad = nquadsingle^(approx.nexogshock+approx.nexogcont)

    approx.nexogshock = nexogshock
    approx.ninter = ninter
    approx.ns = ns
    approx.number_shock_values = number_shock_values

    #Set exogvarinfo
    exogvarinfo = Array{Int64}(undef,nexogadj,approx.ns)
    exogvarinfo[1:approx.nexogshock,:] = exoggridindex(approx.nshockgrid,approx.nexogshock,approx.ns)
    exogvarinfo[approx.nexogshock+1:nexogadj,:] .= 1
    approx.exogvarinfo = exogvarinfo

    #get matrix used for interpolating the shocks
    interpolatemat=Array{Int64}(undef,approx.nexogshock,2^approx.nexogshock)
    blocksize = 1
    for i in approx.nexogshock:-1:1
        if (i == approx.nexogshock)
            blocksize = 1
        else
            blocksize = 2*blocksize
        end
        #blocksize=2^(approx[:nexogshock]-i)
        ncall = div(2^(approx.nexogshock-1),blocksize)
        for j in 1:ncall
            for k = 1:2
                left=2*blocksize*(j-1)+blocksize*(k-1)+1
                right=2*blocksize*(j-1)+blocksize*k
                interpolatemat[i,left:right] .= k-1
            end
        end
    end
    approx.interpolatemat = interpolatemat

    #get quadrature nodes and weights
    nquadadj = approx.nexogshock+approx.nexogcont
    nquad,ghnodes,ghweights=ghquadrature(nquadsingle,nquadadj)
    approx.nquad = nquad
    approx.ghnodes = ghnodes
    approx.ghweights = ghweights

    #construct sparse grid, bb matrix and its inverse
    xgrid, bbt, bbtinv=sparsegrid(nmsvadj,approx.nindplus,approx.ngrid,approx.indplus)
    approx.xgrid = xgrid
    approx.bbt = bbt
    approx.bbtinv = bbtinv

    approx.startingguess = false

    approx.slopeconmsv = Array{Float64}(undef,2*nmsvadj)
    approx.shockbounds = Array{Float64}(undef,approx.nexogshock,2)
    approx.shockdistance = Array{Float64}(undef,approx.nexogshock)
    approx.exoggrid = Array{Float64}(undef,nexogadj,approx.ns)

    return

end
#=IGNORE FOR NOW
function initializetestsolution!(test_solution)

    data=Base.DataFmt.readdlm("solution%poly.txt")
    numericData=convert(Array{Int},data[1:end-1,3])
    test_solution.poly.nfunc       =numericData[1]
    test_solution.poly.nmsv        =numericData[2]
    test_solution.poly.nvars       =numericData[3]
    test_solution.poly.ngrid       =numericData[4]
    test_solution.poly.nparams     =numericData[5]
    test_solution.poly.nindplus    =numericData[6]
    test_solution.poly.nexog       =numericData[7]
    test_solution.poly.nexogshock  =numericData[8]
    test_solution.poly.nexogcont   =numericData[9]
    test_solution.poly.ns          =numericData[10]
    test_solution.poly.nsexog      =numericData[11]
    test_solution.poly.ninter      =numericData[12]
    test_solution.poly.nquad       =numericData[13]
    test_solution.poly.zlbswitch   =true

    data=readdlm("solution%poly%ghnodes.txt")
    test_solution.poly.ghnodes=data

    data=readdlm("solution%poly%ghweights.txt")
    test_solution.poly.ghweights=data'

    data=readdlm("solution%poly%indplus.txt")
    test_solution.poly.indplus=data

    data=readdlm("solution%poly%nshockgrid.txt",Int)
    test_solution.poly.nshockgrid=data'

    data=readdlm("solution%poly%interpolatemat.txt")
    test_solution.poly.interpolatemat=data #Somthing may be wrong with this ....? Why is it all zeros?

    data=readdlm("solution%poly%endogsteady.txt")
    test_solution.poly.endogsteady=data # I think some of the values are undefined

    data=readdlm("solution%poly%exoggrid.txt")
    test_solution.poly.exoggrid=data

    data=readdlm("solution%poly%endogsteady.txt")
    test_solution.poly.endogsteady=data

    data=readdlm("solution%poly%shockbounds.txt")
    test_solution.poly.shockbounds=zeros(size(data)) #something is wrong with the .txt file

    data=readdlm("solution%poly%shockdistance.txt")
    test_solution.poly.shockdistance=data

    data=readdlm("solution%poly%slopeconmsv.txt")
    test_solution.poly.slopeconmsv=data
end
=#
#end
