include("../../../../zlb-in-julia/jl/src/model/polydef.jl")
using HDF5

nexog = 6
nshockgrid = [7 2 2 2 2 1]
nmsv = 7
nindplus = 1
nexogcont = 0
ngrid = 2*(nmsv+nexogcont) + 2*nindplus + 1
nquadsingle = 3
indplus = [3]

nexogs, ninter, ns, nsv = Main.polydef.setgridsize(nexog,nshockgrid)

exoggridi = Main.polydef.exoggridindex(nshockgrid,nexog,ns)

nquad, ghnodes, ghweights = Main.polydef.ghquadrature(nquadsingle,nexog)

xgrid, bbt, bbtinv = Main.polydef.sparsegrid(nmsv, nindplus, ngrid, indplus)

smolyak = Main.polydef.smolyakpoly(nmsv, ngrid, nindplus, indplus, xgrid[:,4])

h5open("polydef.h5", "w") do file
    @write file nexogs
    @write file ninter
    @write file ns
    @write file nsv
    @write file exoggridi
    @write file nquad
    @write file ghnodes
    @write file ghweights
    @write file xgrid
    @write file bbt
    @write file bbtinv
    @write file smolyak
end

nshockgrid2 = [7 3 3 3 3 2]

nexogs2, ninter2, ns2, nsv2 = Main.polydef.setgridsize(nexog,nshockgrid2)

exoggridi2 = Main.polydef.exoggridindex(nshockgrid2,nexog,ns2)

h5open("polydef2.h5", "w") do file
    @write file nexogs2
    @write file ninter2
    @write file ns2
    @write file nsv2
    @write file exoggridi2
end
