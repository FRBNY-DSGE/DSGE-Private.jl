# Get the variables here from the zlb-in-julia polydef.jl functions

using Test, HDF5, DSGE
m = GHLS()
path = dirname(@__FILE__)

# Reading in reference output
nexogs_ref, ninter_ref, ns_ref, nsv_ref, exoggridi_ref = h5open("polydef.h5", "r") do file
    read(file, "nexogs"), read(file, "ninter"), read(file, "ns"), read(file, "nsv"),
    read(file, "exoggridi")
end

nexogs, ninter, ns, nsv = setgridsize(m.approx.nexog, m.approx.nshockgrid)

@testset "setgridsize" begin
    @test nexogs == nexogs_ref
    @test ninters == ninter_ref
    @test ns == ns_ref
    @test nsv == nsv_ref
end


exoggridi = exoggridindex(m.approx.nshockgrid,m.approx.nexog,ns)
@test exoggridi == exoggridi_ref


nquads, ghnodess, ghweightss = ghquadrature(3,m.approx.nexog)

@assert(nquads == Int(readdlm(string(path, "nquads.txt"))[1]))
@assert(ghnodess == readdlm(string(path, "ghnodes.txt")))
@assert(ghweightss == readdlm(string(path, "ghweights.txt")))

xgrid, bbt, bbtinv = sparsegrid(m.approx.nmsv, m.approx.nindplus, m.approx.ngrid, m.approx.indplus)
@assert(xgrid == readdlm(string(path, "xgrid.txt")))
@assert(bbt == readdlm(string(path, "bbt.txt")))
@assert(bbtinv == readdlm(string(path, "bbtinv.txt")))

smolyak = smolyakpoly(m.approx.nmsv, m.approx.ngrid, m.approx.nindplus, m.approx.indplus, xgrid[:,4])
@assert(smolyak == vec(readdlm(string(path, "smolyak.txt"))))

# Testing with nshockgrid having 3 instead of 2
m.approx.nshockgrid = [7 3 3 3 3 1]

nexogs2, ninter2, ns2, nsv2 = setgridsize(m.approx.nexog, m.approx.nshockgrid)
@assert([nexogs2, ninter2, ns2, nsv2] == round.(Int, vec(readdlm(string(path,"setgridsize2.txt")))))

exoggridi2 = exoggridindex(m.approx.nshockgrid,m.approx.nexog,ns2)
@assert(exoggridi2 == readdlm(string(path, "exoggridi2.txt")))
