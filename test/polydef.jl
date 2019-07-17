# Get the variables here from the zlb-in-julia polydef.jl functions

using Test, HDF5, DSGE
m = GHLS()
path = dirname(@__FILE__)

# Reading in reference output
nexogs_ref, ninter_ref, ns_ref, nsv_ref, exoggridi_ref, nquad_ref, ghnodes_ref, ghweights_ref, xgrid_ref, bbt_ref, bbtinv_ref, smolyak_ref = h5open("polydef.h5", "r") do file
    read(file, "nexogs"), read(file, "ninter"), read(file, "ns"), read(file, "nsv"),
    read(file, "exoggridi"), read(file, "nquad"), read(file, "ghnodes"), read(file, "ghweights"), read(file, "xgrid"), read(file, "bbt"), read(file, "bbtinv"), read(file, "smolyak")
end

nexogs, ninter, ns, nsv = setgridsize(m.approx.nexog, m.approx.nshockgrid)

@testset "setgridsize" begin
    @test nexogs == nexogs_ref
    @test ninter == ninter_ref
    @test ns == ns_ref
    @test nsv == nsv_ref
end


exoggridi = exoggridindex(m.approx.nshockgrid,m.approx.nexog,ns)
@testset "exoggridindex" begin
    @test exoggridi == exoggridi_ref
end

nquads, ghnodess, ghweightss = ghquadrature(3,m.approx.nexog)
@testset "ghquadrature" begin
    @test nquads == nquad_ref
    @test ghnodess == ghnodes_ref
    @test ghweightss == ghweights_ref
end


#@assert(nquads == Int(readdlm(string(path, "nquads.txt"))[1]))

xgrid, bbt, bbtinv = sparsegrid(m.approx.nmsv, m.approx.nindplus, m.approx.ngrid, m.approx.indplus)
@testset "sparsegrid" begin
    @test xgrid == xgrid_ref
    @test bbt == bbt_ref
    @test bbtinv == bbtinv_ref
end

smolyak = smolyakpoly(m.approx.nmsv, m.approx.ngrid, m.approx.nindplus, m.approx.indplus, xgrid[:,4])
@testset "smolyakpoly" begin
    @test smolyak == vec(smolyak_ref)
end

# Testing with nshockgrid having 3 instead of 2
m.approx.nshockgrid = [7 3 3 3 3 2]

# Reading in reference output
nexogs_ref2, ninter_ref2, ns_ref2, nsv_ref2, exoggridi_ref2, = h5open("polydef2.h5", "r") do file
    read(file, "nexogs2"), read(file, "ninter2"), read(file, "ns2"), read(file, "nsv2"),
    read(file, "exoggridi2")
end

nexogs2, ninter2, ns2, nsv2 = setgridsize(m.approx.nexog, m.approx.nshockgrid)
@testset "setgridsize" begin
    @test nexogs2 == nexogs_ref2
    @test ninter2 == ninter_ref2
    @test ns2 == ns_ref2
    @test nsv2 == nsv_ref2
end

#@assert([nexogs2, ninter2, ns2, nsv2] == round.(Int, vec(readdlm(string(path,"setgridsize2.txt")))))

exoggridi2 = exoggridindex(m.approx.nshockgrid,m.approx.nexog,ns2)
@testset "exogridindex2" begin
    @test exoggridi2 == exoggridi_ref2
end
#@assert(exoggridi2 == readdlm(string(path, "exoggridi2.txt")))
