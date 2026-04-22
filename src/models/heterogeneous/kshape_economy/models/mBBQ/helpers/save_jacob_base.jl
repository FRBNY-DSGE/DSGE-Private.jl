using LinearAlgebra
using JLD2

"""
    save_jacob_base(F1_aux, F2_aux, F3_aux, F4_aux, grid;
                    adjust=nothing, save_to_file=false, output_dir=pwd())

Build the baseline Jacobian block dictionary (MATLAB `Jacob_base`) from the
auxiliary Jacobians `F1_aux`-`F4_aux`.

If `save_to_file=true` and `adjust` is `"G"` or `"LT"`, the object is also saved
as a JLD2 file whose base name matches the MATLAB naming convention.
"""
function save_jacob_base(
    F1_aux::AbstractMatrix,
    F2_aux::AbstractMatrix,
    F3_aux::AbstractMatrix,
    F4_aux::AbstractMatrix,
    m;
    adjust=nothing,
    # save_to_file::Bool=false,
    # output_dir::AbstractString=pwd(),
)
    getgrid(g, k) =
        g isa AbstractDict ? (
            haskey(g, k) ? g[k] :
            haskey(g, String(k)) ? g[String(k)] :
            error("Missing grid key: $(k)")
        ) : getproperty(g, k)

    numstates = Int(getgrid(m.dicts[:grid], "numstates"))
    os = Int(getgrid(m.dicts[:grid], "os"))
    os_agg = Int(getgrid(m.dicts[:grid], "os_agg"))
    os_process = Int(getgrid(m.dicts[:grid], "os_process"))
    numcontrols = Int(getgrid(m.dicts[:grid], "numcontrols"))
    oc = Int(getgrid(m.dicts[:grid], "oc"))
    oc_summary = Int(getgrid(m.dicts[:grid], "oc_summary"))
    oc_agg = Int(getgrid(m.dicts[:grid], "oc_agg"))

    # Collect blocks that do not need to be updated.
    F1Xnext = F1_aux[1:numstates-os, :]
    F21next = zeros(eltype(F1_aux), os_agg, numstates-os)
    F31next = zeros(eltype(F1_aux), os_process, numstates-os)
    F32next = zeros(eltype(F1_aux), os_process, os_agg)
    F33next = Matrix{eltype(F1_aux)}(I, os_process, os_process)
    F3Xnext = hcat(F31next, F32next, F33next)
    F41next = zeros(eltype(F1_aux), numcontrols-oc, numstates-os)
    F42next = F1_aux[numstates+1:end-oc, end-os+1:end-os_process]
    F43next = zeros(eltype(F1_aux), numcontrols-oc, os_process)
    F4Xnext = hcat(F41next, F42next, F43next)
    F51next = zeros(eltype(F1_aux), oc_summary, numstates-os)
    F53next = zeros(eltype(F1_aux), oc_summary, os_process)
    F5Xnext = F1_aux[end-oc+1:end-oc+oc_summary, :]
    F61next = zeros(eltype(F1_aux), oc_agg, numstates-os)

    # F22next, F23next, F52next, F62next, F63next are updated elsewhere.
    F1Ynext = F2_aux[1:numstates-os, :]
    F24next = zeros(eltype(F2_aux), os_agg, numcontrols-oc)
    F3Ynext = zeros(eltype(F2_aux), os_process, numcontrols)
    F4Ynext = F2_aux[numstates+1:end-oc, :]
    F5Ynext = F2_aux[end-oc+1:end-oc+oc_summary, :]
    F64next = F2_aux[end-oc_agg+1:end, 1:end-oc]

    # F25next, F26next, F65next, F66next are updated elsewhere.
    F1X = F3_aux[1:numstates-os, :]
    F21 = zeros(eltype(F3_aux), os_agg, numstates-os)
    F31 = zeros(eltype(F3_aux), os_process, numstates-os)
    F32 = zeros(eltype(F3_aux), os_process, os_agg)
    F4X = F3_aux[numstates+1:end-oc, :]
    F51 = F3_aux[end-oc+1:end-oc_agg, 1:numstates-os]
    F5X = F3_aux[end-oc+1:end-oc+oc_summary, :]
    F61 = F3_aux[end-oc_agg+1:end, 1:numstates-os]

    # F22, F23, F33, F52, F53, F62, F63 are updated elsewhere.
    F1Y = F4_aux[1:numstates-os, :]
    F24 = F4_aux[numstates-os+1:numstates-os+os_agg, 1:end-oc]
    F3Y = zeros(eltype(F4_aux), os_process, numcontrols)
    F4Y = F4_aux[numstates+1:end-oc, :]
    F54 = zeros(eltype(F4_aux), oc_summary, numcontrols-oc)
    F5Y = F4_aux[end-oc+1:end-oc+oc_summary, :]
    F64 = F4_aux[end-oc_agg+1:end, 1:end-oc]

    # F25, F26, F55, F56, F65, F66 are updated elsewhere.
    Jacob_base = Dict{Symbol, Any}()
    Jacob_base[:F1_aux] = F1_aux
    Jacob_base[:F2_aux] = F2_aux
    Jacob_base[:F3_aux] = F3_aux
    Jacob_base[:F4_aux] = F4_aux

    Jacob_base[:F1Xnext] = F1Xnext
    Jacob_base[:F21next] = F21next
    Jacob_base[:F31next] = F31next
    Jacob_base[:F32next] = F32next
    Jacob_base[:F33next] = F33next
    Jacob_base[:F3Xnext] = F3Xnext
    Jacob_base[:F41next] = F41next
    Jacob_base[:F42next] = F42next
    Jacob_base[:F43next] = F43next
    Jacob_base[:F4Xnext] = F4Xnext
    Jacob_base[:F51next] = F51next
    Jacob_base[:F53next] = F53next
    Jacob_base[:F61next] = F61next
    Jacob_base[:F1Ynext] = F1Ynext
    Jacob_base[:F24next] = F24next
    Jacob_base[:F3Ynext] = F3Ynext
    Jacob_base[:F4Ynext] = F4Ynext
    Jacob_base[:F5Xnext] = F5Xnext
    Jacob_base[:F5Ynext] = F5Ynext
    Jacob_base[:F64next] = F64next
    Jacob_base[:F1X] = F1X
    Jacob_base[:F21] = F21
    Jacob_base[:F31] = F31
    Jacob_base[:F32] = F32
    Jacob_base[:F4X] = F4X
    Jacob_base[:F51] = F51
    Jacob_base[:F61] = F61
    Jacob_base[:F1Y] = F1Y
    Jacob_base[:F24] = F24
    Jacob_base[:F3Y] = F3Y
    Jacob_base[:F4Y] = F4Y
    Jacob_base[:F54] = F54
    Jacob_base[:F5X] = F5X
    Jacob_base[:F5Y] = F5Y
    Jacob_base[:F64] = F64

    # if save_to_file
    #     if !isnothing(adjust) && string(adjust) == "G"
    #         filename = joinpath(output_dir, "Jacob_base_G_tvcopula.jld2")
    #         JLD2.jldsave(filename; Jacob_base_G_tvcopula=Jacob_base)
    #     elseif !isnothing(adjust) && string(adjust) == "LT"
    #         filename = joinpath(output_dir, "Jacob_base_LT_tvcopula.jld2")
    #         JLD2.jldsave(filename; Jacob_base_LT_tvcopula=Jacob_base)
    #     else
    #         @warn "Skipping file save: expected adjust=\"G\" or adjust=\"LT\"."
    #     end
    # end

    return Jacob_base
end
