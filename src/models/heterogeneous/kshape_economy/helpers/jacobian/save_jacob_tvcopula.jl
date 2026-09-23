using JLD2

"""
    save_jacob_tvcopula(F1_aux, F2_aux, F3_aux, F4_aux, m; adjust=nothing)

Build CSC tvcopula Jacobian base blocks from auxiliary Jacobians.
Follows the same style as `models/mBBQ/helpers/save_jacob_base.jl`.
"""
function save_jacob_tvcopula(
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

    g = m.dicts[:grid]
    numstates = Int(getgrid(g, "numstates"))
    os = Int(getgrid(g, "os"))
    numcontrols = Int(getgrid(g, "numcontrols"))
    oc = Int(getgrid(g, "oc"))
    oc_summary = Int(getgrid(g, "oc_summary"))

    # Effects of t states/controls on distribution and aggregates
    F1Xnext = F1_aux[1:numstates-os, :]
    F4Xnext = F1_aux[numstates+1:numstates+numcontrols-oc, :]
    F5Xnext = F1_aux[end-oc+1:end-oc+oc_summary, :]

    F1X = F3_aux[1:numstates-os, :]
    F4X = F3_aux[numstates+1:end-oc, :]
    F5X = F3_aux[end-oc+1:end-oc+oc_summary, :]

    F1Ynext = F2_aux[1:numstates-os, :]
    F4Ynext = F2_aux[numstates+1:end-oc, :]
    F5Ynext = F2_aux[end-oc+1:end-oc+oc_summary, :]

    F1Y = F4_aux[1:numstates-os, :]
    F4Y = F4_aux[numstates+1:end-oc, :]
    F5Y = F4_aux[end-oc+1:end-oc+oc_summary, :]

    Jacob_base = Dict{Symbol, Any}()
    Jacob_base[:F1_aux] = F1_aux
    Jacob_base[:F2_aux] = F2_aux
    Jacob_base[:F3_aux] = F3_aux
    Jacob_base[:F4_aux] = F4_aux

    Jacob_base[:F1Xnext] = F1Xnext
    Jacob_base[:F4Xnext] = F4Xnext
    Jacob_base[:F5Xnext] = F5Xnext

    Jacob_base[:F1X] = F1X
    Jacob_base[:F4X] = F4X
    Jacob_base[:F5X] = F5X

    Jacob_base[:F1Ynext] = F1Ynext
    Jacob_base[:F4Ynext] = F4Ynext
    Jacob_base[:F5Ynext] = F5Ynext

    Jacob_base[:F1Y] = F1Y
    Jacob_base[:F4Y] = F4Y
    Jacob_base[:F5Y] = F5Y

    # if save_to_file
    #     filename = joinpath(output_dir, "Jacob_base_CSC_tvcopula.jld2")
    #     JLD2.jldsave(filename; Jacob_base_CSC_tvcopula = Jacob_base)
    # end

    return Jacob_base
end
