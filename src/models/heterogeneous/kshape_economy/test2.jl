#run this to compare matrices after running test.jl
using Test


names = collect(keys(F))
state_eqn_names = vcat(fill("dont know name", 88), names[1:os])
control_eqn_names = names[os+1:end]


names2 = collect(keys(state_id))
state_var_names = vcat(fill("dont know name", 88), names2[5:end])

names3 = collect(keys(control_id))
control_var_names = names3[4:end]
new_syms = [:J_t2, :J_t3, :J_t4, :J_t5]
splice!(control_var_names, 18:17, new_syms)


control_var_name = []
#for i = 17692:17732

function compare_matrices(AD, FD, name::String, eqn_state::Bool, var_state; atol=1e-4,
                          exceptions=Vector{Vector{Int}}())
    exc_set = Set(Tuple(e) for e in exceptions)
    @testset "$name" begin
        @test size(AD) == size(FD)
        if size(AD) == size(FD)
            diffs = findall(i -> !isapprox(AD[i], FD[i]; atol=atol), eachindex(AD))
            filter!(i -> begin
                r, c = Tuple(CartesianIndices(AD)[i])
                (r, c) ∉ exc_set
            end, diffs)
            if !isempty(diffs)
                println("  $(length(diffs)) differing elements in $name:")
                for i in diffs
                    r, c = Tuple(CartesianIndices(AD)[i])
                    eqn = eqn_state ? state_eqn_names[r+88] : control_eqn_names[r]
                    var = var_state ? state_var_names[c] : control_var_names[c]
                    println("eqn (row) $eqn, var (col) $var") 
                    println("[$r,$c]  julia=$(AD[i])  original=$(FD[i])  ")
                    println("===============================================")
                end
            end
            @test isempty(diffs)
        end
    end
end

#compare_matrices(F21_ad, F21_aux_trim, "F21 (state eqns wrt x_t)",   true,  true, exceptions=[[5, 110]])
#compare_matrices(F22_ad, F22_aux_trim, "F22 (state eqns wrt y_t+1)", true,  false)
#compare_matrices(F23_ad, F23_aux_trim, "F23 (state eqns wrt x_t-1)", true,  true)
compare_matrices(F24_ad, F24_aux_trim, "F24 (state eqns wrt y_t)",   true,  false)
compare_matrices(F41_ad, F41_aux_trim, "F41 (ctrl eqns wrt x_t)",    false, true)
compare_matrices(F42_ad, F42_aux_trim, "F42 (ctrl eqns wrt y_t+1)",  false, false)
compare_matrices(F43_ad, F43_aux_trim, "F43 (ctrl eqns wrt x_t-1)",  false, true)
compare_matrices(F44_ad, F44_aux_trim, "F44 (ctrl eqns wrt y_t)",    false, false)
