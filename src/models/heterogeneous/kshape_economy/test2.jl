#run this to compare matrices after running test.jl
using Test


names = collect(keys(F))
state_names = names[1:os]
control_names = names[os+1:end]

function compare_matrices(AD, FD, name::String, eqn_state::Bool, var_state; atol=1e-6)
    @testset "$name" begin
        @test size(AD) == size(FD)
        if size(AD) == size(FD)
            diffs = findall(i -> !isapprox(AD[i], FD[i]; atol=atol), eachindex(AD))
            if !isempty(diffs)
                println("  $(length(diffs)) differing elements in $name:")
                for i in diffs
                    r, c = Tuple(CartesianIndices(AD)[i])
                    eqn = eqn_state ? state_names[r] : control_names[r]
                    var = var_state ? state_names[c] : control_names[c]
                    println("eqn (row) $eqn, var (col) $var") 
                    println("[$r,$c]  julia=$(AD[i])  original=$(FD[i])  ")
                    println("===============================================")
                end
            end
            @test isempty(diffs)
        end
    end
end

compare_matrices(F21_ad, F21_aux_trim, "F21 (state eqns wrt x_t)",   true,  true)
compare_matrices(F22_ad, F22_aux_trim, "F22 (state eqns wrt y_t+1)", true,  false)
compare_matrices(F23_ad, F23_aux_trim, "F23 (state eqns wrt x_t-1)", true,  true)
compare_matrices(F24_ad, F24_aux_trim, "F24 (state eqns wrt y_t)",   true,  false)
compare_matrices(F41_ad, F41_aux_trim, "F41 (ctrl eqns wrt x_t)",    false, true)
compare_matrices(F42_ad, F42_aux_trim, "F42 (ctrl eqns wrt y_t+1)",  false, false)
compare_matrices(F43_ad, F43_aux_trim, "F43 (ctrl eqns wrt x_t-1)",  false, true)
compare_matrices(F44_ad, F44_aux_trim, "F44 (ctrl eqns wrt y_t)",    false, false)
