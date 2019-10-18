# Test get_fixed_parameter_indices
@testset "Test getting fixed parameter indices" begin
    m = AnSchorfheide()
    out = get_fixed_parameter_indices(m)
    exp_out = vcat(falses(13), trues(3))
    @test out == exp_out
end

nothing
