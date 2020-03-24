using DSGE, ModelConstructors, SparseArrays, SparsityDetection, SparseDiffTools

m = AnSchorfheide()
function f1(dx, x)
    ModelConstructors.update!(m.parameters, x; change_value_type = true)
    steadystate!(m)
    θ = m.keys
    for (i,k) in zip(1:length(x), keys(θ))
        dx[i] = x[θ[k]]^2
    end
end
function f2(dx, x)
    ModelConstructors.update!(m.parameters, x; change_value_type = true)
    steadystate!(m)
    θ = m.keys
    p = map(y -> y.value, m.parameters)
    for (i,k) in zip(1:length(x), keys(θ))
        dx[i] = p[θ[k]]^2
    end
end
k = Dict(i => i for i in 1:16)
function f3(dx, x)
    for i in 1:length(x)
        dx[i] = x[k[i]]^2
    end
end
function f4(dx, x)
    for i in 1:length(x)
        dx[i] = x[i]^2
    end
end


input = map(x -> x.value, m.parameters)
output = similar(input)
sparsity_pattern1 =sparsity!(f1, output, input)
sparsity_pattern2 =sparsity!(f2, output, input)
sparsity_pattern3 =sparsity!(f3, output, input)
sparsity_pattern4 =sparsity!(f4, output, input)
jac1 = Float64.(sparse(sparsity_pattern4))
jac2 = Float64.(sparse(sparsity_pattern4))
jac3 = Float64.(sparse(sparsity_pattern4))
jac4 = Float64.(sparse(sparsity_pattern4))
colors = matrix_colors(jac4)
forwarddiff_color_jacobian!(jac1, f1, input, colorvec = colors)
forwarddiff_color_jacobian!(jac2, f2, input, colorvec = colors)
forwarddiff_color_jacobian!(jac3, f3, input, colorvec = colors)
forwarddiff_color_jacobian!(jac4, f4, input, colorvec = colors)
