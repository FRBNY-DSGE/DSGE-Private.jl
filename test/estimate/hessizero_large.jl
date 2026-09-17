using DSGE
using Test
using LinearAlgebra

# Test `hessizero` in context of Extended Rosenbrock function
# Extended Rosenbrock: f(x) = sum_{i=1}^{n-1} [a*(x[i+1] - x[i]^2)^2 + b*(1 - x[i])^2]
function extended_rosenbrock(x::Vector)
    a = 100.0
    b = 1.0
    n = length(x)
    result = 0.0
    for i in 1:(n-1)
        result += a * (x[i+1] - x[i]^2)^2 + b * (1 - x[i])^2
    end
    return result
end

function extended_rosenbrock_hessian(x::Vector)
    n = length(x)
    a = 100.0
    b = 1.0
    H = zeros(eltype(x), n, n)

    # Diagonal elements
    # H[1,1] from first term only
    H[1,1] = -4*a*x[2] + 12*a*x[1]^2 + 2*b

    # H[i,i] for 1 < i < n (contributions from terms i-1 and i)
    for i in 2:(n-1)
        H[i,i] = 2*a + (-4*a*x[i+1] + 12*a*x[i]^2 + 2*b)
    end

    # H[n,n] from last term only
    H[n,n] = 2*a

    # Off-diagonal elements (tridiagonal structure)
    for i in 1:(n-1)
        H[i,i+1] = -4*a*x[i]
        H[i+1,i] = -4*a*x[i]
    end

    return H
end


# Dimension of the problem
n_dim = 50

# Bounds
lb = fill(-9999.0, n_dim)
ub = fill(9999.0, n_dim)

# At the minimum x = [1, 1, ..., 1], ensure no negatives in diagonal
x0 = ones(n_dim)


diag_order_ls = [2,4]
offdiag_order_ls = [1,2,4]
o4_method = [:richardson, :sixteenstencil]

for diag_order in diag_order_ls
    for offdiag_order in offdiag_order_ls
        for method in o4_method
            if offdiag_order != 4 && method == :sixteenstencil
                continue
            else

                hessian_expected = extended_rosenbrock_hessian(x0)
                start = time()
                hessian, = DSGE.hessizero(extended_rosenbrock, x0, lb, ub; diag_order_error = diag_order, offdiag_order_error = offdiag_order, offdiag_method = method, check_neg_diag=true)
                endt = time() - start
                inf_norm = norm(hessian_expected - hessian, Inf)
                l2_norm = norm(hessian_expected - hessian)
                
                println("================================================================================")
                if offdiag_order == 4

                println("O(h^$(diag_order)) diag, O(h^$(offdiag_order)) offdiag, $(method) offdiag method")
                else
                println("O(h^$(diag_order)) diag, O(h^$(offdiag_order)) offdiag")
                end

                println("inf norm = $(inf_norm)")
                println("l2_norm = $(l2_norm)")
                println("seconds = $(endt)")

            end
        end
    end
end


