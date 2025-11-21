using DSGE
using Test

# Test `hessizero` in context of Rosenbrock function
function rosenbrock(x::Vector)
    a = 100
    b = 1
    return a*(x[2]-x[1]^2.0)^2.0 + b*(1-x[1])^2.0
end

function rosenbrock_hessian(x::Vector)
    H = zeros(eltype(x), 2, 2)
    H[1,1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2.0
    H[1,2] = -400.0 * x[1]
    H[2,1] = -400.0 * x[1]
    H[2,2] = 200.0
    return H
end


#bounds
lb = [-9999.,-9999.]
ub = [9999.,9999.]

# At the min, ensure no negatives in diagonal
x0 = [1.0, 1.0]


diag_order_ls = [2,4]
offdiag_order_ls = [1,2,4]
o4_method = [:richardson, :sixteenstencil]

for diag_order in diag_order_ls
    for offdiag_order in offdiag_order_ls
        for method in o4_method
            if offdiag_order != 4 && method == :sixteenstencil
                continue
            else

                hessian_expected = rosenbrock_hessian(x0)
                hessian, = DSGE.hessizero(rosenbrock, x0, lb, ub; diag_order_error = diag_order, offdiag_order_error = offdiag_order, offdiag_method = method, check_neg_diag=true)

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

            end
        end
    end
end


