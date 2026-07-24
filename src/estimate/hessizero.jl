"""
```
hessizero(fcn::Function, x::Vector{T};
          check_neg_diag::Bool=false,
          verbose::Symbol=:none,
          distr::Bool=true) where T<:AbstractFloat
```

Compute Hessian of function `fcn` evaluated at `x`.

### Arguments
- `check_neg_diag`: Throw an error if any negative diagonal elements are detected.
- `verbose`: Print verbose output
- `distr`: Use available parallel workers to increase performance.
"""
function hessizero(fcn::Function,
                   x::Vector{T},
                   lb,
                   ub;
                   diag_order_error::Int  = 4,
                   offdiag_order_error::Int = 2,
                   offdiag_method::Symbol = :richardson,
                   check_neg_diag::Bool=false,
                   verbose::Symbol=:none,
                   distr::Bool=true) where T<:AbstractFloat
    

    hess_diag_element_fcn = if diag_order_error == 2
        hess_diag_element_o2
    elseif diag_order_error == 4
        hess_diag_element_o4
    end

    hess_offdiag_element_fcn = if offdiag_order_error == 1
        hess_offdiag_element_o1
    elseif offdiag_order_error == 2
        hess_offdiag_element_o2
    elseif offdiag_order_error == 4
        hess_offdiag_element_o4
    end



    n_para = length(x)
    hessian  = zeros(n_para, n_para)

    # Compute diagonal elements first
    if distr && nworkers() > 1
        diag_elements = @sync @distributed (vcat) for i = 1:n_para
            hess_diag_element_fcn(fcn, x, i, lb, ub; check_neg_diag = check_neg_diag, verbose = verbose)
        end
        hessian = diagm(diag_elements)
    else
        for i = 1:n_para
            hessian[i,i] = hess_diag_element_fcn(fcn, x, i, lb, ub; check_neg_diag = check_neg_diag,
                                             verbose = verbose)
        end
    end

    # Now compute off-diagonal elements
    # Make sure that correlations are between -1 and 1
    # invalid_corr indexes elements that are invalid
    invalid_corr = Dict{Tuple{Int,Int}, Float64}()

    # Build indices to iterate over
    n_off_diag_els = Int(n_para * (n_para - 1) / 2)
    off_diag_inds = Vector{Tuple{Int,Int}}(undef, n_off_diag_els)
    k = 1
    for i = 1:(n_para - 1), j = (i + 1):n_para
        off_diag_inds[k] = (i,j)
        k = k + 1
    end

    # Iterate over off diag elements
    # (n_off_diag_els == 0 for a scalar parameter vector; skip the distributed
    #  path in that case, since @distributed (hcat) over an empty range errors
    #  with "reducing over an empty collection is not allowed".)
    if distr && n_off_diag_els > 0
        off_diag_out = @sync @distributed (hcat) for (i,j) in off_diag_inds
            σ_xσ_y = sqrt(abs(hessian[i, i]*hessian[j, j]))
            hess_offdiag_element_fcn(fcn, x, i, j, σ_xσ_y, lb, ub; method = offdiag_method, verbose=verbose)
        end
        # Ensure off_diag_out is array
        off_diag_out = hcat(off_diag_out)
    else
        off_diag_out = Array{Tuple{T, T},1}(undef, n_off_diag_els)
        for (k,(i,j)) in enumerate(off_diag_inds)
            σ_xσ_y = sqrt(abs(hessian[i, i]*hessian[j, j]))
            off_diag_out[k] = hess_offdiag_element_fcn(fcn, x, i, j, σ_xσ_y, lb, ub; method = offdiag_method, verbose=verbose)
        end
    end

    # Fill in values
    for k=1:n_off_diag_els
        (i,j) = off_diag_inds[k]
        (value, ρ_xy) = off_diag_out[k]

        hessian[i,j] = value
        hessian[j,i] = value

        if ρ_xy < -1 || 1 < ρ_xy
            invalid_corr[(i, j)] = ρ_xy
        end
    end

    has_errors = false
    if !isempty(invalid_corr)
        println("Errors: $invalid_corr")
        has_errors = true
    end

    return hessian, has_errors
end


# Compute diag element
function hess_diag_element_o4(fcn::Function,
                           x::Vector{T},
                           i::Int,
                           lb,
                           ub;
                           ndx::Int=6,
                           check_neg_diag::Bool=false,
                           verbose::Symbol=:none) where T<:AbstractFloat
    # Setup
    n_para = length(x)
    dxscale  = ones(n_para, 1)
    dx       = exp.(-(6:2:(6+(ndx-1)*2))')
    hessdiag = zeros(ndx, 1)

    println(verbose, :low, "Hessian element: ($i, $i)")

    # Diagonal element computation
    for k = 3:4
        hi = dx[k]*dxscale[i]

        forward_i_valid = (x[i] + 5 * hi <= ub[i])
        backward_i_valid = (x[i] - 5 * hi >= lb[i])
        

       
         
        #do center difference O(h^4)
        if forward_i_valid && backward_i_valid
            #println("CENTER DIFF o4")

            paradx = copy(x)
            parady = copy(x)
            para2dx = copy(x)
            para2dy = copy(x)

            paradx[i] += hi
            parady[i] -= hi
            para2dx[i] += (2*hi)
            para2dy[i] -= (2*hi)

            fx  = fcn(x)
            fdx = fcn(paradx)
            fdy = fcn(parady)
            f2dx = fcn(para2dx)
            f2dy = fcn(para2dy)

            hessdiag[k]  = (-f2dx + 16*fdx - 30*fx + 16*fdy - f2dy) / (12*hi^2)
            #println(hessdiag[k]) 

        #do backward difference O(h^4)
        elseif backward_i_valid
            #println("BWARD DIFF O4")
            parady = copy(x)
            para2dy = copy(x)
            para3dy = copy(x)
            para4dy = copy(x)
            para5dy = copy(x)

            parady[i] -= hi
            para2dy[i] -= (2*hi)
            para3dy[i] -= (3*hi)
            para4dy[i] -= (4*hi)
            para5dy[i] -= (5*hi)

            fx  = fcn(x)
            fdy = fcn(parady)
            f2dy = fcn(para2dy)
            f3dy = fcn(para3dy)
            f4dy = fcn(para4dy)
            f5dy = fcn(para5dy)

            hessdiag[k]  = (45*fx - 154*fdy + 214*f2dy - 156*f3dy + 61*f4dy - 10*f5dy) / (12*hi^2)
     #println(hessdiag[k]) 

        #do forward difference O(h^4)
        elseif forward_i_valid
            #println("FWARD DIFF O4")
            paradx = copy(x)
            para2dx = copy(x)
            para3dx = copy(x)
            para4dx = copy(x)
            para5dx = copy(x)

            paradx[i] += hi
            para2dx[i] += (2*hi)
            para3dx[i] += (3*hi)
            para4dx[i] += (4*hi)
            para5dx[i] += (5*hi)

            fx  = fcn(x)
            fdx = fcn(paradx)
            f2dx = fcn(para2dx)
            f3dx = fcn(para3dx)
            f4dx = fcn(para4dx)
            f5dx = fcn(para5dx)

            hessdiag[k]  = (45*fx - 154*fdx + 214*f2dx - 156*f3dx + 61*f4dx - 10*f5dx) / (12*hi^2)
    #println(hessdiag[k]) 

        end
    end

    #println(verbose, :high, "Values: $(hessdiag)")

    value = (hessdiag[3]+hessdiag[4])/2

    if check_neg_diag && value < 0
        #println("ORIGINAL o4")
        #println(value)
        value = hess_diag_element_o2(fcn, x, i, lb, ub; check_neg_diag = check_neg_diag,verbose = verbose)
               #error("Negative diagonal in Hessian")
    end

    println(verbose, :high, "Value used: $value")

    return value
end

# Compute off diag element
function hess_offdiag_element_o4(fcn::Function,
                              x::Vector{T},
                              i::Int,
                              j::Int,
                              σ_xσ_y::T,
                              lb,
                              ub;
                              ndx::Int=6,
                              method::Symbol=:richardson,
                              verbose::Symbol=:none) where T<:AbstractFloat
    # Setup
    n_para = length(x)
    dxscale  = ones(n_para, 1)
    dx       = exp.(-(6:2:(6+(ndx-1)*2))')
    hessdiag = zeros(ndx, 1)

    # Computation
    println(verbose, :low, "Hessian element: ($i, $j)")

    for k = 3:4
        hi = dx[k]*dxscale[i]
        hj = dx[k]*dxscale[j]

        forward_i_valid = (x[i] + hi <= ub[i])
        forward_j_valid = (x[j] + hj <= ub[j])
        backward_i_valid = (x[i] - hi >= lb[i])
        backward_j_valid = (x[j] - hj >= lb[j])


        #taking O(h^2) centered difference and doing Richard extrapolation
        #A = 4/3 f_xy(h/2) + 1/3 f_xy(h)
        if method == :richardson && forward_i_valid && backward_i_valid && forward_j_valid && backward_j_valid
            step_sizes = [1, 0.5]
            A = [0.,0.]

            for (l, t) in enumerate(step_sizes)
            hit = hi * t
            hjt = hj * t

            #f(x_i - h_i, x_j - h_j)
            para_idy_jdy = copy(x)
            para_idy_jdy[i] -= hit
            para_idy_jdy[j] -= hjt
            f_idy_jdy = fcn(para_idy_jdy)

            #f(x_i + h_i, x_j - h_j)
            para_idx_jdy = copy(x)
            para_idx_jdy[i] += hit
            para_idx_jdy[j] -= hjt
            f_idx_jdy = fcn(para_idx_jdy)
 
            #f(x_i - h_i, x_j + h_j)
            para_idy_jdx = copy(x)
            para_idy_jdx[i] -= hit
            para_idy_jdx[j] += hjt
            f_idy_jdx = fcn(para_idy_jdx)

            #f(x_i + h_i, x_j + h_j)
            para_idx_jdx = copy(x)
            para_idx_jdx[i] += hit
            para_idx_jdx[j] += hjt
            f_idx_jdx = fcn(para_idx_jdx)

            A[l] = (f_idy_jdy + f_idx_jdx - f_idx_jdy - f_idy_jdx) / (4*(hit * hjt))

            end

            hessdiag[k] = -(1/3) * A[1] + (4/3) * A[2]

         #4x4 stencil
        elseif method == :sixteenstencil && forward_i_valid && backward_i_valid && forward_j_valid && backward_j_valid

            #f(x_i - 2h_i, x_j - 2h_j)
            para_i2dy_j2dy = copy(x)
            para_i2dy_j2dy[i] -= (2*hi)
            para_i2dy_j2dy[j] -= (2*hj)
            f_i2dy_j2dy = fcn(para_i2dy_j2dy)

            #f(x_i - h_i, x_j - 2h_j)
            para_idy_j2dy = copy(x)
            para_idy_j2dy[i] -= hi
            para_idy_j2dy[j] -= (2*hj)
            f_idy_j2dy = fcn(para_idy_j2dy)

            #f(x_i + h_i, x_j - 2h_j)
            para_idx_j2dy = copy(x)
            para_idx_j2dy[i] += hi
            para_idx_j2dy[j] -= (2*hj)
            f_idx_j2dy = fcn(para_idx_j2dy)

            #f(x_i + 2h_i, x_j - 2h_j)
            para_i2dx_j2dy = copy(x)
            para_i2dx_j2dy[i] += (2*hi)
            para_i2dx_j2dy[j] -= (2*hj)
            f_i2dx_j2dy = fcn(para_i2dx_j2dy)

            #f(x_i - 2h_i, x_j - h_j)
            para_i2dy_jdy = copy(x)
            para_i2dy_jdy[i] -= (2*hi)
            para_i2dy_jdy[j] -= hj
            f_i2dy_jdy = fcn(para_i2dy_jdy)

            #f(x_i - h_i, x_j - h_j)
            para_idy_jdy = copy(x)
            para_idy_jdy[i] -= hi
            para_idy_jdy[j] -= hj
            f_idy_jdy = fcn(para_idy_jdy)

            #f(x_i + h_i, x_j - h_j)
            para_idx_jdy = copy(x)
            para_idx_jdy[i] += hi
            para_idx_jdy[j] -= hj
            f_idx_jdy = fcn(para_idx_jdy)

            #f(x_i + 2h_i, x_j - h_j)
            para_i2dx_jdy = copy(x)
            para_i2dx_jdy[i] += (2*hi)
            para_i2dx_jdy[j] -= hj
            f_i2dx_jdy = fcn(para_i2dx_jdy)

            #f(x_i - 2h_i, x_j + h_j)
            para_i2dy_jdx = copy(x)
            para_i2dy_jdx[i] -= (2*hi)
            para_i2dy_jdx[j] += hj
            f_i2dy_jdx = fcn(para_i2dy_jdx)

            #f(x_i - h_i, x_j + h_j)
            para_idy_jdx = copy(x)
            para_idy_jdx[i] -= hi
            para_idy_jdx[j] += hj
            f_idy_jdx = fcn(para_idy_jdx)

            #f(x_i + h_i, x_j + h_j)
            para_idx_jdx = copy(x)
            para_idx_jdx[i] += hi
            para_idx_jdx[j] += hj
            f_idx_jdx = fcn(para_idx_jdx)

            #f(x_i + 2h_i, x_j + h_j)
            para_i2dx_jdx = copy(x)
            para_i2dx_jdx[i] += (2*hi)
            para_i2dx_jdx[j] += hj
            f_i2dx_jdx = fcn(para_i2dx_jdx)

            #f(x_i - 2h_i, x_j + 2h_j)
            para_i2dy_j2dx = copy(x)
            para_i2dy_j2dx[i] -= (2*hi)
            para_i2dy_j2dx[j] += (2*hj)
            f_i2dy_j2dx = fcn(para_i2dy_j2dx)

            #f(x_i - h_i, x_j + 2h_j)
            para_idy_j2dx = copy(x)
            para_idy_j2dx[i] -= hi
            para_idy_j2dx[j] += (2*hj)
            f_idy_j2dx = fcn(para_idy_j2dx)

            #f(x_i + h_i, x_j + 2h_j)
            para_idx_j2dx = copy(x)
            para_idx_j2dx[i] += hi
            para_idx_j2dx[j] += (2*hj)
            f_idx_j2dx = fcn(para_idx_j2dx)

            #f(x_i + 2h_i, x_j + 2h_j)
            para_i2dx_j2dx = copy(x)
            para_i2dx_j2dx[i] += (2*hi)
            para_i2dx_j2dx[j] += (2*hj)
            f_i2dx_j2dx = fcn(para_i2dx_j2dx)

            #O(h^4) with 4x4 stencil via Pavel Holoborodko
            hessdiag[k] = (f_i2dy_j2dy - 8*f_idy_j2dy + 8*f_idx_j2dy - f_i2dx_j2dy
                           - 8*f_i2dy_jdy + 64*f_idy_jdy - 64*f_idx_jdy + 8*f_i2dx_jdy
                           + 8*f_i2dy_jdx - 64*f_idy_jdx + 64*f_idx_jdx - 8*f_i2dx_jdx
                           - f_i2dy_j2dx + 8*f_idy_j2dx - 8*f_idx_j2dx + f_i2dx_j2dx) / (144*hi*hj)
        

        else
            hessdiag[k], _ = hess_offdiag_element_o2(fcn, x, i, j, σ_xσ_y, lb, ub; verbose=verbose)
        end

    end

    println(verbose, :high, "Values: $(hessdiag)")

    value = (hessdiag[3]+hessdiag[4])/2

    if value == 0 || σ_xσ_y == 0
        ρ_xy = 0
    else
        ρ_xy = value / σ_xσ_y
    end

    if ρ_xy < -1 || 1 < ρ_xy
        value = 0
    end

    println(verbose, :high, "Value used: $value")
    println(verbose, :high, "Correlation: $ρ_xy")

    return value, ρ_xy
end


# Compute off diag element
function hess_offdiag_element_o2(fcn::Function,
                              x::Vector{T},
                              i::Int,
                              j::Int,
                              σ_xσ_y::T,
                              lb,
                              ub;
                              ndx::Int=6,
                              method::Symbol=:none,
                              verbose::Symbol=:none) where T<:AbstractFloat
    # Setup
    n_para = length(x)
    dxscale  = ones(n_para, 1)
    dx       = exp.(-(6:2:(6+(ndx-1)*2))')
    hessdiag = zeros(ndx, 1)

    # Computation
    println(verbose, :low, "Hessian element: ($i, $j)")

    for k = 3:4
        hi = dx[k]*dxscale[i]
        hj = dx[k]*dxscale[j]

        forward_i_valid = (x[i] + hi <= ub[i])
        forward_j_valid = (x[j] + hj <= ub[j])
        backward_i_valid = (x[i] - hi >= lb[i])
        backward_j_valid = (x[j] - hj >= lb[j])
        

        #do forward(i)-backwards(j) O(h)
        if forward_i_valid && forward_j_valid && backward_i_valid && backward_j_valid
            
            #f(x_i - h_i, x_j - h_j)
            para_idy_jdy = copy(x)
            para_idy_jdy[i] -= hi
            para_idy_jdy[j] -= hj
            f_idy_jdy = fcn(para_idy_jdy)

            #f(x_i + h_i, x_j - h_j)
            para_idx_jdy = copy(x)
            para_idx_jdy[i] += hi
            para_idx_jdy[j] -= hj
            f_idx_jdy = fcn(para_idx_jdy)
 
            #f(x_i - h_i, x_j + h_j)
            para_idy_jdx = copy(x)
            para_idy_jdx[i] -= hi
            para_idy_jdx[j] += hj
            f_idy_jdx = fcn(para_idy_jdx)

            #f(x_i + h_i, x_j + h_j)
            para_idx_jdx = copy(x)
            para_idx_jdx[i] += hi
            para_idx_jdx[j] += hj
            f_idx_jdx = fcn(para_idx_jdx)

             hessdiag[k] = (f_idy_jdy + f_idx_jdx - f_idx_jdy - f_idy_jdx) / (4*(hi * hj))
        else
            hessdiag[k], _ = hess_offdiag_element_o1(fcn, x, i, j, σ_xσ_y, lb, ub; verbose=verbose)
 


        end

    end

    println(verbose, :high, "Values: $(hessdiag)")

    value = (hessdiag[3]+hessdiag[4])/2

    if value == 0 || σ_xσ_y == 0
        ρ_xy = 0
    else
        ρ_xy = value / σ_xσ_y
    end

    if ρ_xy < -1 || 1 < ρ_xy
        value = 0
    end

    println(verbose, :high, "Value used: $value")
    println(verbose, :high, "Correlation: $ρ_xy")

    return value, ρ_xy
end


# Compute diag element
function hess_diag_element_o2(fcn::Function,
                           x::Vector{T},
                           i::Int,
                           lb,
                           ub;
                           ndx::Int=6,
                           check_neg_diag::Bool=false,
                           verbose::Symbol=:none) where T<:AbstractFloat
    # Setup
    n_para = length(x)
    dxscale  = ones(n_para, 1)
    dx       = exp.(-(6:2:(6+(ndx-1)*2))')
    hessdiag = zeros(ndx, 1)

    println(verbose, :low, "Hessian element: ($i, $i)")

    # Diagonal element computation
    for k = 3:4
        hi = dx[k]*dxscale[i]

        forward_i_valid = (x[i] + 2 * hi <= ub[i])
        backward_i_valid = (x[i] - 2* hi >= lb[i])
        

       
         
        #do center difference O(h^2)
        if forward_i_valid && backward_i_valid
            paradx = copy(x)
            parady = copy(x)

            paradx[i] += hi
            parady[i] -= hi

            fx  = fcn(x)
            fdx = fcn(paradx)
            fdy = fcn(parady)

            hessdiag[k]  = (-2fx + fdx + fdy) / (dx[k]*dxscale[i])^2
    

        #do backward difference O(h^2)
        elseif backward_i_valid
            parady = copy(x)
            para2dy = copy(x)
            para3dy = copy(x)
            
            parady[i] -= hi
            para2dy[i] -= (2*hi)
            para3dy[i] -= (3*hi)

            fx  = fcn(x)
            fdy = fcn(parady)
            f2dy = fcn(para2dy)
            f3dy = fcn(para3dy)


            hessdiag[k]  = (2*fx - 5*fdy + 4*f2dy - f3dy) / (dx[k]*dxscale[i])^2
    
        #do forward difference O(h^2)
        elseif forward_i_valid
            paradx = copy(x)
            para2dx = copy(x)
            para3dx = copy(x)
            
            paradx[i] += hi
            para2dx[i] += (2*hi)
            para3dx[i] += (3*hi)

            fx  = fcn(x)
            fdx = fcn(paradx)
            f2dx = fcn(para2dx)
            f3dx = fcn(para3dx)
            
            hessdiag[k]  = (2*fx - 5*fdx + 4*f2dx - f3dx) / (dx[k]*dxscale[i])^2
   
        end
    end

    println(verbose, :high, "Values: $(hessdiag)")

    value = (hessdiag[3]+hessdiag[4])/2

    if check_neg_diag && value < 0
        error("Negative diagonal in Hessian")
    end

    println(verbose, :high, "Value used: $value")

    return value
end

# Compute off diag element
function hess_offdiag_element_o1(fcn::Function,
                              x::Vector{T},
                              i::Int,
                              j::Int,
                              σ_xσ_y::T,
                              lb,
                              ub;
                              ndx::Int=6,
                              method::Symbol=:none,
                              verbose::Symbol=:none) where T<:AbstractFloat
    # Setup
    n_para = length(x)
    dxscale  = ones(n_para, 1)
    dx       = exp.(-(6:2:(6+(ndx-1)*2))')
    hessdiag = zeros(ndx, 1)

    # Computation
    println(verbose, :low, "Hessian element: ($i, $j)")

    for k = 3:4
        hi = dx[k]*dxscale[i]
        hj = dx[k]*dxscale[j]

        forward_i_valid = (x[i] + hi <= ub[i])
        forward_j_valid = (x[j] + hj <= ub[j])
        backward_i_valid = (x[i] - hi >= lb[i])
        backward_j_valid = (x[j] - hj >= lb[j])
        

        #do forward(i)-backwards(j) O(h)
        if forward_i_valid && backward_j_valid
            paradx = copy(x)
            parady = copy(x)
            paradx[i] += hi
            parady[j] -= hj

            paradxdy    = copy(paradx)
            paradxdy[j] -= hj

            fx    = fcn(x)
            fdx   = fcn(paradx)
            fdy   = fcn(parady)
            fdxdy = fcn(paradxdy)

            hessdiag[k]  = -(fx - fdx - fdy + fdxdy) / (dx[k]*dx[k]*dxscale[i]*dxscale[j])

        #do forward(j)-backwards(i) O(h)
        elseif forward_j_valid && backward_i_valid
            paradx      = copy(x)
            parady      = copy(x)
            paradx[j]   += hj
            parady[i]   -= hi

            paradxdy    = copy(paradx)
            paradxdy[i] -= hi

            fx    = fcn(x)
            fdx   = fcn(paradx)
            fdy   = fcn(parady)
            fdxdy = fcn(paradxdy)

            hessdiag[k]  = -(fx - fdx - fdy + fdxdy) / (dx[k]*dx[k]*dxscale[i]*dxscale[j])

        #do backward-backward O(h)
        elseif backward_i_valid && backward_j_valid
            paradx = copy(x)
            parady = copy(x)

            paradx[i] -= hi
            parady[j] -= hj

            paradxdy    = copy(paradx)
            paradxdy[j] -= hj

            fx    = fcn(x)
            fdx   = fcn(paradx)
            fdy   = fcn(parady)
            fdxdy = fcn(paradxdy)

            hessdiag[k]  = (fx - fdx - fdy + fdxdy) / (dx[k]*dx[k]*dxscale[i]*dxscale[j])

        #do forward-forward O(h)
        elseif forward_i_valid && forward_j_valid
            paradx = copy(x)
            parady = copy(x)

            paradx[i] += hi
            parady[j] += hj

            paradxdy    = copy(paradx)
            paradxdy[j] += hj

            fx    = fcn(x)
            fdx   = fcn(paradx)
            fdy   = fcn(parady)
            fdxdy = fcn(paradxdy)

            hessdiag[k]  = (fdxdy - fdx - fdy + fx) / (dx[k]*dx[k]*dxscale[i]*dxscale[j])

        end

    end

    println(verbose, :high, "Values: $(hessdiag)")

    value = (hessdiag[3]+hessdiag[4])/2

    if value == 0 || σ_xσ_y == 0
        ρ_xy = 0
    else
        ρ_xy = value / σ_xσ_y
    end

    if ρ_xy < -1 || 1 < ρ_xy
        value = 0
    end

    println(verbose, :high, "Value used: $value")
    println(verbose, :high, "Correlation: $ρ_xy")

    return value, ρ_xy
end
