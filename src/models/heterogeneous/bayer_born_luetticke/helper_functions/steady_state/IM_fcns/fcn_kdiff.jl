"""
```
Kdiff(K_guess, m, initial = true,
      Vm_guess = [], Vk_guess = [], distr_guess = [];
      verbose = :none, coarse = false,
      parallel = false)
```
Calculate the difference between the capital stock that is assumed and the capital
stock that prevails under that guessed capital stock's implied prices when
households face idiosyncratic income risk (Aiyagari model).

### Inputs
- `K_guess::Float64`: capital stock guess
- `m::BayerBornLuetticke`: model object representing the heterogeneous agent DSGE of Bayer, Born, and Luetticke
- `initial::Bool`: is `K_guess` the initial guess?
- `Vm_guess::AbstractArray`, `Vk_guess::AbstractArray`, `distr_guess::AbstractArray`: guesses
    for marginal value functions and distributions

### Keyword Arguments
- `verbose::Symbol = :none`: verbosity of print statements with values allowed among `[:none, :low, :high]`.
- `coarse::Bool = false`: use a coarse grid when true.

"""
function Kdiff(K_guess::Float64, grids::OrderedDict, idiosyncratic_gridpts::Tuple{Array{T1,1}, Array{T1,1}, Array{T1,1}}, distr::Array{T1,3}, θ::NamedTuple, param_dict::Dict,
               initial::Bool = true, Vm_guess::AbstractArray = zeros(1, 1, 1),
               Vk_guess::AbstractArray = zeros(1, 1, 1), distr_guess::AbstractArray = zeros(1, 1, 1);
               verbose::Symbol = :none, coarse::Bool = false,
               parallel::Bool = false, ϵ::Float64 = 1e-5,
               max_value_function_iters::Int64 = 1000,
               n_direct_transition_iters::Int64 = 10000,
               kfe_method::Symbol = :krylov, ny::Int64 = 6) where {T1 <: Real}

    # Some type declarations b/c grids is an OrderedDict
    # => ensures type stability, or else unnecessary allocations are made
    Π                   = grids[:Π]::Matrix{T1}
    y_grid              = get_gridpts(grids, :y_grid)::Vector{T1}
    m_ndgrid            = grids[:m_ndgrid]::Array{T1, 3}
    k_ndgrid            = grids[:k_ndgrid]::Array{T1, 3}
    y_ndgrid            = grids[:y_ndgrid]::Array{T1, 3}
    H                   = grids[:H]::T1
    HW                  = grids[:HW]::T1

    # TODO: check if there's a notable speed up by only passing settings, grids,
    #       and parameters into this function as kwargs
    #----------------------------------------------------------------------------
    # Calculate other prices from capital stock
    #----------------------------------------------------------------------------
    N           = _bbl_employment(K_guess, 1.0 / (param_dict[:μ_p] * param_dict[:μ_w]), param_dict[:α],      # employment
                                  param_dict[:τ_lev], param_dict[:τ_prog], param_dict[:γ])
    w           = _bbl_wage(K_guess, 1.0 / param_dict[:μ_p], N, param_dict[:α])                     # wages
    rk          = _bbl_interest(K_guess, 1.0 / param_dict[:μ_p], N, param_dict[:α], param_dict[:δ_0])        # Return on illiquid asset
    profits     = (1.0 - 1.0 / param_dict[:μ_p]) .* _bbl_output(K_guess, 1.0, N, param_dict[:α])    # Profit income
    RB          = param_dict[:RB] / param_dict[:π]                                                  # Real return on liquid assets
    neg_liq_ret = RB + param_dict[:Rbar]
    eff_int     = [x <= 0. ? neg_liq_ret : RB for x in m_ndgrid]        # effective rate depending on assets
    GHHFA       = (param_dict[:γ] + param_dict[:τ_prog]) / (param_dict[:γ] + 1.0)                            # transformation (scaling) for composite good

    #----------------------------------------------------------------------------
    # Array (inc) to store incomes
    # inc[1] = labor income , inc[2] = rental income,
    # inc[3]= liquid assets income, inc[4] = capital liquidation income
    #----------------------------------------------------------------------------
    Paux            = grids[:Paux]::Matrix{T1}                                 # Grab ergodic income distribution from transitions
    distr_y         = Paux[1, :]                                                 # stationary income distribution
    inc             = Array{Array{Float64, 3}}(undef, 4)                         # container for income
    mcw             = 1.0 / param_dict[:μ_w]                                              # wage markup

    # gross (labor) incomes
    eff_unit_inc    = mcw * w * N / H         # gross labor income per efficiency unit
    incgross        = y_grid .* eff_unit_inc  # gross income workers (wages)
    incgross[end]   = y_grid[end] * profits   # gross income entrepreneurs (profits)

    # net (labor) incomes
    incnet          = param_dict[:τ_lev] * incgross .^ (1.0 - param_dict[:τ_prog])
    incnet[end]     = param_dict[:τ_lev] .* (y_grid[end] .* profits).^(1. - param_dict[:τ_prog])

    # average tax rate
    av_tax_rate     = dot((incgross - incnet), distr_y) / dot(incgross, distr_y)
    # ny              = get_setting(m, coarse ? :coarse_ny : :ny)
    inc[1]          = (GHHFA * param_dict[:τ_lev]) .* (y_ndgrid .* eff_unit_inc) .^ (1.0 - param_dict[:τ_prog]) .+
        ((1.0 - mcw) * w * N * (1.0 - av_tax_rate) * HW)         # labor income net of taxes incl. union profits
    inc[1][:,:,end] = param_dict[:τ_lev] * (view(y_ndgrid, :, :, ny) * profits) .^ (1.0 - param_dict[:τ_prog]) # profit income net of taxes

    # incomes out of wealth # TODO: replace these steps OR use list comprehension later on
    inc[2]          = rk .* k_ndgrid                                  # rental income
    inc[3]          = eff_int .* m_ndgrid                             # liquid asset income
    inc[4]          = k_ndgrid                                        # capital liquidation income (q=1 in steady state)

    #----------------------------------------------------------------------------
    # Initialize policy function (guess/stored values)
    #----------------------------------------------------------------------------

    # initial guess consumption and marginal values (if not set)
    if initial # TODO: pass in m_ndgrid .> 0. if that's already calculated elsewhere
        c_guess     = inc[1] .+ inc[2] .* (inc[2] .> 0) .+ inc[3] .* (m_ndgrid .> 0.)
        if any(x -> x < 0., c_guess)
            @warn "negative consumption guess"
        end
        Vm          = eff_int .* _bbl_mutil(c_guess, param_dict[:ξ])
        Vk          = (rk + param_dict[:λ]) .* _bbl_mutil(c_guess, param_dict[:ξ])
        #distr       = get_untransformed_values(m[:distr_star])::Array{T1, 3}
    else
        Vm          = Vm_guess
        Vk          = Vk_guess
        #distr       = distr_guess
    end

    #----------------------------------------------------------------------------
    # Calculate supply of funds for given prices
    #----------------------------------------------------------------------------
    #θ               = parameters2namedtuple(m)
    KS              = Ksupply(RB, 1.0 + rk, grids, idiosyncratic_gridpts, θ, Vm, Vk, distr, inc, eff_int;
                              verbose = verbose, coarse = coarse, parallel = parallel,
                              ϵ=ϵ, max_value_function_iters=max_value_function_iters,
                              n_direct_transition_iters=n_direct_transition_iters,
                              kfe_method=kfe_method)
    K               = KS[1]                                                     # capital
    Vm              = KS[end-2]                                                 # marginal value of liquid assets
    Vk              = KS[end-1]                                                 # marginal value of illiquid assets
    distr           = KS[end]                                                   # stationary distribution
    diff            = K - K_guess                                               # excess supply of funds
    return diff, Vm, Vk, distr
end
