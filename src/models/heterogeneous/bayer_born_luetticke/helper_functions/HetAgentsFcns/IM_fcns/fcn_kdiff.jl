"""
```
Kdiff(K_guess,n_par,m_par)
```
Calculate the difference between the capital stock that is assumed and the capital
stock that prevails under that guessed capital stock's implied prices when
households face idiosyncratic income risk (Aiyagari model).

Requires global functions `employment(K,A,m_par)`, `interest(K,A,N,m_par)`,
`wage(K,A,N,m_par)`, `output(K,TFP,N,m_par)`, and [`Ksupply()`](@ref).

# Arguments
- `K_guess::Float64`: capital stock guess
"""
function Kdiff(K_guess::Float64, m::BayerBornLuetticke,
               initial::Bool = true, Vm_guess::AbstractArray = zeros(1, 1, 1),
               Vk_guess::AbstractArray = zeros(1, 1, 1), distr_guess::AbstractArray = zeros(1, 1, 1);
               verbose::Symbol = :none, coarse::Bool = false)
    # TODO: check if there's a notable speed up by only passing settings, grids,
    #       and parameters into this function as kwargs
    #----------------------------------------------------------------------------
    # Calculate other prices from capital stock
    #----------------------------------------------------------------------------
    N           = _bbl_employment(K_guess, 1.0 / (m[:μ_p] * m[:μ_w]), m[:α],      # employment
                                  m[:τ_lev], m[:τ_prog], m[:γ])
    w           = _bbl_wage(K_guess, 1.0 / m[:μ_p], N, m[:α])                     # wages
    rk          = _bbl_interest(K_guess, 1.0 / m[:μ_p], N, m[:α], m[:δ_0])        # Return on illiquid asset
    profits     = (1.0 - 1.0 / m[:μ_p]) .* _bbl_output(K_guess, 1.0, N, m[:α])    # Profit income
    RB          = m[:RB] / m[:π]                                                  # Real return on liquid assets
    neg_liq_ret = RB + m[:Rbar]
    eff_int     = [x <= 0. ? neg_liq_ret : RB for x in m.grids[:m_ndgrid]]        # effective rate depending on assets
    GHHFA       = (m[:γ] + m[:τ_prog]) / (m[:γ] + 1.0)                            # transformation (scaling) for composite good

    #----------------------------------------------------------------------------
    # Array (inc) to store incomes
    # inc[1] = labor income , inc[2] = rental income,
    # inc[3]= liquid assets income, inc[4] = capital liquidation income
    #----------------------------------------------------------------------------
    Paux            = m.grids[:Paux]                                             # Grab ergodic income distribution from transitions
    distr_y         = Paux[1, :]                                                 # stationary income distribution
    inc             = Array{Array{Float64, 3}}(undef, 4)                         # container for income
    mcw             = 1.0 / m[:μ_w]                                              # wage markup

    # gross (labor) incomes
    incgross        = get_gridpts(m, :y_grid) .* (mcw * w * N / m[:H_star])      # gross income workers (wages)
    incgross[end]   = get_gridpts(m, :y_grid)[end] * profits                     # gross income entrepreneurs (profits)

    # net (labor) incomes
    incnet          = m[:τ_lev] * incgross .^ (1.0 - m[:τ_prog])

    # average tax rate
    av_tax_rate     = dot((incgross - incnet), distr_y) / dot(incgross, distr_y)

    # TODO: replace the y_ndgrid calculation with just repeating the incnet vector OR use list comprehension later on
    ny              = get_setting(m, coarse ? :coarse_ny : :ny)
    inc[1]          = (GHHFA * m[:τ_lev]) .* (m.grids[:y_ndgrid] .* (mcw * w * N / m[:H_star])) .^ (1.0 - m[:τ_prog]) .+
        ((1.0 - mcw) * w * N * (1.0 - av_tax_rate) * m[:HW_star])         # labor income net of taxes incl. union profits
    inc[1][:,:,end] = m[:τ_lev] * (view(m.grids[:y_ndgrid], :, :, ny) * profits) .^ (1.0 - m[:τ_prog]) # profit income net of taxes

    # incomes out of wealth # TODO: replace these steps OR use list comprehension later on
    inc[2]          = rk .* m.grids[:k_ndgrid]                                  # rental income
    inc[3]          = eff_int .* m.grids[:m_ndgrid]                             # liquid asset income
    inc[4]          = m.grids[:k_ndgrid]                                        # capital liquidation income (q=1 in steady state)

    #----------------------------------------------------------------------------
    # Initialize policy function (guess/stored values)
    #----------------------------------------------------------------------------

    # initial guess consumption and marginal values (if not set)
    if initial # TODO: pass in m.grids[:m_ndgrid] .> 0. if that's already calculated elsewhere
        # c_guess     = inc[1] .+ inc[2] .* (inc[2] .> 0) .+ inc[3] .* (m.grids[:m_ndgrid] .> 0.)
        c_guess     = copy(inc[1])
        pos_rental_income = inc[2] .> 0.
        pos_liquid_assets = m.grids[:m_ndgrid] .> 0.
        c_guess[pos_rental_income] .+= inc[2][pos_rental_income]
        c_guess[pos_liquid_assets] .+= inc[3][pos_liquid_assets]
        if any(x -> x < 0., c_guess)
            @warn "negative consumption guess"
        end
        Vm          = eff_int .* _bbl_mutil(c_guess, m[:ξ])
        Vk          = (rk + m[:λ]) .* _bbl_mutil(c_guess, m[:ξ])
        distr       = get_untransformed_values(m[:distr_star])
    else
        Vm          = Vm_guess
        Vk          = Vk_guess
        distr       = distr_guess
    end

    #----------------------------------------------------------------------------
    # Calculate supply of funds for given prices
    #----------------------------------------------------------------------------
    KS              = Ksupply(RB, 1.0 + rk, m, Vm, Vk, distr,
                              inc, eff_int; verbose = verbose, coarse = coarse)
    K               = KS[1]                                                     # capital
    Vm              = KS[end-2]                                                 # marginal value of liquid assets
    Vk              = KS[end-1]                                                 # marginal value of illiquid assets
    distr           = KS[end]                                                   # stationary distribution
    diff            = K - K_guess                                               # excess supply of funds
    return diff, Vm, Vk, distr
end
