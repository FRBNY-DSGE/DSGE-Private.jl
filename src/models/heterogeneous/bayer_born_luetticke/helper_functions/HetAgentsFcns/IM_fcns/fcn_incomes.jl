function _bbl_incomes(θ::NamedTuple, grids::OrderedDict, KSS::Real, distrSS::Array{<: Real, 3})

    NSS       = _bbl_employment(KSS, 1.0 / (θ[:μ_p] * θ[:μ_w]), θ[:α], θ[:τ_lev], θ[:τ_prog], θ[:γ])
    rkSS      = _bbl_interest(KSS, 1.0 / θ[:μ_p], NSS, θ[:α], θ[:δ_0])
    wSS       = _bbl_wage(KSS, 1.0 / θ[:μ_p], NSS, θ[:α])
    YSS       = _bbl_output(KSS, 1.0, NSS, θ[:α])
    ProfitsSS = (1.0 - 1.0 / θ[:μ_p]) * YSS
    ISS       = θ[:δ_0] * KSS
    RBSS      = θ[:RB]
    GHHFA     = ((θ[:γ] + θ[:τ_prog]) / (θ[:γ] + 1.0))      # transformation (scaling) for composite good

    neg_liq_r = RBSS + θ[:Rbar]
    eff_int   = [x <= 0. ? neg_liq_r : RBSS for x in grids[:m_ndgrid]] # effective rate depending on assets
    H         = grids[:H]
    HW        = grids[:HW]

    incgross = Array{Array{Float64, 3}}(undef, 5) # gross income
    inc = Array{Array{Float64, 3}}(undef, 6)      # net (of taxes) income

    mcw = 1.0 / θ[:μ_w]

    incgross[1] = (grids[:y_ndgrid] .* (1. / θ[:μ_w]) .* wSS .* NSS ./ H) .+ # labor income (NEW)
        (1.0 .- 1.0 ./ θ[:μ_w]) .* wSS .* NSS .* HW
    incgross[2] = rkSS .* grids[:k_ndgrid]                                   # rental income
    incgross[3] = eff_int .* grids[:m_ndgrid]                                # liquid asset Income
    incgross[4] = grids[:k_ndgrid]                                           # points to ndgrid but won't be copying it
    incgross[5] = (1.0 ./ θ[:μ_w]) .* wSS .* NSS .* grids[:y_ndgrid] ./ H    # capital liquidation Income (q=1 in steady state)

    ny                     = size(grids[:y_ndgrid], 3)
    incgross[1][:, :, end] = view(grids[:y_ndgrid], :, :, ny) .* ProfitsSS   # profit income net of taxes
    incgross[5][:, :, end]        = incgross[1][:, :, end]                   # this copies b/c [:, :, end] allocates a new matrix

    # TODO: remove this redundant calculation of inc[1] here and then recalculation later with union profits.
    # Just use calculation later on since inc[1] is not used at all
    inc[1] = (GHHFA .* θ[:τ_lev] .* (grids[:y_ndgrid] .* (1.0 ./ θ[:μ_w]) .*
                                     wSS .* NSS ./ H).^(1.0 - θ[:τ_prog])) .+
                                     ((1.0 .- 1.0 ./ θ[:μ_w]) .* wSS .* NSS) .* HW # labor income (NEW)
    inc[2] = incgross[2]                                                           # rental income
    inc[3] = incgross[3]                                                           # liquid asset income
    inc[4] = incgross[4]                                                           # points to ndgrid but won't be copying it
    inc[5] = θ[:τ_lev] .* ((1.0 ./ θ[:μ_w]) .* wSS .* NSS .*
                           grids[:y_ndgrid] ./ H) .^ (1.0 - θ[:τ_prog]) .*
                           ((1.0 - θ[:τ_prog]) / (θ[:γ] + 1))
    inc[6] = θ[:τ_lev] .* ((1.0 ./ θ[:μ_w]) .* wSS .* NSS .*
                           grids[:y_ndgrid] ./ H) .^ (1.0 - θ[:τ_prog])            # capital liquidation Income (q=1 in steady state)

    inc[1][:, :, end] = θ[:τ_lev] .* (view(grids[:y_ndgrid], :, :, ny) .*
                                      ProfitsSS) .^ (1.0 - θ[:τ_prog])             # profit income net of taxes
    inc[5][:, :, end]       .= 0.0
    inc[6][:, :, end]        = inc[1][:, :, end]                                   # this copies b/c [:, :, end] allocates a new matrix

    taxrev        = incgross[5] - inc[6]
#=    tot_taxrev    = dot(distrSS, taxrev) # if a DimensionMismatch error occurs, it may be because you are using the wrong coarseness
    av_tax_rateSS = tot_taxrev / dot(distrSS, incgross[5]) # of the grid (e.g. you need to recall `DSGE.init_grids!(m; coarse = ...)`=#
    tot_taxrev    = sum(distrSS .* taxrev) # if a DimensionMismatch error occurs, it may be because you are using the wrong coarseness
    av_tax_rateSS = tot_taxrev / sum(distrSS .* incgross[5]) # of the grid (e.g. you need to recall `DSGE.init_grids!(m; coarse = ...)`

    # apply taxes to union profits # TODO: inc[1] and inc[6] are closely related computations => can avoid further redundancies
    inc[1]             = (GHHFA .* θ[:τ_lev] .* (grids[:y_ndgrid] .* (1.0 ./ θ[:μ_w]) .*        # labor income
                                                 wSS .* NSS ./ H) .^ (1.0 - θ[:τ_prog])) .+
                                                     ((1.0 .- 1.0 ./ θ[:μ_w]) .* wSS .* NSS) .* # labor union income
                                                     (1.0 .- av_tax_rateSS) .* HW
    inc[1][:, :, end] .= inc[6][:, :, end]
    inc[6]             = (θ[:τ_lev] .* (grids[:y_ndgrid] .* (1.0 ./θ[:μ_w]) .*     # labor income
                                        wSS .* NSS ./ H) .^ (1.0 - θ[:τ_prog])) .+
                                        ((1.0 .- 1.0 ./ θ[:μ_w]) .* wSS .* NSS) .* # labor union income
                                        (1.0 .- av_tax_rateSS) .* HW
    inc[6][:, :, end] .= inc[1][:, :, end]


    return incgross, inc, NSS, rkSS, wSS, YSS, ProfitsSS, ISS, RBSS, taxrev, tot_taxrev, av_tax_rateSS, eff_int
end
