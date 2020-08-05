using DSGE, Test, Plots

m = HetDSGEGovDebt(; ref_dir = joinpath(dirname(@__FILE__), "../../../../../src/models/heterogeneous/het_dsge_gov_debt/reference"))
# DSGE.steadystate!(m)
DSGE.method3_steadystate!(m; maxit = 20, use_quadrature = false)
