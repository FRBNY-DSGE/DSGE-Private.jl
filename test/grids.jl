using DSGE, Test

x = collect(0.:.25:1.)
a, b = DSGE.ndgrid(x, x)
@test a == ([x[i] for i in 1:length(x), j in 1:length(x)])
@test b == ([x[j] for i in 1:length(x), j in 1:length(x)])

a, b, c = DSGE.ndgrid(x, x, x)
@test a == ([x[i] for i in 1:length(x), j in 1:length(x), k in 1:length(x)])
@test b == ([x[j] for i in 1:length(x), j in 1:length(x), k in 1:length(x)])
@test c == ([x[k] for i in 1:length(x), j in 1:length(x), k in 1:length(x)])
