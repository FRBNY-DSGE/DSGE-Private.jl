# To Do

## Short-Run
- Thorough testing of Metropolis-Hastings
..- Investigate how the discrepancy between Matlab and Julia computed Hessians affects how the posterior distribution is explored during Metropolis-Hastings
..- Step through `csminwel` and confirm that the outputted mode as well as intermediate values are the same in Matlab and Julia
- Complete unit tests for estimation step
..- Write program to print out TeX tables of posterior steady state values :white_check_mark:

## Medium-Run
- Speed up the Kalman filter (right now 2x slower than Matlab)
- *Model completion: implement forecasting, plotting, and loading data*


## Long-Run
- Consider reimplementing eqcond, transition, and measurement matrices using sparse matrices
- Add option for `eqcond` to take in Γ0, Γ1, etc. matrices as arguments so we can reassign to those matrices instead of reallocating `zeros(82, 82)` each time
- Same with other functions that also do this
- Replace csminwel, Hessian computation, etc. with external packages
- Compute likelihood without partitioning
- Unmodulify FinancialFrictionsExt
- Convert all .mat files to .h5 :white_check_mark:
- Standardize variable names (TTT, etc.)
- Get rid of src/init/