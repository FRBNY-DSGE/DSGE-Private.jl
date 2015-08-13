# To Do

## Short-Run
- Investigate how the discrepancy between Matlab and Julia computed Hessians affects how the posterior distribution is explored during Metropolis-Hastings
- Step through `csminwel` and confirm that the outputted mode as well as intermediate values are the same in Matlab and Julia
- Thorough testing of Metropolis-Hastings

## Medium-Run
- Speed up the Kalman filter (right now 2x slower than Matlab)
- Complete unit tests for estimation step
- *Model completion: implement forecasting, plotting, and loading data*

## Long-Run
- Consider reimplementing eqcond, transition, and measurement matrices using sparse matrices
- Add option for `eqcond` to take in Γ0, Γ1, etc. matrices as arguments so we can reassign to those matrices instead of reallocating `zeros(82, 82)` each time
- Same with other functions that also do this
- Replace csminwel, Hessian computation, etc. with external packages
- Compute likelihood without partitioning
- Unmodulify FinancialFrictionsExt
- Convert all .mat files to .h5
- Standardize variable names (TTT, etc.)
- Get rid of src/init/