# To Do

## Public Launch
- Complete unit tests for estimation step (Tentative deadline: Monday 8/31/2015)
  - Write program to print out TeX tables of posterior steady state values :white_check_mark:
  - Compare tables of moments from Matlab and Julia estimation stages

- Thorough testing of Metropolis-Hastings (Tentative deadline: Monday 9/14/2015)
  - Investigate how the discrepancy between Matlab and Julia computed Hessians affects how the posterior distribution is explored during Metropolis-Hastings
  - Step through `csminwel` and confirm that the outputted mode as well as intermediate values are the same in Matlab and Julia

- Set up public repository (Tentative deadline: Monday 9/21/2015)
  - Seek input from collaborators regarding:
    - Maintenance of public repo
    - Responsiveness to user requests/comments/questions
    - Development workflow following public launch 
  - Add license
  - Write detailed README
  - Strip all identifying Fed information

## Model Completion
- Model completion: implement forecasting, plotting, and loading data
  1. Load data (Tentative deadline: Monday 9/28/2015)
  2. Forecast step (Tentative deadline: Monday 10/19/2015)
    - Learn to write parallel code in Julia
    - Speed up Kalman filter and smoother (right now 2x slower than Matlab)
    - Compute forecasts
  3. Plotting (Tentative deadline: Monday 11/2/2015)


## Continuing Model Development
- Convert all .mat files to .h5 :white_check_mark:
- Unmodulify FinancialFrictionsExt :white_check_mark:
- Get rid of src/init :white_check_mark: 
- Consider reimplementing eqcond, transition, and measurement matrices using sparse matrices
- Add option for `eqcond` to take in Γ0, Γ1, etc. matrices as arguments so we can reassign to those matrices instead of reallocating `zeros(82, 82)` each time
- Same with other functions that also do this
- Replace csminwel, Hessian computation, etc. with external packages
- Compute likelihood without partitioning
- Standardize variable names (TTT, etc.)
