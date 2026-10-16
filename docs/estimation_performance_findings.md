# Estimation performance findings for Model1002

The estimation path updates model parameters, constructs a state-space system,
and runs the likelihood filter for each proposed draw. The observations below
come from the code path in `src/estimate/posterior.jl`, `src/solve/solve.jl`, and
the Model1002 implementation. They have not yet been ranked by a runtime profile.

## 1. Variance-only draws recomputed the steady state — implemented

`update!` called `steadystate!` for every draw. Model1002's steady state includes
a nonlinear root solve for the financial-frictions block. Changes to ordinary
shock standard deviations such as `σ_g` and `σ_b` do not enter those
calculations. `src/models/representative/m1002/estimation_cache.jl` now defines
an explicit, conservative set of parameters whose changes may skip the
steady-state calculation. All other changes retain the normal update path.
The optimization transform also called `steadystate!` a second time immediately
after `update!`; that duplicate calculation has been removed.

## 2. Identical Gensys problems were solved repeatedly — implemented

Two regimes with the same `Γ0`, `Γ1`, `C`, `Ψ`, and `Π` have the same raw Gensys
solution even if their `QQ` matrices differ. Model1002 now keeps a bounded
per-model cache of valid raw Gensys solutions, keyed by exact canonical matrix
contents. The solver reuses these solutions across regimes and parameter draws.
State augmentation and measurement are still rebuilt for each regime or draw,
so `QQ` and other regime-specific quantities are updated. The existing manual
`identical_eqcond_regimes` and `perfect_credibility_identical_transitions`
settings can still avoid more work where their equivalence assumptions hold.

This cache does not treat all `σ_*` parameters as variance-only. For example,
proportional anticipated-shock coefficients and `κ_std_bcshocks` change `Ψ`,
while `σ_c` changes the structural equations. Those inputs cannot reuse a full
Gensys solution. A future refinement could cache a structural factorization and
recompute only shock responses when `Ψ` changes.

## 3. SMC time tempering builds the system twice per proposal — pending

`src/estimate/smc/mutation.jl` calls `likelihood` once for new data and once
for old data. Both calls rebuild the same state-space system. Passing a single
system to both filters would avoid the duplicate construction. The code also
evaluates the old-data likelihood when the new-data likelihood is `-Inf`, even
though the proposal is already rejected.

## 4. Estimation constructs more regimes than the filter uses — pending

`compute_system_helper` reads `n_hist_regimes` but solves and constructs
measurements for all `n_regimes`. The likelihood filter uses regimes in the
observed sample. Historical expectations can depend on future transitions, so
the needed transition set must be determined before restricting construction;
forecast-only measurement equations are a narrower first optimization.

## 5. SMC proposal distributions are rebuilt inside the mutation loop — pending

`src/estimate/smc/mutation.jl` repeatedly constructs block multivariate normal
distributions and proposal-density objects while their covariance matrices are
unchanged. Precomputing block factorizations once per mutation call should
reduce allocations and matrix work.

## Validation and measurement

The intended invariants are identical `TTT`, `RRR`, and `CCC` for variance-only
changes; changed `QQ` where the variance changed; and a cache miss whenever a
canonical Gensys input changes. Timing should compare uncached and cached
likelihood evaluations after Julia compilation for representative estimation
subspecs, especially regime-switching ones. Runtime gains remain unmeasured.
