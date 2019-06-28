"""
This script holds planned tests that will be written
once I have a better idea of what the exact structure
of the PoolModel, etc. type will be

PoolModel type tests
- Should be able to solve and estimate all underlying models correctly
- keyword: static = true -> changes the priors of PoolModel for you
- check all other keywords
- check instantiation is properly done

Solve tests
- solve(PoolModel) grabs all the underlying ParticleCloud data from smc output
  a. compare to actual ParticleCloud data

Estimate, set as long test
- estimate(PoolModel) estimates the dynamic prediction pool, compare to Matlab output
  (or something like that, or replicate figures properly, or there are numbers in
  the paper I should be able to replicate)

Check states sum to one
"""
