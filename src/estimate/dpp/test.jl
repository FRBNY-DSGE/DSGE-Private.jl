"""
This script holds planned tests that will be written
once I have a better idea of what the exact structure
of the PoolModel, etc. type will be

PoolModel type tests
- Should be able to solve all underlying models correctly
- check instantiation is properly done
    a. Need to use Matlab to check whether you forecast to T - h or to T
    b. Load particle clouds
    c. Load data in properly
- use Matlab code to check Figure 4/9 results (predictive densities)


Check TPF all works properly
Check if you feed in a non-dynamic model to the dynamic model, it estimates the same thing
- with AnSchorfheide
Check if you feed in a simple dynamic measurement (only dynamic in period T and T - 1)
    that it works and returns the same thing if you run period 1:T - 1 and different otherwise
- with AnSchorfheide
Check weight_kernel, mutation, whole TPF works in test file
Compare TPF (with zero tempering option, figure this out) to filtered densities from
one round of filtering in the Matlab code

Estimate, set as long test
- estimate(PoolModel) with static pool and replicate results or at least produce them
- estimate(PoolModel) estimates the dynamic prediction pool, compare to Matlab output
  (or something like that, or replicate figures properly, or there are numbers in
  the paper I should be able to replicate)
"""
