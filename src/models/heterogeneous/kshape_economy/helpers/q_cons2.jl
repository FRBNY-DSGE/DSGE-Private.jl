"""
    q_cons2(c_a_guess, c_n_guess, W_fc, nn, AProb, mu_dist, meshes, param, grid, tau_w)

Compute average consumption including home production/leisure component.

This function calculates the consumption aggregated over employed and unemployed agents,
weighted by their distribution and adjustment probabilities.

# Arguments
- `c_a_guess`: Consumption policy with adjustment
- `c_n_guess`: Consumption policy without adjustment
- `W_fc`: Wage rate
- `nn`: Employment rate
- `AProb`: Adjustment probability
- `mu_dist`: Distribution
- `meshes`: State space meshes
- `param`: Parameter dictionary
- `grid`: Grid dictionary
- `tau_w`: Labor tax rate

# Returns
- Average consumption across the distribution
"""
function q_cons2(c_a_guess, c_n_guess, W_fc, nn, AProb, mu_dist, meshes, param, grid, tau_w)
    # Support both string and symbol keys for backward compatibility
    ns = Int(haskey(grid, :ns) ? grid[:ns] : grid["ns"])
    xi = haskey(param, :xi) ? param[:xi] : param["xi"]
    
    # Get productivity mesh (support both Dict and OrderedDict)
    se_mesh = haskey(meshes, :se) ? meshes[:se] : meshes["se"]

    # For employed agents (1:ns): consumption + home production
    employed_a = sum((c_a_guess[:,:,1:ns] .+ (1-tau_w)/(1+xi)*W_fc*se_mesh[:,:,1:ns].*nn) .*
                      AProb[:,:,1:ns] .* mu_dist[:,:,1:ns])

    employed_n = sum((c_n_guess[:,:,1:ns] .+ (1-tau_w)/(1+xi)*W_fc*se_mesh[:,:,1:ns].*nn) .*
                      (1 .- AProb[:,:,1:ns]) .* mu_dist[:,:,1:ns])

    # For unemployed/entrepreneur agents (ns+1:end): consumption only
    unemployed_a = sum(c_a_guess[:,:,ns+1:end] .* AProb[:,:,ns+1:end] .* mu_dist[:,:,ns+1:end])

    unemployed_n = sum(c_n_guess[:,:,ns+1:end] .* (1 .- AProb[:,:,ns+1:end]) .* mu_dist[:,:,ns+1:end])

    # Average consumption across distribution
    out = (employed_a + employed_n + unemployed_a + unemployed_n) / sum(mu_dist)

    return out
end
