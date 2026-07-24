# This script holds measurement error, which figures
# out what EE and MM matrices to return based on the subspec of the model
function measurement_error(m::DSGEVECM{T}) where {T <: Real}

    # Check if cointegrating terms are added to VECM
    n_coint = DSGE.n_cointegrating(m)

    if n_coint > 0
        EE = zeros(T, n_observables(m) + n_coint, n_observables(m) + n_coint)
        MM = zeros(T, n_observables(m) + n_coint, n_shocks(m))
    else
        EE = zeros(T, n_observables(m), n_observables(m))
        MM = zeros(T, n_observables(m), n_shocks(m))
    end

    return EE, MM
end
