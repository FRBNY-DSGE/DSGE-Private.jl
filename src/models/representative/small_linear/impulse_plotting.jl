# Make sure you are using the correct Julia environment before running this
# RUN CODE: julia> include("<your_file_path>/impulse_plotting.jl")

# Initialize model
using DSGE
m = SmallLinear()
system = compute_system(m)

# ENTER horizon, shocks used, shock values
horizon = 20
shock_names = [:r_sh, :r_f_sh]
shock_values = [100.0, 100.0]

# Generate responses for states
states_irf, obs_irf, pseudo_irf = impulse_responses(m, system, horizon, shock_names, shock_values)
states_irf_by_shock = Dict()
for i in 1:length(shock_names)
    states_irf_by_shock[shock_names[i]] = states_irf[:,:,i]
end

# Create plots
using Plots
t = 1:horizon
states = collect(keys(m.endogenous_states))
plots_by_shock = Dict()
for i in 1:length(shock_names)
    plots_by_shock[shock_names[i]] = Dict()
end
for i in 1:length(shock_names)
    for j in 1:length(states)
        s = string(states[j],"__", shock_names[i])
        plots_by_shock[shock_names[i]][states[j]] = 
            plot(t, states_irf_by_shock[shock_names[i]][j,:], title = s)
    end
end

# CHOOSE plots to write out (Change :r_sh and :c_t to specify shock and state variable)
plot(plots_by_shock[shock_names[m.exogenous_shocks[:r_sh]]][states[m.endogenous_states[:c_t]]],
    plots_by_shock[shock_names[m.exogenous_shocks[:r_sh]]][states[m.endogenous_states[:y_t]]], 
    plots_by_shock[shock_names[m.exogenous_shocks[:r_sh]]][states[m.endogenous_states[:r_n_t]]], 
    plots_by_shock[shock_names[m.exogenous_shocks[:r_sh]]][states[m.endogenous_states[:e_rt]]], 
    plots_by_shock[shock_names[m.exogenous_shocks[:r_sh]]][states[m.endogenous_states[:c_t_f]]], 
    plots_by_shock[shock_names[m.exogenous_shocks[:r_sh]]][states[m.endogenous_states[:y_t_f]]], 
    plots_by_shock[shock_names[m.exogenous_shocks[:r_sh]]][states[m.endogenous_states[:r_n_t_f]]], 
    legend = false)
savefig("DSGE_plots.pdf")




