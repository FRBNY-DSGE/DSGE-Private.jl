function compute_reduction(m::BayerBornLuetticke)

#for now, hard coded are: nstates, the inputs for LOMstate and State2Control,
## nm, nk here are 40 whereas in BBL are 50 each, different lengths of value function compression indices
tNo = get_setting(m, :nk) + get_setting(m, :ny) + get_setting(m, :nm)
tNo4 = length(get_setting(m, :dct_compression_indices)[:copula])
shocks = keys(m.exogenous_shocks)

shock_index = Dict()

j = 0

## shock index alignment is off (for BBL is not necessarily linearly asigned from 1080 onwards)
for i in shocks
   shock_index[i] = tNo + tNo4 -3 + j ## parallels BBL getfield(sr.indexes,i) in their compute_reduction.jl
    j = j + 1
end

#-------------------------------------------------
#STEP 1: Long Run Covariance
#-------------------------------------------------

sd_dictionary = get_setting(m, :shock_to_deviation_dict)
nstates = 1080
SCov = zeros(nstates, nstates)
for i in shocks
    SCov[shock_index[i], shock_index[i]] =(m.parameters[sd_dictionary[i]].value).^2
end
@save "scov.jld2" SCov

StateCOVAR = lyapd(get_setting(m, :LOMstate), SCov)

State2Control = get_setting(m, :State2Control)
ControlCOVAR = State2Control*StateCOVAR*State2Control'
ControlCOVAR = (ControlCOVAR + ControlCOVAR') ./ 2

#--------------------------------------------------
#STEP 2: produce eigenvalue decomposition
#--------------------------------------------------

compression_indices = get_setting(m, :dct_compression_indices)
#nstates = get_setting(m, :n_backward_looking_states)
ntotal = length(compression_indices[:Vm]) + length(compression_indices[:Vk]) + length(compression_indices[:copula])

Dindex = compression_indices[:copula] ## values of the indices do not match
evalS, evecS = eigen(StateCOVAR[Dindex, Dindex])
keepD = abs.(evalS).>maximum(evalS)*get_setting(m, :further_compress_critS)
indKeepD = Dindex[keepD]
nstates_reduced = nstates - length(Dindex) + length(indKeepD)

Vindex = [compression_indices[:Vm] ; compression_indices[:Vk]]
#Vindex = 1081:2119
evalC, evecC = eigen(ControlCOVAR[Vindex .- nstates, Vindex .- nstates])
keepV = abs.(evalC).>maximum(evalC)*get_setting(m, :further_compress_critC)
indKeepV = Vindex[keepV]

#-----------------------------------------------------------
#Setp 3: Put together projection matrices and update indexes
#-----------------------------------------------------------

PRightStates_aux = float(I[1:nstates, 1:nstates])
PRightStates_aux[Dindex, Dindex] = evecS
keep = ones(Bool, nstates)
keep[Dindex[.!keepD]] .= false
m <= Setting(:PRightStates, PRightStates_aux[:, keep])

PRightAll_aux = float(I[1:ntotal, 1:ntotal])
PRightAll_aux[Dindex, Dindex] = evecS
PRightAll_aux[Vindex, Vindex] = evecC
keep = ones(Bool, ntotal)
keep[Dindex[.!keepD]] .= false
keep[Vindex[.!keepV]] .= false
m <= Setting(:PRightAll, PRightAll_aux[:, keep])

update_compression_indices!(m, [:Vm, :Vk, :copula], keepV[keepV][1:2], keepV[keepV][3:end], keepD[keepD])

end
