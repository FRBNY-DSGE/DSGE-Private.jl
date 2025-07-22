using CSV, DataFrames, JLD2


function InOutData()
    data_df = DataFrame(CSV.read("/data/dsge_data_dir/proc/dsge/briefings/202412/Model_Data/pricesticky_emissions_aggregated.csv", DataFrame))
    InOut = Array(CSV.read("/data/dsge_data_dir/proc/dsge/briefings/202412/Model_Data/IO_70Sector.csv", DataFrame))





    subset_inds = Array(select(data_df, :freqch_IO => x-> x .> 0.0)) #Need to do it this way to subset the IO matrix correctly
    data_df2 = subset(data_df, :freqch_IO => x-> x .> 0.0)
    InOut = InOut[subset_inds[:], subset_inds[:]]
    n = size(data_df2, 1)

    IO_unscaled = InOut .* repeat(data_df2[!, :ind_totout]', n)




    #Parameters
    params = Dict()
    params["dt"] = 1/12 #Length of a period; monthly here
    params["β"] = 0.96^params["dt"]
    params["ρ_τ"] = 0.4^params["dt"]
    params["ζ"] = 2.
    params["η"] = 0.6
    params["ν"] = 0.2
    params["ξ"] = 0.1


    params["oil"] = 3.
    params["gas"] = 4.
    params["coal"] = 5.
    params["dirty"] = [params["oil"] params["gas"] params["coal"]]

    params["cons"] = data_df2[!, :ind_findemand] - data_df2[!, :ind_imports]
    params["ind_totout"] = params["cons"] .+ sum(IO_unscaled, dims=2)

    params["ii_usage"] = sum(IO_unscaled, dims=1)'
    params["mx"] = data_df2[!, :ind_comp] .+ params["ii_usage"]

    params["inpshare"] = IO_unscaled'.*repeat(1 ./ params["mx"], 1, n)
    params["labshare"] = data_df2[!, :ind_comp] ./ params["mx"]
    params["totintshare"] = params["ii_usage"] ./ params["mx"]
    params["to_mx"] = params["ind_totout"] ./ params["mx"]
    params["taxshare"] = data_df2[!, :emissions_raw] ./ params["mx"]


    IO2 = IO_unscaled .* repeat(1 ./ params["ind_totout"], 1, n)

    params["ω_tilde"] = IO_unscaled' .* repeat(1 ./ params["ii_usage"], 1, n)

    #Expenditure shares
    params["gam"] = params["cons"]/sum(params["cons"])

    params["pc_px"] = params["cons"]./params["ind_totout"]

    params["core"] = data_df2[!, :core]
    params["food"] = data_df2[!, :food]
    params["energy"] = data_df2[!, :energy]

    #Price index weights
    params["gam_core"] = (params["gam"] .* params["core"])/sum(params["gam"] .* params["core"])
    params["gam_food"] = (params["gam"] .* params["food"])/sum(params["gam"] .* params["food"])
    params["gam_energy"] = (params["gam"] .* params["energy"])/sum(params["gam"] .* params["energy"])


    #Intermediate shares of energy and non-energy inputs by sector
    params["omE_unscaled"] = params["ω_tilde"] .* (params["energy"]')
    params["varsig_tilde"] = sum(params["omE_unscaled"], dims = 2) #Share of energy in total
    params["ωE_tilde"] = params["omE_unscaled"] ./ params["varsig_tilde"] #Share of j in energy
    params["omN_unscaled"] = params["ω_tilde"] .* (1 .- params["energy"]')
    params["ωN_tilde"] = params["omN_unscaled"] ./ (1 .- params["varsig_tilde"]) #Share of j in non-energy

    #Emissions shares
    params["ei"] = data_df2[!, :emissions_raw] / sum(data_df2[!, :emissions_raw])
    #Monthly frequency of price change
    params["freqch"] = 1 ./ data_df2[!, :inv_freqch_IO]

    params["Θ"] = 1 .- params["freqch"]
    params["invkap"] = params["Θ"] ./ ((1 .- params["β"] * params["Θ"]) .* (1 .- params["Θ"]))
    params["τ"] = 0

    params["Θw"] = 0.9^(1/3)
    params["kapw"] = (1- params["β"]*params["Θw"]) * (1-params["Θw"])/params["Θw"]
    params["invkapw"] = 1/params["kapw"] #Slope of inverse wage phillips curve

    #Policy rule
    params["mp_cpi_infl"] = 1. #weight on cpi inflation in MP rule
    params["mp_cons"] = 0. #weight on consumption in MP rule
    params["mp_cstar"] = 0. #weight on potential consumption
    params["mp_core"] = 0. #Weight on core inflation

    params["τ_new"] = 0.1

    params["IO"] = InOut
    params["IO2"] = IO2
    params["IO_unscaled"] = IO_unscaled
    return params
end
