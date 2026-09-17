using CSV, DataFrames, XLSX

function plot_irfs_rep(m::OnionModel, IRFs::AbstractDict;
                       filestring_addl::String = "")
    """
    This function plots all irfs from compute_irfs_taylor.

    TODO: Document, flexible plotting location, etc

As of now, all plotting will go to "figures/IRFs/*filename*"
    """



    #Need to plot against Kanzig IRFs!

    if get_setting(m, :irf_type) == "taylor"
        kanzig_csv = CSV.read("OnionModel/cps_irfs.csv", DataFrame; header=true)

        # All plots used in Kanzig_irfs.pdf
        kanzig_energy = vcat(0,kanzig_csv[!, "HICPENERGY"])
        kanzig_emissions = vcat(0,kanzig_csv[!, "EMISSIONS"])
        kanzig_cpi = vcat(0,kanzig_csv[!, "HICP"])
        kanzig_it_2yr = vcat(0,kanzig_csv[!, "RATE"])
        kanzig_oil = vcat(0,kanzig_csv[!, "POIL"])
        kanzig_core = vcat(0,kanzig_csv[!, "HICPCORE"])

        scale_factor = 1/IRFs["cumpt_energy"][2] #Normalize impact effect on energy price = 1
        tgrid = -1:get_setting(m,:T)-2

        p1 = plot(tgrid, [vec(scale_factor*IRFs["cumpt_energy"]), vec(kanzig_energy)], linewidth = 2, titlefont=font(8), legend=false,
                  title = "Energy price")
        #labels = ["Our Model" "Kanzig IRFs"])
        #savefig(p1, "OnionModel/figures/IRFs/pt_energy$(filestring_addl).pdf")


        p2 = plot(tgrid, [vec(scale_factor*IRFs["et"]), vec(kanzig_emissions)], linewidth = 2, titlefont=font(8), legend=false,
                  title = "Emissions")
        #labels = ["Our Model" "Kanzig IRFs"])
        #savefig(p2, "OnionModel/figures/IRFs/emissions$(filestring_addl).pdf")


        p3 = plot(tgrid, [vec(scale_factor*IRFs["cumpt_cpi"]), vec(kanzig_cpi)], linewidth = 2, titlefont=font(8), legend=false,
                  title = "Consumer Price Level")
        #labels = ["Our Model" "Kanzig IRFs"])
        #savefig(p3, "OnionModel/figures/IRFs/pt_cpi$(filestring_addl).pdf")

        p4 = plot(tgrid, [vec(scale_factor*IRFs["it_2yr"]), vec(kanzig_it_2yr)], linewidth = 2, titlefont=font(8), legend=false,
                  title = "2yr int. rate")
        #labels = ["Our Model" "Kanzig IRFs"])
        #savefig(p4, "OnionModel/figures/IRFs/it_2yr$(filestring_addl).pdf")

        p5 = plot(tgrid, [vec(scale_factor*IRFs["cumpt_oil"]), vec(kanzig_oil)], linewidth = 2, titlefont=font(8),
                  title = "Oil price",
                  labels = ["Our Model" "Kanzig IRFs"])
        #savefig(p5, "OnionModel/figures/IRFs/pt_oil$(filestring_addl).pdf")

        p6 = plot(tgrid, [vec(scale_factor*IRFs["cumpt_core"]), vec(kanzig_core)], linewidth = 2, titlefont=font(8), legend=false,
                  title = "Core price level")
        #labels = ["Our Model" "Kanzig IRFs"])
        #savefig(p6, "OnionModel/figures/IRFs/pt_core$(filestring_addl).pdf")


        p7 = plot(p1, p2, p3, p4, p5, p6, layout=6)
        savefig(p7, "OnionModel/figures/IRFs/kanzig_IRFs_replication_all$(filestring_addl).pdf")



    elseif get_setting(m, :irf_type) == "oil"

        xfl = XLSX.readxlsx("OnionModel/IRF_oilsupplynews_update.xlsx")
        kanzoil = DataFrame(XLSX.readtable("IRF_oilsupplynews_update.xlsx", "PE"))
        kanzoil68U = DataFrame(XLSX.readtable("IRF_oilsupplynews_update.xlsx", "68U"))
        kanzoil90U = DataFrame(XLSX.readtable("IRF_oilsupplynews_update.xlsx", "90U"))
        kanzoil68L = DataFrame(XLSX.readtable("IRF_oilsupplynews_update.xlsx", "68L"))
        kanzoil90L = DataFrame(XLSX.readtable("IRF_oilsupplynews_update.xlsx", "90L"))
        #The plots only use 90 for now, so I will do the same

        kanzoil_oilr = vcat(0, kanzoil[!, "RealOilPrice"])    #pt_oilr
        kanzoil_oilr90U = vcat(0, kanzoil90U[!, "RealOilPrice"])
        kanzoil_oilr90L = vcat(0, kanzoil90L[!, "RealOilPrice"])


        kanzoil_CPI= vcat(0, kanzoil[!, "CPIHeadline"])       #pt_cpi
        kanzoil_CPI90U = vcat(0, kanzoil90U[!, "CPIHeadline"])
        kanzoil_CPI90L = vcat(0, kanzoil90L[!, "CPIHeadline"])


        kanzoil_CPIE= vcat(0, kanzoil[!, "CPIEnergy"])     #pt_energy
        kanzoil_CPIE90U = vcat(0, kanzoil90U[!, "CPIEnergy"])
        kanzoil_CPIE90L = vcat(0, kanzoil90L[!, "CPIEnergy"])


        kanzoil_CPIC= vcat(0, kanzoil[!, "CPICore"])  #pt_core
        kanzoil_CPIC90U = vcat(0, kanzoil90U[!, "CPICore"])
        kanzoil_CPIC90L = vcat(0, kanzoil90L[!, "CPICore"])

        kanzoil_SERV= vcat(0, kanzoil[!, "Services"]) #pt_services
        kanzoil_SERV90U = vcat(0, kanzoil90U[!, "Services"])
        kanzoil_SERV90L = vcat(0, kanzoil90L[!, "Services"])


        kanzoil_PCE= vcat(0, kanzoil[!, "PCE"])
        kanzoil_PCE90U = vcat(0, kanzoil90U[!, "PCE"])
        kanzoil_PCE90L = vcat(0, kanzoil90L[!, "PCE"])




        scale_factor = 10/IRFs["cumpt_oilr"][2] #Normalize impact effect on energy price = 1
        tgrid = -1:get_setting(m,:T)-2
        #labels = ["Our Model" "Kanzig IRFs"])
        #savefig(p1, "OnionModel/figures/IRFs/kanzig_oil_IRF_oil$(filestring_addl).pdf")

        p1 = plot(tgrid, vec(scale_factor*IRFs["cumpt_oilr"]), linewidth = 2, linestyle = :solid, linecolor = :red, titlefont=font(8), label="Our Model",
                  title = "Oil Price")
plot!(tgrid, vec(kanzoil_oilr), linewidth = 2, linestyle = :solid, linecolor = :blue,  label="Kanzig IRF")
plot!(tgrid, vec(kanzoil_oilr90U), linewidth = 1, linestyle = :dot, linecolor = :blue,  label=false)
plot!(tgrid, vec(kanzoil_oilr90L), linewidth = 1, linestyle = :dot, linecolor = :blue,  label=false)


p2 = plot(tgrid, [vec(scale_factor*IRFs["cumpt_energy"]), vec(kanzoil_CPIE)], linewidth = 2, titlefont=font(8), legend=false, linestyle = :solid, linecolor = [:red :blue],
          title = "Energy Price")

plot!(tgrid, [vec(kanzoil_CPIE90U), vec(kanzoil_CPIE90L)], linewidth = 1, legend=false, linestyle = :dot, linecolor = :blue)
        #labels = ["Our Model" "Kanzig IRFs"])
        #savefig(p2, "OnionModel/figures/IRFs/kanzig_oil_IRF_energy$(filestring_addl).pdf")


p3 = plot(tgrid, [vec(scale_factor*IRFs["cumpt_cpi"]), vec(kanzoil_CPI)], linewidth = 2, titlefont=font(8), legend=false, linestyle = [:solid :solid], linecolor = [:red :blue :blue :blue],
          title = "Consumer Price Level")
plot!(tgrid, [vec(kanzoil_CPI90U), vec(kanzoil_CPI90L)], linewidth = 1, legend=false, linestyle = :dot, linecolor = :blue)
        #labels = ["Our Model" "Kanzig IRFs"])
    #savefig(p3, "OnionModel/figures/IRFs/kanzig_oil_IRF_cpi$(filestring_addl).pdf")

p4 = plot(tgrid, [vec(scale_factor*IRFs["cumpt_core"]), vec(kanzoil_CPIC)], linewidth = 2, titlefont=font(8), legend=false, linestyle = [:solid :solid], linecolor = [:red :blue],
          title = "Core Price Level")
plot!(tgrid, [vec(kanzoil_CPIC90U), vec(kanzoil_CPIC90L)], linewidth = 1, titlefont=font(8), legend=false, linestyle = :dot, linecolor = :blue)
              #title = "Core Price Level")
              #labels = ["Our Model" "Kanzig IRFs"])
    #savefig(p4, "OnionModel/figures/IRFs/kanzig_oil_IRF_core$(filestring_addl).pdf")

p5 = plot(tgrid, [vec(kanzoil_SERV90U), vec(kanzoil_SERV90L)], linewidth = 1, titlefont=font(8), linestyle = :dot, linecolor = :blue, legend=false)

plot!(tgrid, [vec(scale_factor*IRFs["cumpt_services"]), vec(kanzoil_SERV)], linewidth = 2, titlefont=font(8), linestyle = [:solid :solid], linecolor = [:red :blue],
      title = "Services",
      labels = ["Our Model" "Kanzig IRFs"])
              #title = "Services",
    #savefig(p5, "OnionModel/figures/IRFs/kanzig_oil_IRF_serv$(filestring_addl).pdf")

p6 = plot(tgrid, [vec(scale_factor*IRFs["ct"]), vec(kanzoil_PCE)], linewidth = 2, titlefont=font(8), legend=false, linestyle = :solid, linecolor = [:red :blue],
          title = "PCE")
plot!(tgrid, [ vec(kanzoil_PCE90U), vec(kanzoil_PCE90L)], linewidth = 1, titlefont=font(8), legend=false, linestyle = :dot, linecolor = :blue,
      title = "PCE")
              #labels = ["Our Model" "Kanzig IRFs"])
    #savefig(p6, "OnionModel/figures/IRFs/kanzig_oil_IRF_pce$(filestring_addl).pdf")


p7 = plot(p1, p2, p3, p4, p5, p6, layout=6)
    savefig(p7, "OnionModel/figures/IRFs/kanzig_oil_IRFs_replication_all$(filestring_addl).pdf")


else
println("Need to specify what to plot")
end

end
