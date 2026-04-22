using CSV
using DataFrames


function load_data_bbq(m)
df = CSV.read("../models/mBBQ/data/data.csv", DataFrame; header=false)
df = df[1:end-1, 2:end]  # remove first column
df = DataFrame(Matrix(df)', :auto)  

m.grids[:obs] = observable_names = [
    :ygpc,          # Real GDP Growth per capita
    :cgpc,          # Real Consumption Growth per capita
    :igpc,          # Real Investment Growth
    :pi,             # Inflation
    :rcb,           # Nominal Interest Rate
    :wg,            # Real Wage Growth
    :u,             # Unemployment Rate
    :lt,            # Lumpsum Transfer
    :profit,        # Profit
    :xcb,           # Central Bank Assets
    # :sp500,         # Stock Returns
]

rename!(df, m.grids[:obs])

return df

end