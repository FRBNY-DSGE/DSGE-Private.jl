using CSV
using DataFrames
using Dates



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


# Generate quarter-end dates starting from 1992-01-31
nrows = nrow(df)
start_date = Date(1992, 3, 31)
dates = [start_date + Month(3 * (i - 1)) for i in 1:nrows]

insertcols!(df, 1, :date => dates)


return df

end