export plot_logs

using CSV, DataFrames, PyPlot

function plot_logs(; path="logs/", logscale=false, loss=["mse", "mae"])

    # create figure
    plt.figure()

    # loop through folders in log folder
    for folder in filter(x -> isdir(string(path, "/", x)), readdir(path))

        # loop through files in folder
        for file in filter(x->occursin(".csv",x), readdir(string(path, folder)))

            # read data from file
            df = DataFrame(CSV.File(string(path, folder, "/", file)))

            # fetch columns 
            cols = names(df)

            # loop through columns
            for col in filter((x) -> x in loss, cols)

                if logscale
                    plt.plot(df[!,"epoch"], log.(df[!, col]), label=string(folder, "-", col, "-", file[1:end-4]))
                else
                    plt.plot(df[!,"epoch"], df[!, col], label=string(folder, "-", col, "-", file[1:end-4]))
                end

            end

        end

    end

    plt.grid()
    plt.legend()
    plt.xlabel("epoch");

end