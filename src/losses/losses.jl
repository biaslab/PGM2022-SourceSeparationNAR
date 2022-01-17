export MSE, MAE

abstract type AbstractLoss end

include("mae.jl")
include("mse.jl")