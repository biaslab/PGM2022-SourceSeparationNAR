export CCE, MSE, MAE

abstract type AbstractLoss end

include("cce.jl")
include("mae.jl")
include("mse.jl")