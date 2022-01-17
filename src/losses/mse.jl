struct MSE <: AbstractLoss end

function loss(::MSE, true_value::T, predicted_value::T) where { T <: Real }
    return abs2(true_value, predicted_value)
end

function loss(::MSE, true_value::T, predicted_value::T) where { T <: AbstractVector }
    len = length(true_value)
    @assert len == length(predicted_value)

    mse = 0.0
    for k = 1:len
        mse += abs2(true_value[k] - predicted_value[k])
    end
    mse /= len
    return mse
end


function dloss(::MSE, true_value::T, predicted_value::T) where { T <: Real }
    return 2*(predicted_value - true_value)
end

function dloss(::MSE, true_value::T, predicted_value::T) where { T <: AbstractVector }
    len = length(true_value)
    @assert len == length(predicted_value)

    dloss = Vector{T}(undef, len)
    return dloss!(Type{MSE}, dloss, true_value, predicted_value)
end

function dloss!(::MSE, dloss::T, true_value::T, predicted_value::T) where { T <: AbstractVector }
    for k = 1:length(true_value)
        dloss[k] = 2*(predicted_value[k] - true_value[k])
    end
    return dloss
end