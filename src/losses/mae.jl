struct MAE <: AbstractLoss end

function loss(::MAE, true_value::T, predicted_value::T) where { T <: Real }
    return abs(true_value, predicted_value)
end

function loss(::MAE, true_value::T, predicted_value::T) where { T <: AbstractVector }
    len = length(true_value)
    @assert len == length(predicted_value)

    mse = 0.0
    for k = 1:len
        mse += abs(true_value[k] - predicted_value[k])
    end
    mse /= len
    return mse
end


function dloss(::MAE, true_value::T, predicted_value::T) where { T <: Real }
    return sign(predicted_value - true_value)
end

function dloss(::MAE, true_value::T, predicted_value::T) where { T <: AbstractVector }
    len = length(true_value)
    @assert len == length(predicted_value)

    dloss = Vector{T}(undef, len)
    return dloss!(Type{MAE}, dloss, true_value, predicted_value)
end

function dloss!(::MAE, dloss::T, true_value::T, predicted_value::T) where { T <: AbstractVector }
    for k = 1:length(true_value)
        dloss[k] = sign(predicted_value[k] - true_value[k])
    end
    return dloss
end