struct CCE <: AbstractLoss end

function loss(::CCE, true_value::T, predicted_value::T) where { T <: AbstractVector }
    len = length(true_value)
    @assert len == length(predicted_value)

    cce = 0.0
    for k = 1:len
        cce -= true_value[k] * log(predicted_value[k])
    end
    cce /= len
    return cce
end


function dloss(::CCE, true_value::T, predicted_value::T) where { T <: AbstractVector }
    len = length(true_value)
    @assert len == length(predicted_value)

    dloss = Vector{T}(undef, len)
    return dloss!(Type{CCE}, dloss, true_value, predicted_value)
end

function dloss!(::CCE, dloss::T, true_value::T, predicted_value::T) where { T <: AbstractVector }
    for k = 1:length(true_value)
        dloss[k] = -true_value[k] / predicted_value[k]
    end
    return dloss
end