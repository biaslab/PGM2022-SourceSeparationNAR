using StatsFuns: logsumexp

struct CCE <: AbstractLoss end

function loss(::CCE, true_value::T1, predicted_value::T2) where { T1 <: AbstractVector, T2 <: AbstractVector }
    len = length(true_value)
    @assert len == length(predicted_value)

    cce = 0.0
    for k = 1:len
        cce -= true_value[k] * log(predicted_value[k])
    end
    cce /= len
    return cce
end

function loss(::CCE, true_value::T, predicted_value::SoftmaxOutput) where { T <: AbstractVector }
    len = length(true_value)
    @assert len == length(predicted_value)

    normalization = logsumexp(predicted_value.value)

    cce = 0.0
    for k = 1:len
        cce -= true_value[k] * (predicted_value[k] - normalization)
    end
    cce /= len
    return cce
end


function dloss(::CCE, true_value::T1, predicted_value::T2) where { T1 <: AbstractVector, T2 <: AbstractVector }
    len = length(true_value)
    @assert len == length(predicted_value)

    dloss = Vector{T}(undef, len)
    return dloss!(Type{CCE}, dloss, true_value, predicted_value)
end
function dloss(::CCE, true_value::T, predicted_value::SoftmaxOutput) where { T <: AbstractVector }
    len = length(true_value)
    @assert len == length(predicted_value)

    dloss = SoftmaxGradientOutput(Vector{T}(undef, len))
    return dloss!(Type{CCE}, dloss, true_value, predicted_value)
end

function dloss!(::CCE, dloss::T, true_value::T1, predicted_value::T2) where { T, T1 <: AbstractVector, T2 <: AbstractVector }
    for k = 1:length(true_value)
        dloss[k] = -true_value[k] / predicted_value[k]
    end
    return dloss
end

function dloss!(::CCE, dloss::T, true_value::T1, predicted_value::SoftmaxOutput) where { T, T1 <: AbstractVector }
    for k = 1:length(true_value)
        dloss[k] = -true_value[k] 
    end
    return dloss
end