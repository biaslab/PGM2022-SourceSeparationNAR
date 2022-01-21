using StatsFuns: logsumexp

struct CCE{T} <: AbstractLoss
    loss :: Vector{T}
end
function CCE(; batch_size::Int=128)
    return CCE(zeros(batch_size))
end

function calculate_loss!(cce::CCE, true_output::T1, predicted_output::T2) where { T1 <: AbstractMatrix, T2 <: AbstractMatrix }
    (ax1, ax2) = axes(true_output)
    @assert (ax1, ax2) == axes(predicted_output)

    loss = cce.loss
    dim = length(ax1)

    @turbo for k2 in ax2
        loss[k2] = 0
        for k1 in ax1
            loss[k2] -= true_output[k1,k2] * log(predicted_output[k1,k2])
        end
        loss[k2] /= dim
    end
    return loss
end

function calculate_loss!(cce::CCE, true_output::T, predicted_output::SoftmaxOutput) where { T <: AbstractMatrix }
    predicted_output_mat = getmat(predicted_output)
    (ax1, ax2) = axes(true_output)
    @assert (ax1, ax2) == axes(predicted_output_mat)

    loss = cce.loss
    idim = 1/length(ax1)

    @inbounds for k2 in ax2
        loss[k2] = 0
        Z = logsumexp_column(predicted_output_mat, k2)
        @inbounds for k1 in ax1
            loss[k2] -= true_output[k1,k2] * (predicted_output_mat[k1,k2] - Z)
        end
        loss[k2] *= idim
    end
    return loss
end

function calculate_dloss!(::CCE, dloss::T, true_output::T, predicted_output::T) where { T <: AbstractMatrix }
    (ax1, ax2) = axes(true_output)
    @assert (ax1, ax2) == axes(predicted_output)

    @turbo for k1 in ax1
        for k2 in ax2
            dloss[k1,k2] = -true_output[k1,k2] / predicted_output[k1,k2]
        end
    end
    return dloss
end

function calculate_dloss!(::CCE, dloss::T1, true_output::T2, predicted_output::SoftmaxOutput) where { T1 <: AbstractMatrix, T2 <: AbstractMatrix }
    predicted_output_mat = getmat(predicted_output)
    (ax1, ax2) = axes(true_output)
    @assert (ax1, ax2) == axes(predicted_output_mat)

    @inbounds for k1 in ax1
        for k2 in ax2
            dloss[k1,k2] = -true_output[k1,k2]
        end
    end
    return dloss
end