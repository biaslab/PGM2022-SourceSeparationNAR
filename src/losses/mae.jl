struct MAE{T} <: AbstractLoss
    loss :: Vector{T}
end
function MAE(; batch_size::Int=128)
    MAE(zeros(batch_size))
end

getid(::MAE) = "mae"

function calculate_loss!(mae::MAE, true_output::T1, predicted_output::T2) where { T1 <: AbstractMatrix, T2 <: AbstractMatrix }
    (ax1, ax2) = axes(true_output)
    @assert (ax1, ax2) == axes(predicted_output)

    loss = mae.loss
    idim = 1/length(ax1)

    @turbo for k2 in ax2
        loss[k2] = 0
        for k1 in ax1
            loss[k2] += abs(predicted_output[k1,k2] - true_output[k1,k2]) * idim
        end
    end
    return loss
end

function calculate_dloss!(::MAE, dloss::T1, true_output::T2, predicted_output::T3) where { T1 <: AbstractMatrix, T2 <: AbstractMatrix, T3 <: AbstractMatrix }
    (ax1, ax2) = axes(true_output)
    @assert (ax1, ax2) == axes(predicted_output)

    @turbo for k1 in ax1
        for k2 in ax2
            dloss[k1,k2] = sign(predicted_output[k1,k2] - true_output[k1,k2])
        end
    end
    return dloss
end