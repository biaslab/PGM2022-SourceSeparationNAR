using StatsFuns: softmax!

import Base: length, size, getindex, setindex!

mutable struct SoftmaxOutput{T <: Real} <: AbstractVector{T}
    value :: Vector{T}
end
Base.length(x::SoftmaxOutput) = length(x.value)
Base.size(x::SoftmaxOutput) = size(x.value)
Base.getindex(x::SoftmaxOutput, i::Int) = x.value[i]
Base.setindex!(x::SoftmaxOutput, y, i::Int) = (x.value[i] = y)

mutable struct SoftmaxGradientOutput{T <: Real} <: AbstractVector{T}
    value :: Vector{T}
end
Base.length(x::SoftmaxGradientOutput) = length(x.value)
Base.size(x::SoftmaxGradientOutput) = size(x.value)
Base.getindex(x::SoftmaxGradientOutput, i::Int) = x.value[i]
Base.setindex!(x::SoftmaxGradientOutput, y, i::Int) = (x.value[i] = y)

mutable struct SoftmaxLayer{T <: Real} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    input           :: Vector{T}
    output          :: SoftmaxOutput{T}
    gradient_input  :: Vector{T}
    gradient_output :: SoftmaxGradientOutput{T}
end
function SoftmaxLayer(dim)
    return SoftmaxLayer(dim, dim, zeros(dim), SoftmaxOutput(zeros(dim)), zeros(dim), SoftmaxGradientOutput(zeros(dim)))
end

function forward!(layer::SoftmaxLayer) 
    
    # fetch input and output in layer
    dim    = layer.dim_in
    input  = layer.input
    output = layer.output.value

    # update output of layer
    @inbounds for k = 1:dim
        output[k] = input[k]
    end
    # softmax!(output) # for numerical stability use custom vector type instead.

    # return output 
    return output
    
end

function propagate_error!(layer::SoftmaxLayer) 
    
    # fetch input and output gradients in layer
    dim             = layer.dim_in
    input           = layer.input
    output          = layer.output.value
    gradient_output = layer.gradient_output.value
    gradient_input  = layer.gradient_input

    # update input gradient of layer
    Z = logsumexp(output)
    @inbounds for k = 1:dim
        gradient_input[k] = (1 - exp(output[k] - Z)) * gradient_output[k] # compensate for the factor output[k] with loss function
    end

    # return gradient input 
    return gradient_input
    
end

update!(layer::SoftmaxLayer) = return

setlr!(layer::SoftmaxLayer, lr) = return