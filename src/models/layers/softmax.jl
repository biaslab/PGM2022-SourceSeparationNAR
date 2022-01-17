using StatsFuns: softmax!

mutable struct SoftmaxLayer{T <: Real} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function SoftmaxLayer(dim)
    return SoftmaxLayer(dim, dim, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end

function forward!(layer::SoftmaxLayer) 
    
    # fetch input and output in layer
    dim    = layer.dim_in
    input  = layer.input
    output = layer.output

    # update output of layer
    @inbounds for k = 1:dim
        output[k] = input[k]
    end
    softmax!(output)

    # return output 
    return output
    
end

function propagate_error!(layer::SoftmaxLayer) 
    
    # fetch input and output gradients in layer
    dim             = layer.dim_in
    input           = layer.input
    output          = layer.output
    gradient_output = layer.gradient_output
    gradient_input  = layer.gradient_input

    # update input gradient of layer
    @inbounds for k = 1:dim
        gradient_input[k] = output[k] * (1 - output[k]) * gradient_output[k]
    end

    # return gradient input 
    return gradient_input
    
end

update!(layer::SoftmaxLayer) = return