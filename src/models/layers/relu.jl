mutable struct ReluLayer{T <: Real} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function ReluLayer(dim)
    return ReluLayer(dim, dim, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end

function forward!(layer::ReluLayer) 
    
    # fetch input and output in layer
    dim    = layer.dim_in
    input  = layer.input
    output = layer.output

    # update output of layer
    @inbounds for k = 1:dim
        output[k] = relu(input[k])
    end

    # return output 
    return output
    
end

function propagate_error!(layer::ReluLayer) 
    
    # fetch input and output gradients in layer
    dim             = layer.dim_in
    input           = layer.input
    gradient_output = layer.gradient_output
    gradient_input  = layer.gradient_input

    # update input gradient of layer
    @inbounds for k = 1:dim
        gradient_input[k] = drelu(input[k]) * gradient_output[k]
    end

    # return gradient input 
    return gradient_input
    
end

update!(layer::ReluLayer) = return

setlr!(layer::ReluLayer, lr) = return

setbatchsize!(layer::ReluLayer, batch_size) = return

isinvertible(layer::ReluLayer) = false

nr_params(layer::ReluLayer) = 0

relu(x) = max(0.0, x)
drelu(x) = x > 0 ? 1.0 : 0.0