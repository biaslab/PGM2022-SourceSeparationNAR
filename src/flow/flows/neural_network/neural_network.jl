export NeuralNetwork
export ReluLayer
export DenseLayer

mutable struct NeuralNetwork{L <: Tuple, T <: Real} <: AbstractFlow
    dim             :: Int
    layers          :: L
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function NeuralNetwork(dim, layers)
    return NeuralNetwork(dim, layers, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end

function forward!(model::NeuralNetwork{L,T}) where { L, T <: Real }

    # set input in model and set output to running input
    input = model.input
    output = model.output
    dim = model.dim
    @inbounds for k in 1:dim
        output[k] = input[k]
    end

    # propagate through layers
    layers = model.layers
    @inbounds for k in 1:length(layers)
        layerk = layers[k]
        outputi = forward!(layerk, output)
        @inbounds for ki in 1:dim
            output[ki] = outputi[ki]
        end
    end

    # return output 
    return output
    
end

function propagate_error!(model::NeuralNetwork{L,T}) where { L, T <: Real }

    # set gradient at output of layer and input as running input
    gradient_output = model.gradient_output
    gradient_input  = model.gradient_input
    dim             = model.dim
    @inbounds for k in 1:dim
        gradient_input[k] = gradient_output[k]
    end

    # propagate gradient through layers
    layers = model.layers
    @inbounds for k in 1:length(layers)
        layerk = layers[k]
        gradient_inputi = propagate_error!(layerk, gradient_input)
        @inbounds for ki in 1:dim
            gradient_input[ki] = gradient_inputi[ki]
        end
    end

    # return gradient at input of layer
    return gradient_input

end

function update!(model::NeuralNetwork)
    layers = model.layers
    @inbounds for k in 1:length(layers)
        update!(layers[k])
    end
end


include("activations/relu.jl")

include("layers/dense.jl")