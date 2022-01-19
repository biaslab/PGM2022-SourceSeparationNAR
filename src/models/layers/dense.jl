using Octavian, LinearAlgebra, Random

mutable struct DenseLayer{T <: Real, O1 <: AbstractOptimizer, O2 <: AbstractOptimizer} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    W               :: Parameter{Matrix{T}, O1}
    b               :: Parameter{Vector{T}, O2}
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function DenseLayer(dim_in, dim_out; initializer::Tuple=(GlorotUniform(dim_in, dim_out), Zeros()), optimizer::Type{<:AbstractOptimizer}=Adam)
    return DenseLayer(dim_in, dim_out, Parameter(rand(initializer[1], (dim_out, dim_in)), optimizer), Parameter(rand(initializer[2], dim_out), optimizer), zeros(dim_in), zeros(dim_out), zeros(dim_in), zeros(dim_out))
end

function forward!(layer::DenseLayer)

    # fetch from layer
    dim_out = layer.dim_out
    W       = layer.W.value
    b       = layer.b.value
    input   = layer.input

    # calculate output of layer
    output = layer.output
    matmul!(output, W, input) # for improved speed, save Wt = W' and call matmul!(output, Wt', input)
    @inbounds for k = 1:dim_out
        output[k] += b[k]
    end  

    # return output 
    return output
    
end

function propagate_error!(layer::DenseLayer)

    # fetch from layer
    dim_in  = layer.dim_in
    dim_out = layer.dim_out
    W       = layer.W
    b       = layer.b

    ∂L_∂x   = layer.gradient_input
    ∂L_∂y   = layer.gradient_output
    ∂L_∂W   = W.gradient
    ∂L_∂b   = b.gradient
    input   = layer.input

    # set gradients for b
    @inbounds for k in 1:dim_out
        ∂L_∂b[k] += ∂L_∂y[k]
    end

    # set gradient for W
    @inbounds for k2 in 1:dim_in
        inputk2 = input[k2]
        @inbounds for k1 in 1:dim_out
            ∂L_∂W[k1,k2] += ∂L_∂y[k1] * inputk2
        end
    end

    # set gradient at input
    matmul!(∂L_∂x, W.value', ∂L_∂y)

    # return gradient at input of layer
    return ∂L_∂x

end

function update!(layer::DenseLayer)

    # update parameters in layer
    update!(layer.W)
    update!(layer.b)
    
end

function setlr!(layer::DenseLayer, lr)

    # update parameters in layer
    setlr!(layer.W, lr)
    setlr!(layer.b, lr)
    
end

function setbatchsize!(layer::DenseLayer, batch_size::Int64)

    # update parameters in layer
    setbatchsize!(layer.W, batch_size)
    setbatchsize!(layer.b, batch_size)
    
end

isinvertible(layer::DenseLayer) = false