using LinearAlgebra, Random

mutable struct DenseLayer{T <: Real, O1 <: AbstractOptimizer, O2 <: AbstractOptimizer} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    W               :: Parameter{Matrix{T}, O1}
    b               :: Parameter{Vector{T}, O2}
    input           :: Matrix{T}
    output          :: Matrix{T}
    gradient_input  :: Matrix{T}
    gradient_output :: Matrix{T}
end
function DenseLayer(dim_in, dim_out; batch_size::Int64=128, initializer::Tuple=(GlorotUniform(dim_in, dim_out), Zeros()), optimizer::Type{<:AbstractOptimizer}=Adam)
    return DenseLayer(dim_in, dim_out, Parameter(rand(initializer[1], (dim_out, dim_in)), optimizer), Parameter(rand(initializer[2], dim_out), optimizer), zeros(dim_in, batch_size), zeros(dim_out, batch_size), zeros(dim_in, batch_size), zeros(dim_out, batch_size))
end

function forward!(layer::DenseLayer)

    # fetch from layer
    W       = layer.W.value
    b       = layer.b.value
    input   = layer.input

    # calculate output of layer
    output = getmatoutput(layer)
    custom_mulp!(output, W, input, b)

    # return output 
    return output
    
end

function propagate_error!(layer::DenseLayer)

    # fetch from layer
    dim_in  = layer.dim_in
    dim_out = layer.dim_out
    W       = layer.W
    b       = layer.b

    ∂L_∂x   = getmatgradientinput(layer)
    ∂L_∂y   = getmatgradientoutput(layer)
    ∂L_∂W   = W.gradient
    ∂L_∂b   = b.gradient
    input   = layer.input
    batch_size = size(input, 2)
    ibatch_size = 1 / batch_size

    # set gradients for b
    @turbo for k1 in 1:dim_out
        ∂L_∂b[k1] = 0
        for k2 in 1:batch_size
            ∂L_∂b[k1] += ∂L_∂y[k1,k2]
        end
        ∂L_∂b[k1] *= ibatch_size
    end

    # set gradient for W #todo
    custom_mul!(∂L_∂W, ∂L_∂y, input')
    @turbo for k1 in 1:dim_out
        for k2 in 1:dim_in
            ∂L_∂W[k1,k2] *= ibatch_size
        end
    end

    # set gradient at input
    custom_mul!(∂L_∂x, W.value', ∂L_∂y)

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

isinvertible(layer::DenseLayer) = false

nr_params(layer::DenseLayer) = length(layer.W) + length(layer.b)