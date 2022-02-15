using LinearAlgebra, Random

mutable struct DenseSNLayer{M <: Union{Nothing, Memory}, T <: Real, O1 <: AbstractOptimizer, O2 <: AbstractOptimizer} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    W               :: Parameter{Matrix{T}, O1}
    Wsn             :: Matrix{T}
    b               :: Parameter{Vector{T}, O2}
    C               :: Float64
    memory          :: M
end
function DenseSNLayer(dim_in, dim_out, C; batch_size::Int64=128, initializer::Tuple=(GlorotUniform(dim_in, dim_out), Zeros()), optimizer::Type{<:AbstractOptimizer}=Adam)
    return DenseSNLayer(dim_in, dim_out, Parameter(rand(initializer[1], (dim_out, dim_in)), optimizer), randn(dim_out,dim_in), Parameter(rand(initializer[2], dim_out), optimizer), C, Memory(dim_in, dim_out, batch_size))
end

function forward(layer::DenseSNLayer, input)

    # fetch from layer
    W       = layer.W.value
    Wsn     = layer.Wsn
    b       = layer.b.value

    # normalize W
    Wsn .= W 
    Wsn ./= (opnorm(W) / layer.C)

    # calculate output of layer
    # output = custom_mulp(W, input, b)
    output = Wsn*input + b
    
    # return output 
    return output
    
end

function jacobian(layer::DenseSNLayer, ::Vector{<:Real})
    return layer.Wsn
end

function forward!(layer::DenseSNLayer{<:Memory,T,O1,O2}) where { T, O1, O2 }

    # fetch from layer
    W       = layer.W.value
    Wsn     = layer.Wsn
    b       = layer.b.value
    input   = getmatinput(layer)

    # normalize W
    Wsn .= W
    Wsn ./= opnorm(W)

    # calculate output of layer
    output = getmatoutput(layer)
    custom_mulp!(output, Wsn, input, b)

    # return output 
    return output
    
end

function propagate_error!(layer::DenseSNLayer{<:Memory,T,O1,O2}) where { T, O1, O2 }

    # fetch from layer
    dim_in  = layer.dim_in
    dim_out = layer.dim_out
    W       = layer.W
    b       = layer.b

    ∂L_∂x   = getmatgradientinput(layer)
    ∂L_∂y   = getmatgradientoutput(layer)
    ∂L_∂W   = W.gradient
    ∂L_∂b   = b.gradient
    input   = getmatinput(layer)
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

    # set gradient for W
    ∂L_∂Wsn = similar(∂L_∂W)
    custom_mul!(∂L_∂Wsn, ∂L_∂y, input')
    σ = opnorm(W.value)

    @turbo for k1 in 1:dim_out
        for k2 in 1:dim_in
            ∂L_∂W[k1,k2] = ∂L_∂Wsn[k1,k2] * ibatch_size / σ
        end
    end



    # set gradient at input
    custom_mul!(∂L_∂x, W.value', ∂L_∂y)

    # return gradient at input of layer
    return ∂L_∂x

end

function update!(layer::DenseSNLayer{<:Memory,T,O1,O2}) where { T, O1, O2 }

    # update parameters in layer
    update!(layer.W)
    update!(layer.b)
    
end

function setlr!(layer::DenseSNLayer{<:Memory,T,O1,O2}, lr) where { T, O1, O2 }

    # update parameters in layer
    setlr!(layer.W, lr)
    setlr!(layer.b, lr)
    
end

isinvertible(layer::DenseSNLayer) = false

nr_params(layer::DenseSNLayer) = length(layer.W) + length(layer.b)

function print_info(layer::DenseSNLayer, level::Int, io)

    # print layer
    write(io, string(["--" for _=1:level]..., " Dense(", layer.dim_in, ", ", layer.dim_out, ")\n"))

end