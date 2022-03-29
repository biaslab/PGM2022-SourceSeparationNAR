using LinearAlgebra, Random

mutable struct DenseSNLayer{M <: Union{Nothing, AbstractMemory}, W <: Union{Matrix, SVDSpectralNormal}, B <: Union{Vector, Parameter}} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    W               :: W
    b               :: B
    memory          :: M
end
function DenseSNLayer(dim_in, dim_out, C; batch_size::Int64=128, initializer::Tuple=(GlorotUniform(dim_in, dim_out), Zeros()), optimizer::Type{<:AbstractOptimizer}=GradientDescent)
    return  DenseSNLayer(
                dim_in, 
                dim_out, 
                SVDSpectralNormal(Parameter(rand(initializer[1], (dim_out, dim_in)), optimizer), C), 
                Parameter(rand(initializer[2], dim_out), optimizer), 
                TrainMemory(dim_in, dim_out, batch_size)
            )
end

function forward(layer::DenseSNLayer, input)

    # calculate and return output of layer
    return custom_mulp(getmat(layer.W), input, getvalue(layer.b))
    
end

function forward!(layer::DenseSNLayer{<:AbstractMemory,W,B}, input) where { W, B }
    
    # set input
    copytoinput!(layer, input)

    # call forward function and return output
    return forward!(layer)
    
end

function forward!(layer::DenseSNLayer{<:AbstractMemory,W,B}) where { W, B }

    # calculate output of layer
    return custom_mulp!(getmatoutput(layer), getmat(layer.W), getmatinput(layer), getvalue(layer.b))
    
end

function forward_jacobian!(layer::DenseSNLayer{<:DeployMemory,WT,BT}) where { WT, BT }
    return forward!(layer), custom_mul!(getmatjacobianoutput(layer), getmat(layer.W), getmatjacobianinput(layer))
end

function jacobian(layer::DenseSNLayer, ::Vector{<:Real})
    return getmat(layer.W)
end

function propagate_error!(layer::DenseSNLayer{<:TrainMemory,W,B}) where { W, B }

    # fetch from layer
    dim_in  = layer.dim_in
    dim_out = layer.dim_out
    b       = layer.b

    ∂L_∂x   = getmatgradientinput(layer)
    ∂L_∂y   = getmatgradientoutput(layer)
    ∂L_∂W   = layer.W.A.gradient
    ∂L_∂b   = b.gradient
    input   = getmatinput(layer)
    batch_size = size(input, 2)
    ibatch_size = 1 / batch_size

    # set gradients for b
    @turbo for k1 in 1:dim_out
        tmp = zero(T)
        for k2 in 1:batch_size
            tmp += ∂L_∂y[k1,k2]
        end
        ∂L_∂b[k1] = tmp * ibatch_size
    end

    # normalize W
    Wsn = normalize!(layer.W)
    σ = layer.W.σ
    u1 = layer.W.u
    v1 = layer.W.v

    # set gradient for W
    ∂L_∂Wsn = similar(∂L_∂W)
    custom_mul!(∂L_∂Wsn, ∂L_∂y, input') # E[δ * h^⊤]
    # λ = mean(∂L_∂y' * Wsn * input) # E[δ^⊤ * (Wsn * h)]
    # println(λ)
    # λ = mean(diag(∂L_∂y' * Wsn * input))
    λ = meandot(∂L_∂y, Wsn, input)

    ibatch_sizeσ = ibatch_size / σ # todo: should C also be here?
    @turbo for k1 in 1:dim_out
        for k2 in 1:dim_in
            ∂L_∂W[k1,k2] = (∂L_∂Wsn[k1,k2] - λ * u1[k1] * v1[k2]) * ibatch_sizeσ
        end
    end

    # set gradient at input
    custom_mul!(∂L_∂x, Wsn', ∂L_∂y)

    # return gradient at input of layer
    return ∂L_∂x

end

function update!(layer::DenseSNLayer{<:TrainMemory,W,B}) where { W, B }

    # update parameters in layer
    update!(layer.W)
    update!(layer.b)
    
end

function setlr!(layer::DenseSNLayer{<:TrainMemory,W,B}, lr) where { W, B }

    # update parameters in layer
    setlr!(layer.W, lr)
    setlr!(layer.b, lr)
    
end

isinvertible(layer::DenseSNLayer) = false

nr_params(layer::DenseSNLayer) = length(layer.W) + length(layer.b)

function deploy(layer::DenseSNLayer; jacobian_start=IdentityMatrix())

    jacobian_layer = jacobian(layer, randn(layer.dim_in))
    jacobian_layer_output = jacobian_layer * jacobian_start


    return DenseSNLayer(
        layer.dim_in,
        layer.dim_out,
        normalize!(layer.W),
        getvalue(layer.b),
        DeployMemory(
            randn(layer.dim_in),
            randn(layer.dim_out),
            jacobian_layer,
            jacobian_start,
            jacobian_layer_output
        )
    )
end

function print_info(layer::DenseSNLayer, level::Int, io)

    # print layer
    write(io, string(["--" for _=1:level]..., " Dense(", layer.dim_in, ", ", layer.dim_out, ")\n"))

end