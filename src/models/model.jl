export Model
export ARLayer, DenseSNLayer, DenseLayer, iARLayer, LeakyReluLayer, MvAdditiveLayer, PermutationLayer, ReluLayer, ResidualLayer, SoftmaxLayer

export forward, backward, jacobian, invjacobian
export forward!, propagate_error!, update!
export setlr!, setbatchsize!
export isinvertible, nr_params, print_info
export deploy

using LinearAlgebra: I

abstract type AbstractModel end
abstract type AbstractLayer end

# Simplified model call
(model::AbstractModel)(x) = forward(model, x)
(layer::AbstractLayer)(x) = forward(layer, x)


# include parameters and memory
include("parameter.jl")
include("memory.jl")

# include layers
include("layers/ar_layer.jl")
include("layers/dense_spectral_norm.jl")
include("layers/dense.jl")
include("layers/leakyrelu.jl")
include("layers/iar_layer.jl")
include("layers/mv_additive_layer.jl")
include("layers/permutation_layer.jl")
include("layers/relu.jl")
include("layers/residual_layer.jl")
include("layers/softmax.jl")

# include helpers
include("helpers.jl")

# Model constructors
mutable struct Model{L <: Tuple, M <: Union{Nothing,AbstractMemory}} <: AbstractModel
    dim_in          :: Int64
    dim_out         :: Int64
    layers          :: L
    memory          :: M
end
function Model(dim, layers; batch_size::Int64=128)
    return Model(dim, dim, layers; batch_size=batch_size)
end
function Model(dim_in, dim_out, layers; batch_size::Int64=128)
    if typeof(last(layers)) <: SoftmaxLayer
        return Model(   dim_in, 
                        dim_out, 
                        layers,
                        TrainMemory(
                            zeros(dim_in,batch_size), 
                            SoftmaxOutput(zeros(dim_out, batch_size)), 
                            zeros(dim_in, batch_size), 
                            SoftmaxGradientOutput(zeros(dim_out, batch_size))
                        ) 
                    )
    else
        return Model(dim_in, dim_out, layers, TrainMemory(dim_in, dim_out, batch_size))
    end
end

# forward function without efficient allocations
function forward(model::Model, input::T) where { T <: AbstractArray }

    # fetch layers
    layers = model.layers

    # set temporary output
    output = copy(input)

    # propagate through layers
    @inbounds for layer in layers
        
        # run current layer forward
        output = forward(layer, output)::T # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
    
    end

    # return output
    return output

end

# forward function with memory
function forward!(model::Model{<:Tuple,<:AbstractMemory}, input)

    # set input in model
    copytoinput!(model, input)

    # run model forward
    output = forward!(model)

    # return output
    return output

end

# internal forward function for model with memory
function forward!(model::Model{<:Tuple,<:TrainMemory}) 

    # fetch layers
    layers = model.layers

    # set current input
    current_input = getmatinput(model)

    # propagate through layers
    @inbounds for layer in layers

        # set input in layer
        linktoinput!(layer, current_input)
        
        # run current layer forward
        current_input = forward!(layer)::Matrix{Float64} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
    
    end

    # update output of model
    copytooutput!(model, current_input)

    # return output
    return getoutput(model)
    
end

# internal forward function for model with memory
function forward!(model::Model{<:Tuple,<:DeployMemory}) 

    # fetch layers
    layers = model.layers

    # set current input
    current_input = getmatinput(model)

    # propagate through layers
    @inbounds for layer in layers

        # set input in layer
        linktoinput!(layer, current_input)
        
        # run current layer forward
        current_input = forward!(layer)::Vector{Float64} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
    
    end

    # update output of model
    copytooutput!(model, current_input)

    # return output
    return getoutput(model)
    
end

# backward function
function backward(model::Model, output::T) where { T <: AbstractArray }

    # fetch layers
    layers = model.layers

    # set temporary input
    input = copy(output)

    # propagate through layers
    @inbounds for layer in reverse(layers)
        
        # run current layer backward
        input = backward(layer, input)::T # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
    
    end

    # return output
    return input

end

# jacobian function
function jacobian(model::Model, input::Vector{T}) where { T <: Real }

    # fetch layers
    layers = model.layers

    # set temporary jacobian
    current_J = I

    # set current input
    current_input = copy(input)

    # propagate through layers
    @inbounds for layer in layers
        
        # run current layer forward
        J_new = jacobian(layer, current_input)#::AbstractMatrix{T} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
    
        # update current jacobian
        current_J = custom_mul(J_new, current_J)

        # run model forward
        current_input = forward(layer, current_input)

    end

    # return jacobian
    return current_J

end

# inverse jacobian function
function invjacobian(model::Model, output::Vector{T}) where { T <: Real }

    # fetch layers
    layers = model.layers

    # set temporary inverse jacobian
    current_invJ = I

    # set current output
    current_output = copy(output)

    # propagate through layers
    @inbounds for layer in layers
        
        # calculate inverse jacobian
        invJ_new = invjacobian(layer, current_output)::Matrix{T} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
    
        # update current jacobian
        current_invJ = custom_mul(invJ_new, current_invJ)

        # run model backward
        current_output = backward(layer, current_output)

    end

    # return inverse jacobian
    return current_invJ

end

# backpropagation requires memory
function propagate_error!(model::Model{<:Tuple,<:TrainMemory}, gradient_output::Matrix)

    # set gradient output in model
    copytogradientoutput!(model, gradient_output)

    # propagate error backwards
    gradient_input = propagate_error!(model)

    # return gradient input
    return gradient_input
    
end

function propagate_error!(model::Model{<:Tuple,<:TrainMemory})

    # fetch layers
    layers = model.layers

    # set current gradient output
    current_gradient_output = getmatgradientoutput(model)

    # propagate through layers
    @inbounds for layer in reverse(layers)

        # set gradient output in layer
        linktogradientoutput!(layer, current_gradient_output)

        # run current layer forward
        current_gradient_output = propagate_error!(layer)::Matrix{Float64} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
    
    end

    # update gradient input of model
    copytogradientinput!(model, current_gradient_output)

    # return gradient input of model
    return getmatgradientinput(model)

end

# update requires memory
function update!(model::Model{<:Tuple,<:TrainMemory})

    # fetch layers
    layers = model.layers

    # update parameters in layers
    @inbounds for layer in layers
        update!(layer)
    end

end

function setlr!(model::Model{<:Tuple,<:TrainMemory}, lr)

    # fetch layers
    layers = model.layers

    # update parameters in layers
    @inbounds for layer in layers
        setlr!(layer, lr)
    end

end

isinvertible(model::Model) = mapreduce(isinvertible, *, model.layers)

nr_params(model::Model) = mapreduce(nr_params, +, model.layers)

function deploy(model::Model)
    return Model(
        model.dim_in,
        model.dim_out,
        tuple([deploy(layer, model.dim_in) for layer in model.layers]...),
        DeployMemory(
            model.dim_in,
            model.dim_out,
            model.dim_in
        )
    )
end

function print_info(model::Model, file::String)

    # open file
    open(file, "w") do io
        print_info(model, 0, io)
    end

end

function print_info(model::Model, level::Int, io)

    # print model info
    if level == 1
        write(io, string("Model(", model.dim_in, ", ", model.dim_out, ")\n"))
    else
        write(io, string(["--" for _=1:level]..., " Model(", model.dim_in, ", ", model.dim_out, ")\n"))
    end

    # print layers
    @inbounds for layer in model.layers
        print_info(layer, level+1, io)
    end

end