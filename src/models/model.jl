export Model
export DenseLayer, MvAdditiveLayer, PermutationLayer, ReluLayer, SoftmaxLayer, UvAdditiveLayer

export forward!, propagate_error!, update!
export setlr!, setbatchsize!

abstract type AbstractModel end
abstract type AbstractLayer end


# include parameters
include("parameter.jl")

# include layers
include("layers/dense.jl")
include("layers/mv_additive_layer.jl")
include("layers/permutation_layer.jl")
include("layers/relu.jl")
include("layers/softmax.jl")
include("layers/uv_additive_layer.jl")

# include helpers
include("helpers.jl")

mutable struct Model{L <: Tuple, T <: Real, V1 <: AbstractVector, V2 <: AbstractVector } <: AbstractModel
    dim_in          :: Int64
    dim_out         :: Int64
    layers          :: L
    input           :: Vector{T}
    output          :: V1
    gradient_input  :: Vector{T}
    gradient_output :: V2
end
function Model(dim, layers)
    if typeof(last(layers)) <: SoftmaxLayer
        return Model(dim, dim, layers, zeros(dim), SoftmaxOutput(zeros(dim)), zeros(dim), SoftmaxGradientOutput(zeros(dim)))
    else
        return Model(dim, dim, layers, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
    end
end
function Model(dim_in, dim_out, layers)
    if typeof(last(layers)) <: SoftmaxLayer
        return Model(dim_in, dim_out, layers, zeros(dim_in), SoftmaxOutput(zeros(dim_out)), zeros(dim_in), SoftmaxGradientOutput(zeros(dim_out)))
    else
        return Model(dim_in, dim_out, layers, zeros(dim_in), zeros(dim_out), zeros(dim_in), zeros(dim_out))
    end
end

function forward!(model::Model{L,T,V1,V2}, input::Vector{T}) where { L, T <: Real, V1, V2}

    # set input in model
    setinput!(model, input)

    # run model forward
    output = forward!(model)

    # return output
    return output

end

function forward!(model::Model{L,T,V1,V2}) where { L, T <: Real, V1, V2 }

    # fetch layers
    layers = model.layers

    # set current input
    current_input = model.input

    # propagate through layers
    @inbounds for layer in layers

        # set input in layer
        setinput!(layer, current_input)

        # run current layer forward
        current_input = forward!(layer)::Vector{T} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
    
    end

    # update output of model
    setoutput!(model, current_input)

    # return output
    return model.output
    
end

function propagate_error!(model::Model{L,T,V1,V2}, gradient_output::Vector{T}) where { L, T <: Real, V1, V2 }

    # set gradient output in model
    setgradientoutput!(model, gradient_output)

    # propagate error backwards
    gradient_input = propagate_error!(model)

    # return gradient input
    return gradient_input
    
end

function propagate_error!(model::Model{L,T,V1,V2}) where { L, T <: Real, V1, V2 }

    # fetch layers
    layers = model.layers

    # set current gradient output
    current_gradient_output = model.gradient_output

    # propagate through layers
    @inbounds for layer in reverse(layers)

        # set gradient output in layer
        setgradientoutput!(layer, current_gradient_output)

        # run current layer forward
        current_gradient_output = propagate_error!(layer)::Vector{T} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
    
    end

    # update gradient input of model
    setgradientinput!(model, current_gradient_output)

    # return gradient input of model
    return model.gradient_input

end

function update!(model::Model)

    # fetch layers
    layers = model.layers

    # update parameters in layers
    @inbounds for layer in layers
        update!(layer)
    end

end

function setlr!(model::Model, lr)

    # fetch layers
    layers = model.layers

    # update parameters in layers
    @inbounds for layer in layers
        setlr!(layer, lr)
    end

end

function setbatchsize!(model::Model, batch_size::Int64)

    # fetch layers
    layers = model.layers

    # update parameters in layers
    @inbounds for layer in layers
        setbatchsize!(layer, batch_size)
    end

end