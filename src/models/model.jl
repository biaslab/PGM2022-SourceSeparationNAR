export Model
export DenseLayer, MvAdditiveLayer, PermutationLayer, ReluLayer, UvAdditiveLayer

export forward!, propagate_error!#, update!

abstract type AbstractModel end
abstract type AbstractLayer end

# include helpers
include("helpers.jl")

# include parameters
include("parameter.jl")

# include layers
include("layers/dense.jl")
include("layers/mv_additive_layer.jl")
include("layers/uv_additive_layer.jl")
include("layers/permutation_layer.jl")
include("layers/relu.jl")


mutable struct Model{L <: Tuple, T <: Real} <: AbstractModel
    dim_in          :: Int64
    dim_out         :: Int64
    layers          :: L
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function Model(dim, layers)
    return Model(dim, dim, layers, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end
function Model(dim_in, dim_out, layers)
    return Model(dim_in, dim_out, layers, zeros(dim_in), zeros(dim_out), zeros(dim_in), zeros(dim_out))
end

function forward!(model::Model{L,T}, input::Vector{T}) where { L, T <: Real }

    # set input in model
    setinput!(model, input)

    # run model forward
    output = forward!(model)

    # return output
    return output

end

function forward!(model::Model{L,T}) where { L, T <: Real }

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
    return current_input
    
end

function propagate_error!(model::Model{L,T}, gradient_output::Vector{T}) where { L, T <: Real }

    # set gradient output in model
    setgradientoutput!(model, gradient_output)

    # propagate error backwards
    gradient_input = propagate_error!(model)

    # return gradient input
    return gradient_input
    
end

function propagate_error!(model::Model{L,T}) where { L, T <: Real }

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
    return current_gradient_output

end

# function update!(model::Model)
#     layers = model.layers
#     @inbounds for k in 1:length(layers)
#         update!(layers[k])
#     end
# end