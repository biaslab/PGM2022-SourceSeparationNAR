export Model
export DenseLayer, MvAdditiveLayer, PermutationLayer, ReluLayer, UvAdditiveLayer

export forward!#, propagate_error!, update!

abstract type AbstractModel end
abstract type AbstractLayer end

# include helpers
include("helpers.jl")

# include parameters
include("parameter.jl")

# include layers
include("layers/dense.jl")
include("layers/mv_additive_layer.jl")
# include("layers/uv_additive_layer.jl")
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

# function propagate_error!(model::FlowModel{L,T}, ∂L_∂y::Vector{T}) where { L, T <: Real }

#     # set gradient at output of layer and input as running input
#     gradient_output = model.gradient_output
#     gradient_input  = model.gradient_input
#     dim             = model.dim
#     @inbounds for k in 1:dim
#         ∂L_∂yk = ∂L_∂y[k]
#         gradient_output[k] = ∂L_∂yk
#         gradient_input[k] = ∂L_∂yk
#     end

#     # propagate gradient through layers
#     layers = model.layers
#     @inbounds for k in 1:length(layers)
#         layerk = layers[k]
#         gradient_inputi = propagate_error!(layerk, gradient_input)::Vector{T} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
#         @inbounds for ki in 1:dim
#             gradient_input[ki] = gradient_inputi[ki]
#         end
#     end

#     # return gradient at input of layer
#     return gradient_input

# end

# function update!(model::FlowModel)
#     layers = model.layers
#     @inbounds for k in 1:length(layers)
#         update!(layers[k])
#     end
# end