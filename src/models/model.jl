export Model
export DenseLayer, LeakyReluLayer, MvAdditiveLayer, PermutationLayer, ReluLayer, ResidualLayer, SoftmaxLayer

export forward!, propagate_error!, update!
export setlr!, setbatchsize!
export isinvertible, nr_params, print_info

abstract type AbstractModel end
abstract type AbstractLayer end


# include parameters
include("parameter.jl")

# include layers
include("layers/dense.jl")
include("layers/leakyrelu.jl")
include("layers/mv_additive_layer.jl")
include("layers/permutation_layer.jl")
include("layers/relu.jl")
include("layers/residual_layer.jl")
include("layers/softmax.jl")

# include helpers
include("helpers.jl")

mutable struct Model{L <: Tuple, T <: Real, V1 <: AbstractMatrix, V2 <: AbstractMatrix } <: AbstractModel
    dim_in          :: Int64
    dim_out         :: Int64
    layers          :: L
    input           :: Matrix{T}
    output          :: V1
    gradient_input  :: Matrix{T}
    gradient_output :: V2
end
function Model(dim, layers; batch_size::Int64=128)
    return Model(dim, dim, layers; batch_size=batch_size)
end
function Model(dim_in, dim_out, layers; batch_size::Int64=128)
    if typeof(last(layers)) <: SoftmaxLayer
        return Model(dim_in, dim_out, layers, zeros(dim_in,batch_size), SoftmaxOutput(zeros(dim_out, batch_size)), zeros(dim_in, batch_size), SoftmaxGradientOutput(zeros(dim_out, batch_size)))
    else
        return Model(dim_in, dim_out, layers, zeros(dim_in,batch_size), zeros(dim_out,batch_size), zeros(dim_in,batch_size), zeros(dim_out,batch_size))
    end
end

function forward!(model::Model{L,T,V1,V2}, input::Matrix{T}) where { L, T <: Real, V1, V2}

    # set input in model
    copytoinput!(model, input)

    # run model forward
    output = forward!(model)

    # return output
    return output

end

function forward!(model::Model{L,T,V1,V2}) where { L, T <: Real, V1, V2 }

    # fetch layers
    layers = model.layers

    # set current input
    current_input = getmatinput(model)

    # propagate through layers
    @inbounds for layer in layers

        # set input in layer
        linktoinput!(layer, current_input)
        
        # run current layer forward
        current_input = forward!(layer)::Matrix{T} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
    
    end

    # update output of model
    copytooutput!(model, current_input)

    # return output
    return model.output
    
end

function propagate_error!(model::Model{L,T,V1,V2}, gradient_output::Matrix{T}) where { L, T <: Real, V1, V2 }

    # set gradient output in model
    copytogradientoutput!(model, gradient_output)

    # propagate error backwards
    gradient_input = propagate_error!(model)

    # return gradient input
    return gradient_input
    
end

function propagate_error!(model::Model{L,T,V1,V2}) where { L, T <: Real, V1, V2 }

    # fetch layers
    layers = model.layers

    # set current gradient output
    current_gradient_output = getmatgradientoutput(model)

    # propagate through layers
    @inbounds for layer in reverse(layers)

        # set gradient output in layer
        linktogradientoutput!(layer, current_gradient_output)

        # run current layer forward
        current_gradient_output = propagate_error!(layer)::Matrix{T} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
    
    end

    # update gradient input of model
    copytogradientinput!(model, current_gradient_output)

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

isinvertible(model::Model) = mapreduce(isinvertible, *, model.layers)

nr_params(model::Model) = mapreduce(nr_params, +, model.layers)

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