using LinearAlgebra, Random

using LoopVectorization: @turbo

mutable struct ResidualLayer{F, M<:Union{Nothing, Memory}} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    f               :: F
    memory          :: M
end
function ResidualLayer(dim, f; batch_size::Int64=128)
    return ResidualLayer(dim, dim, f, Memory(dim, batch_size))
end

function forward(layer::ResidualLayer, input)

    # fetch from layer
    output  = copy(input)
    f       = layer.f

    # run internal model forward
    output_f = forward(f, input)

    # add output of internal model to layer output
    @turbo output .+= output_f

    # return output 
    return output
    
end

function forward!(layer::ResidualLayer{F,<:Memory}) where { F }

    # fetch from layer
    input   = layer.input
    output  = layer.output
    f       = layer.f

    # copy the input to the output of the layer
    copytooutput!(layer, input)

    # set input of model in layer
    linktoinput!(f, input)

    # run internal model forward
    output_f = forward!(f)

    # add output of internal model to layer output
    @turbo output .+= output_f

    # return output 
    return output
    
end

function propagate_error!(layer::ResidualLayer{F,<:Memory}) where { F }

    # fetch from layer
    ∂L_∂x   = getmatgradientinput(layer)
    ∂L_∂y   = getmatgradientoutput(layer)
    f       = layer.f

    # set output gradient of model in layer
    linktogradientoutput!(f, ∂L_∂y)

    # propagate error in internal model
    gradient_input_f = propagate_error!(f)

    # copy gradient to input of layer
    copytogradientinput!(layer, ∂L_∂y)

    # add gradient of internal model
    @turbo ∂L_∂x .+= gradient_input_f

    # return gradient at input of layer
    return ∂L_∂x

end

function update!(layer::ResidualLayer{F,<:Memory}) where { F }

    # update model in layer
    update!(layer.f)
    
end

function setlr!(layer::ResidualLayer{F,<:Memory}, lr) where { F }

    # update learning rate in layer
    setlr!(layer.f, lr)
    
end

isinvertible(layer::ResidualLayer) = false

nr_params(layer::ResidualLayer) = nr_params(layer.f)

function print_info(layer::ResidualLayer, level::Int, io)

    # print layer
    write(io, string(["--" for _=1:level]..., " ResidualLayer(", layer.dim_in, ")\n"))

    # loop through model
    print_info(layer.f, level+1, io)

end