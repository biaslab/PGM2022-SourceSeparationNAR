mutable struct PermutationLayer{M <: Union{Nothing,<:AbstractMemory}} <: AbstractLayer
    dim_in          :: Int64
    P               :: PermutationMatrix
    memory          :: M
end
function PermutationLayer(dim::Int; batch_size::Int64=128)
    return PermutationLayer(dim, PermutationMatrix(dim); batch_size=batch_size)
end
function PermutationLayer(dim::Int, P::PermutationMatrix; batch_size::Int64=128)
    return PermutationLayer(dim, P, TrainMemory(dim, batch_size))
end

function forward(layer::PermutationLayer, input)
    
    # calculate output of layer
    output = mul(layer.P, input)

    # return output 
    return output
    
end

function forward!(layer::PermutationLayer{<:TrainMemory})
    
    # calculate output of layer
    input = getmatinput(layer)
    output = getmatoutput(layer)
    mul!(output, layer.P, input)

    # return output 
    return output
    
end

function propagate_error!(layer::PermutationLayer{<:TrainMemory})

    # copy input in layer
    gradient_input  = getmatgradientinput(layer)
    gradient_output = getmatgradientoutput(layer)
    
    # calculate output of layer
    mulT!(gradient_input, layer.P, gradient_output) # mul!(gradient_input, P', ∂L_∂y)

    # return gradient at input of layer
    return gradient_input

end

update!(::PermutationLayer{<:TrainMemory}) = return

setlr!(::PermutationLayer{<:TrainMemory}, lr) = return

isinvertible(::PermutationLayer) = true

nr_params(::PermutationLayer) = 0

function print_info(layer::PermutationLayer, level::Int, io)

    # print layer
    write(io, string(["--" for _=1:level]..., " PermutationLayer(", layer.dim_in, ")\n"))

end