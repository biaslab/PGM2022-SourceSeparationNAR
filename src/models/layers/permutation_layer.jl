mutable struct PermutationLayer{T <: Real} <: AbstractLayer
    dim_in          :: Int64
    P               :: PermutationMatrix
    input           :: Matrix{T}
    output          :: Matrix{T}
    gradient_input  :: Matrix{T}
    gradient_output :: Matrix{T}
end
function PermutationLayer(dim::Int; batch_size::Int64=128)
    return PermutationLayer(dim, PermutationMatrix(dim); batch_size=batch_size)
end
function PermutationLayer(dim::Int, P::PermutationMatrix; batch_size::Int64=128)
    return PermutationLayer(dim, P, zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim, batch_size))
end

function forward!(layer::PermutationLayer)
    
    # calculate output of layer
    input = getmatinput(layer)
    output = getmatoutput(layer)
    mul!(output, layer.P, input)

    # return output 
    return output
    
end

function propagate_error!(layer::PermutationLayer)

    # copy input in layer
    gradient_input  = getmatgradientinput(layer)
    gradient_output = getmatgradientoutput(layer)
    
    # calculate output of layer
    mulT!(gradient_input, layer.P, gradient_output) # mul!(gradient_input, P', ∂L_∂y)

    # return gradient at input of layer
    return gradient_input

end

update!(::PermutationLayer) = return

setlr!(::PermutationLayer, lr) = return

isinvertible(::PermutationLayer) = true

nr_params(::PermutationLayer) = 0