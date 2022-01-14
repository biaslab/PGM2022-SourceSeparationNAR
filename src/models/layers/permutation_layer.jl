mutable struct PermutationLayer{T <: Real} <: AbstractLayer
    dim_in          :: Int64
    P               :: PermutationMatrix
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function PermutationLayer(dim::Int)
    return PermutationLayer(dim, PermutationMatrix(dim))
end
function PermutationLayer(dim::Int, P::PermutationMatrix)
    return PermutationLayer(dim, P, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end

function forward!(layer::PermutationLayer)
    
    # calculate output of layer
    input = layer.input
    output = layer.output
    mul!(output, layer.P, input)

    # return output 
    return output
    
end

function propagate_error!(layer::PermutationLayer)

    # copy input in layer
    gradient_output = layer.gradient_output
    gradient_input  = layer.gradient_input
    
    # calculate output of layer
    mulT!(gradient_input, layer.P, gradient_output) # mul!(gradient_input, P', ∂L_∂y)

    # return gradient at input of layer
    return gradient_input

end

# update!(layer::PermutationLayer) = return