mutable struct PermutationLayer{T <: Real} <: AbstractCouplingLayer
    dim             :: Int64
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

function forward!(layer::PermutationLayer, x::Vector{<:Real})

    # copy input in layer
    dim = layer.dim
    input = layer.input
    @inbounds for k in 1:dim
        input[k] = x[k]
    end
    
    # calculate output of layer
    output = layer.output
    mul!(output, layer.P, x)

    # return output 
    return output
    
end

function propagate_error!(layer::PermutationLayer, ∂L_∂y::Vector{<:Real})

    # copy input in layer
    dim = layer.dim
    gradient_output = layer.gradient_output
    @inbounds for k in 1:dim
        gradient_output[k] = ∂L_∂y[k]
    end
    
    # calculate output of layer
    gradient_input = layer.gradient_input
    mulT!(gradient_input, layer.P, ∂L_∂y) # mul!(gradient_input, P', ∂L_∂y)

    # return gradient at input of layer
    return gradient_input

end

update!(layer::PermutationLayer) = return