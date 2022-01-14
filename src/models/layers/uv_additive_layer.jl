mutable struct UvAdditiveLayer{F <: Tuple, T <: Real} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    f               :: F
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function UvAdditiveLayer(dim::Int, f)
    # add assert
    return UvAdditiveLayer(dim, dim, (f,), zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end
function UvAdditiveLayer(dim::Int, f::Tuple)
    # add assert
    return UvAdditiveLayer(dim, dim, f, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end

function forward!(layer::UvAdditiveLayer)

    # fetch from layer
    dim    = layer.dim
    f      = layer.f
    input  = layer.input
    output = layer.output

    # set output in layer (additive component)
    set_output!(layer, input)

    # calculate output
    @inbounds for k in 1:dim-1

        # fetch current f
        current_f = f[k]

        # set input in current f
        setinput!(current_f, input[k])

        # run forward and update output
        output[k+1] += forward!(current_f)
        
    end

    # return output 
    return output
    
end

# function propagate_error!(layer::UvAdditiveLayer, ∂L_∂y::Vector{<:Real})

#     # set gradient at output of layer
#     gradient_output = layer.gradient_output
#     ∂L_∂x  = layer.gradient_input
#     dim = layer.dim
#     @inbounds for k in 1:dim
#         ∂L_∂yk = ∂L_∂y[k]
#         gradient_output[k] = ∂L_∂yk
#         ∂L_∂x[k]           = ∂L_∂yk
#     end

#     # propagate gradient to input of layer
#     @inbounds for k in 1:dim-1
#         ∂L_∂x[k] += propagate_error!(layer.f[k], ∂L_∂y[k+1])
#     end

#     # return gradient at input of layer
#     return ∂L_∂x

# end

# function update!(layer::UvAdditiveLayer)
#     f = layer.f
#     @inbounds for k in 1:length(f)
#         update!(f[k])
#     end
# end