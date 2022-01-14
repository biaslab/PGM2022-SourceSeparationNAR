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
    # todo: add assert
    return UvAdditiveLayer(dim, dim, (f,), zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end
function UvAdditiveLayer(dim::Int, f::Tuple)
    # todo: add assert
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

function propagate_error!(layer::UvAdditiveLayer, ∂L_∂y::Vector{<:Real})

    # set gradient input of layer (the additive component)
    gradient_input  = layer.gradient_input
    gradient_output = layer.gradient_output
    setgradientinput!(layer, gradient_output)
    
    # fetch partition dimension
    f     = layer.f
    len_f = length(f)

    # loop through coupling functions 
    for k in 1:len_f

        # fetch current function
        current_f = f[k]

        # set gradient outputs of current function
        setgradientoutput!(current_f, gradient_output[k+1])

        # run current function error backward
        gradient_input[k] += propagate_error!(current_f)::Float64

    end

    # return gradient at input of layer
    return gradient_input
    
end

function update!(layer::UvAdditiveLayer)
    f = layer.f
    @inbounds for k in 1:length(f)
        update!(f[k])
    end
end