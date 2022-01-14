mutable struct ReluLayer{T <: Real} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function ReluLayer(dim)
    return ReluLayer(dim, dim, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end

function forward!(layer::ReluLayer) 
    
    # fetch input and output in layer
    dim    = layer.dim_in
    input  = layer.input
    output = layer.output

    # update output of layer
    @inbounds for k = 1:dim
        output[k] = relu(input[k])
    end

    # return output 
    return output
    
end

# function propagate_error!(layer::ReluLayer, ∂L_∂y::Vector{<:Real})

#     # set gradients for input and output and bias term
#     dim = length(∂L_∂y)
#     gradient_output = layer.gradient_output
#     ∂L_∂x           = layer.gradient_input
#     input           = layer.input
#     @inbounds for k in 1:dim
#         ∂L_∂yk = ∂L_∂y[k]
#         gradient_output[k] = ∂L_∂yk
#         ∂L_∂x[k] = ∂L_∂yk*drelu(input[k])
#     end

#     # return gradient at input of layer
#     return ∂L_∂x

# end

# update!(layer::ReluLayer) = return

relu(x) = max(0.0, x)
drelu(x) = x > 0 ? 1.0 : 0.0