mutable struct AdditiveCouplingLayer{F <: Tuple, T <: Real} <: AbstractCouplingLayer
    dim             :: Int
    f               :: F
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function AdditiveCouplingLayer(dim::Int, f::T) where { T <: AbstractFlow}
    return AdditiveCouplingLayer(dim, (f,), zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end
function AdditiveCouplingLayer(dim::Int, f::Tuple)
    return AdditiveCouplingLayer(dim, f, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end

function forward!(layer::AdditiveCouplingLayer, x::Vector{<:Real})

    # set input and output in layer
    input = layer.input
    output = layer.output
    dim = layer.dim
     @inbounds for k in 1:dim
        xk = x[k]
        input[k] = xk
        output[k] = xk
    end

    # calculate output
     @inbounds for k in 1:dim-1
        output[k+1] += forward!(layer.f[k], x[k])
    end

    # return output 
    return output
    
end

function propagate_error!(layer::AdditiveCouplingLayer, ∂L_∂y::Vector{<:Real})

    # set gradient at output of layer
    gradient_output = layer.gradient_output
    ∂L_∂x  = layer.gradient_input
    dim = layer.dim
     @inbounds for k in 1:dim
        ∂L_∂yk = ∂L_∂y[k]
        gradient_output[k] = ∂L_∂yk
        ∂L_∂x[k]           = ∂L_∂yk
    end

    # propagate gradient to input of layer
     @inbounds for k in 1:dim-1
        ∂L_∂x[k] += propagate_error!(layer.f[k], ∂L_∂y[k+1])
    end

    # return gradient at input of layer
    return ∂L_∂x

end

function update!(layer::AdditiveCouplingLayer)
    f = layer.f
     @inbounds for k in 1:length(f)
        update!(f[k])
    end
end