mutable struct AdditiveCouplingLayerSplit{F <: AbstractFlow, T <: Real} <: AbstractCouplingLayer
    dim             :: Int
    f               :: F
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function AdditiveCouplingLayerSplit(dim::Int, f::T) where { T <: AbstractFlow}
    return AdditiveCouplingLayerSplit(dim, f, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end

function forward!(layer::AdditiveCouplingLayerSplit, x::Vector{<:Real})

    # set input and output in layer
    input = layer.input
    output = layer.output
    dim = layer.dim
     @inbounds for k in 1:dim
        xk = x[k]
        input[k] = xk
        output[k] = xk
    end
    
    # set input of flow (we partition a vector, so to prevent allocations we need to copy here)
    pdim = dim ÷ 2
    f = layer.f
    finput = f.input
     @inbounds for k in 1:pdim
        finput[k] = input[pdim+k]
    end

    # update flow
    outputi = forward!(f)

    # copy output
     @inbounds for k in 1:pdim
        output[pdim+k] += outputi[pdim]
    end

    # return output 
    return output
    
end

function propagate_error!(layer::AdditiveCouplingLayerSplit, ∂L_∂y::Vector{<:Real})

    # set gradient at output and input of layer
    gradient_output = layer.gradient_output
    ∂L_∂x  = layer.gradient_input
    dim = layer.dim
    @inbounds for k in 1:dim
        ∂L_∂yk = ∂L_∂y[k]
        gradient_output[k] = ∂L_∂yk
        ∂L_∂x[k]           = ∂L_∂yk
    end

    # set output gradient of flow (we partition a vector, so to prevent allocations we need to copy here)
    pdim = dim ÷ 2
    f = layer.f
    f_gradient_output = f.gradient_output
    @inbounds for k in 1:pdim
        f_gradient_output[k] = ∂L_∂y[pdim+k]
    end

    # propagate gradient through flow
    f_input_gradient = propagate_error!(f)

    # copy output
    @inbounds for k in 1:pdim
        ∂L_∂x[k] += f_input_gradient[k]
    end

    # return gradient at input of layer
    return ∂L_∂x

end

function update!(layer::AdditiveCouplingLayerSplit)
    update!(layer.f)
end