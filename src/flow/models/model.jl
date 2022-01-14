mutable struct FlowModel{L <: Tuple, T <: Real} <: AbstractModel
    dim             :: Int
    layers          :: L
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function FlowModel(dim, layers)
    return FlowModel(dim, layers, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end

function forward!(model::FlowModel{L,T}, x::Vector{T}) where { L, T <: Real }

    # set input in model and set output to running input
    input = model.input
    output = model.output
    dim = model.dim
     @inbounds for k in 1:dim
        xk = x[k]
        input[k] = xk
        output[k] = xk
    end

    # propagate through layers
    layers = model.layers
     @inbounds for k in 1:length(layers)
        layerk = layers[k]
        outputi = forward!(layerk, output)::Vector{T} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
        @inbounds for ki in 1:dim
            output[ki] = outputi[ki]
        end
    end

    # return output 
    return output
    
end

function propagate_error!(model::FlowModel{L,T}, ∂L_∂y::Vector{T}) where { L, T <: Real }

    # set gradient at output of layer and input as running input
    gradient_output = model.gradient_output
    gradient_input  = model.gradient_input
    dim             = model.dim
     @inbounds for k in 1:dim
        ∂L_∂yk = ∂L_∂y[k]
        gradient_output[k] = ∂L_∂yk
        gradient_input[k] = ∂L_∂yk
    end

    # propagate gradient through layers
    layers = model.layers
    @inbounds for k in 1:length(layers)
        layerk = layers[k]
        gradient_inputi = propagate_error!(layerk, gradient_input)::Vector{T} # for type stability. Having more than 3 different layer types results into Tuple{Any}, from which the output of forward! cannot be determined anymore
        @inbounds for ki in 1:dim
            gradient_input[ki] = gradient_inputi[ki]
        end
    end

    # return gradient at input of layer
    return gradient_input

end

function update!(model::FlowModel)
    layers = model.layers
     @inbounds for k in 1:length(layers)
        update!(layers[k])
    end
end