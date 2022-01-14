mutable struct MvAdditiveLayer{L <: Tuple, T <: Real} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    f               :: L
    input           :: Vector{T}
    output          :: Vector{T}
    gradient_input  :: Vector{T}
    gradient_output :: Vector{T}
end
function MvAdditiveLayer(dim::Int, f::L) where { L <: Tuple}
    # TODO add assert for dims in f
    return MvAdditiveLayer(dim, dim, f, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end
function MvAdditiveLayer(dim::Int, f)
    # TODO add assert for dims in f
    return MvAdditiveLayer(dim, dim, (f,), zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end

function forward!(layer::MvAdditiveLayer)

    # set output of layer (the additive component)
    input  = layer.input
    output = layer.output
    setoutput!(layer, input)
    
    # fetch partition dimension
    f     = layer.f
    len_f = length(f)
    pdim  = layer.dim_in ÷ (len_f + 1)

    # loop through coupling functions 
    for k in 1:len_f

        # fetch current function
        current_f = f[k]

        # set input of current function (custom to prevent allocs)
        current_f_input = current_f.input
        @inbounds for ki in 1:pdim
            current_f_input[ki] = input[ki + (k-1)*pdim]
        end

        # run current function forward
        current_f_output = forward!(current_f)::Vector{Float64}

        # process current output
        @inbounds for ki in 1:pdim
            output[ki+k*pdim] += current_f_output[ki]
        end

    end

    # return output 
    return output
    
end

# function propagate_error!(layer::AdditiveCouplingLayerSplit, ∂L_∂y::Vector{<:Real})

#     # set gradient at output and input of layer
#     gradient_output = layer.gradient_output
#     ∂L_∂x  = layer.gradient_input
#     dim = layer.dim
#     @inbounds for k in 1:dim
#         ∂L_∂yk = ∂L_∂y[k]
#         gradient_output[k] = ∂L_∂yk
#         ∂L_∂x[k]           = ∂L_∂yk
#     end

#     # set output gradient of flow (we partition a vector, so to prevent allocations we need to copy here)
#     pdim = dim ÷ 2
#     f = layer.f
#     f_gradient_output = f.gradient_output
#     @inbounds for k in 1:pdim
#         f_gradient_output[k] = ∂L_∂y[pdim+k]
#     end

#     # propagate gradient through flow
#     f_input_gradient = propagate_error!(f)

#     # copy output
#     @inbounds for k in 1:pdim
#         ∂L_∂x[k] += f_input_gradient[k]
#     end

#     # return gradient at input of layer
#     return ∂L_∂x

# end

# function update!(layer::AdditiveCouplingLayerSplit)
#     update!(layer.f)
# end