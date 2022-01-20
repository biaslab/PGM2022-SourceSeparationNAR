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
    for k in length(f)-1
        @assert f[k+1].dim_in == f[k].dim_out
    end
    ind = 0
    for fk in f
        ind += fk.dim_in 
    end
    @assert ind + f[end].dim_out == dim
    return MvAdditiveLayer(dim, dim, f, zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end
function MvAdditiveLayer(dim::Int, f)
    @assert f.dim_in + f.dim_out == dim
    return MvAdditiveLayer(dim, dim, (f,), zeros(dim), zeros(dim), zeros(dim), zeros(dim))
end

function forward!(layer::MvAdditiveLayer)

    # set output of layer (the additive component)
    input  = layer.input
    output = layer.output
    setoutput!(layer, input)
    
    # fetch coupling functions
    f     = layer.f
    len_f = length(f)

    # set starting index
    ind   = 0

    # loop through coupling functions 
    @inbounds for k in 1:len_f

        # fetch current function
        current_f = f[k]

        # set input of current function (custom to prevent allocs)
        current_f_input = current_f.input
        current_f_dim   = current_f.dim_in
        @turbo for ki in 1:current_f_dim
            current_f_input[ki] = input[ki + ind]
        end

        # run current function forward
        current_f_output = forward!(current_f)::Vector{Float64}

        # process current output
        @turbo for ki in 1:current_f_dim
            output[ki+current_f_dim+ind] += current_f_output[ki]
        end

        # update index
        ind += current_f_dim

    end

    # return output 
    return output
    
end

function propagate_error!(layer::MvAdditiveLayer)

    # set gradient input of layer (the additive component)
    gradient_input  = layer.gradient_input
    gradient_output = layer.gradient_output
    setgradientinput!(layer, gradient_output)
    
    # fetch partition dimension
    f     = layer.f
    len_f = length(f)
    
    # set starting index
    ind   = 0

    # loop through coupling functions 
    @inbounds for k in 1:len_f

        # fetch current function
        current_f = f[k]
        current_f_dim = current_f.dim_in

        # set gradient outputs of current function (custom to prevent allocs)
        current_f_gradient_output = current_f.gradient_output
        @turbo for ki in 1:current_f_dim
            current_f_gradient_output[ki] = gradient_output[ki + current_f_dim + ind]
        end

        # run current function error backward
        current_f_gradient_input = propagate_error!(current_f)::Vector{Float64}

        # process current gradient input
        @turbo for ki in 1:current_f_dim
            gradient_input[ki+ind] += current_f_gradient_input[ki]
        end
        
        # update index
        ind += current_f_dim

    end

    # return gradient at input of layer
    return gradient_input

end

function update!(layer::MvAdditiveLayer)
    
    # fetch functions
    f = layer.f

    # update parameters in functions
    @inbounds for fk in f
        update!(fk)
    end
    
end

function setlr!(layer::MvAdditiveLayer, lr)
    
    # fetch functions
    f = layer.f

    # update parameters in functions
    @inbounds for fk in f
        setlr!(fk, lr)
    end
    
end

function setbatchsize!(layer::MvAdditiveLayer, batch_size::Int64)
    
    # fetch functions
    f = layer.f

    # update parameters in functions
    @inbounds for fk in f
        setbatchsize!(fk, batch_size)
    end
    
end

isinvertible(layer::MvAdditiveLayer) = true

nr_params(layer::MvAdditiveLayer) = mapreduce(nr_params, +, layer.f)