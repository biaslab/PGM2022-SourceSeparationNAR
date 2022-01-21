mutable struct MvAdditiveLayer{L <: Tuple, T <: Real} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    f               :: L
    input           :: Matrix{T}
    output          :: Matrix{T}
    gradient_input  :: Matrix{T}
    gradient_output :: Matrix{T}
end
function MvAdditiveLayer(dim::Int, f::L; batch_size::Int64=128) where { L <: Tuple}
    for k in length(f)-1
        @assert f[k+1].dim_in == f[k].dim_out
    end
    ind = 0
    for fk in f
        ind += fk.dim_in 
    end
    @assert ind + f[end].dim_out == dim
    return MvAdditiveLayer(dim, dim, f, zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim, batch_size), zeros(dim, batch_size))
end
function MvAdditiveLayer(dim::Int, f; batch_size::Int64=128)
    @assert f.dim_in + f.dim_out == dim
    return MvAdditiveLayer(dim, dim, (f,), zeros(dim, batch_size), zeros(dim,batch_size), zeros(dim,batch_size), zeros(dim,batch_size))
end

function forward!(layer::MvAdditiveLayer)

    # set output of layer (the additive component)
    input  = layer.input
    output = layer.output
    copytooutput!(layer, input)
    batch_size = size(input,2)
    
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
        @turbo for k1 in 1:current_f_dim
            for k2 in 1:batch_size
                current_f_input[k1,k2] = input[k1 + ind,k2]
            end
        end

        # run current function forward
        current_f_output = forward!(current_f)::Matrix{Float64}

        # process current output
        @turbo for k1 in 1:current_f_dim
            for k2 in 1:batch_size
                output[k1+current_f_dim+ind,k2] += current_f_output[k1,k2]
            end
        end

        # update index
        ind += current_f_dim

    end

    # return output 
    return output
    
end

function propagate_error!(layer::MvAdditiveLayer)

    # set gradient input of layer (the additive component)
    gradient_input  = getmatgradientinput(layer)
    gradient_output = getmatgradientoutput(layer)
    copytogradientinput!(layer, gradient_output)
    batch_size = size(gradient_input,2)
    
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
        current_f_gradient_output = getmatgradientoutput(current_f)
        @turbo for k1 in 1:current_f_dim
            for k2 in 1:batch_size
                current_f_gradient_output[k1,k2] = gradient_output[k1 + current_f_dim + ind,k2]
            end
        end

        # run current function error backward
        current_f_gradient_input = propagate_error!(current_f)::Matrix{Float64}

        # process current gradient input
        @turbo for k1 in 1:current_f_dim
            for k2 in 1:batch_size
                gradient_input[k1+ind,k2] += current_f_gradient_input[k1,k2]
            end
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

isinvertible(::MvAdditiveLayer) = true

nr_params(layer::MvAdditiveLayer) = mapreduce(nr_params, +, layer.f)