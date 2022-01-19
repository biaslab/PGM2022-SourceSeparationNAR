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

function propagate_error!(layer::MvAdditiveLayer)

    # set gradient input of layer (the additive component)
    gradient_input  = layer.gradient_input
    gradient_output = layer.gradient_output
    setgradientinput!(layer, gradient_output)
    
    # fetch partition dimension
    f     = layer.f
    len_f = length(f)
    pdim  = layer.dim_in ÷ (len_f + 1)

    # loop through coupling functions 
    for k in 1:len_f

        # fetch current function
        current_f = f[k]

        # set gradient outputs of current function (custom to prevent allocs)
        current_f_gradient_output = current_f.gradient_output
        @inbounds for ki in 1:pdim
            current_f_gradient_output[ki] = gradient_output[ki + k*pdim]
        end

        # run current function error backward
        current_f_gradient_input = propagate_error!(current_f)::Vector{Float64}

        # process current gradient input
        @inbounds for ki in 1:pdim
            gradient_input[ki+(k-1)*pdim] += current_f_gradient_input[ki]
        end

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