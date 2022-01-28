mutable struct NarLayer{M, T <: Real} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    f               :: M
    input           :: Matrix{T}
    output          :: Matrix{T}
    gradient_input  :: Matrix{T}
    gradient_output :: Matrix{T}
end
function NarLayer(dim, f; batch_size::Int64=128)
    @assert dim == f.dim_in + f.dim_out
    return NarLayer(dim, dim, f, zeros(dim,batch_size), zeros(dim,batch_size), zeros(dim,batch_size), zeros(dim,batch_size))
end

function forward!(layer::NarLayer) 
    
    # fetch input and output in layer
    input  = getmatinput(layer)
    output = getmatoutput(layer)
    f = layer.f
    dim_in = f.dim_in
    dim_out = f.dim_out
    (sz1, sz2) = size(input)

    # update output of layer (shifted component)
    @turbo for k1 in 1:dim_in
        for k2 in 1:sz2
            output[k1+dim_out,k2] = input[k1,k2]
        end
    end 

    # set input of internal function (custom to prevent allocs)
    f_input = f.input
    @turbo for k1 in 1:dim_in
        for k2 in 1:sz2
            f_input[k1,k2] = input[k1,k2]
        end
    end

    # run current function forward
    f_output = forward!(f)::Matrix{Float64}

    # process current output
    @turbo for k1 in 1:dim_out
        for k2 in 1:sz2
            output[k1,k2] = f_output[k1,k2] + input[k1+dim_in,k2]
        end
    end

    # return output 
    return output
    
end

function propagate_error!(layer::NarLayer) 
    
    # fetch input and output gradients in layer
    gradient_input  = getmatgradientinput(layer)
    gradient_output = getmatgradientoutput(layer)
    f = layer.f
    dim_in = f.dim_in
    dim_out = f.dim_out
    (sz1, sz2) = size(gradient_input)

    # update gradient input of layer (additive component)
    @turbo for k1 in 1:dim_out
        for k2 in 1:sz2
            gradient_input[k1+dim_in,k2] = gradient_output[k1,k2]
        end
    end 

    # set gradient output of internal function (custom to prevent allocs)
    f_gradient_output = f.gradient_output
    @turbo for k1 in 1:dim_out
        for k2 in 1:sz2
            f_gradient_output[k1,k2] = gradient_output[k1,k2]
        end
    end

    # run current function backward
    f_gradient_input = propagate_error!(f)::Matrix{Float64}

    # process current output
    @turbo for k1 in 1:dim_in
        for k2 in 1:sz2
            gradient_input[k1,k2] = f_gradient_input[k1,k2] + gradient_output[k1+dim_out,k2]
        end
    end

    # return gradient input 
    return gradient_input
    
end

update!(layer::NarLayer) = update!(layer.f)

setlr!(layer::NarLayer, lr) = setlr!(layer.f, lr)

isinvertible(::NarLayer) = true

nr_params(layer::NarLayer) = nr_params(layer.f)

function print_info(layer::NarLayer, level::Int, io)

    # print layer
    write(io, string(["--" for _=1:level]..., " NarLayer(", layer.dim_in, ")\n"))

    # print internal model
    print_info(layer.f, level+1, io)

end