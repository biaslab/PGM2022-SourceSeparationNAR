mutable struct ARLayer{F, M<:Union{Nothing, AbstractMemory}} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    f               :: F
    memory          :: M
end
function ARLayer(dim, f; batch_size::Int64=128)
    @assert dim == f.dim_in + f.dim_out
    return  ARLayer(
                dim, 
                dim, 
                f, 
                TrainMemory(dim, batch_size)
            )
end

function forward(layer::ARLayer, input::Vector)
    output = similar(input)
    f = layer.f
    dim_in = f.dim_in
    dim_out = f.dim_out

    # update output of layer (shifted component)
    @turbo for k in 1:dim_in
        output[k+dim_out] = input[k]
    end 

    # run current function forward
    f_output = forward(f, input[1:dim_in])#::Vector{Float64}

    # process current output
    @turbo for k in 1:dim_out
        output[k] = f_output[k]
    end

    # return output
    return output

end

function jacobian(layer::ARLayer, input::Vector{T}) where { T <: Real }

    # fetch information
    len = length(input)
    f = layer.f

    # initialize jacobian
    J = zeros(T, len, len)

    # set identity diagonal
    @inbounds for k = 1:len-f.dim_out
        J[k+f.dim_out,k] = 1
    end

    # fetch jacobian of internal function
    J_internal = jacobian(f, input[1:f.dim_in])

    # update jacobian of layer with internal jacobian
    J[1:f.dim_out, 1:f.dim_in] .= J_internal

    # return jacobian
    return J

end

function forward!(layer::ARLayer{F,<:TrainMemory}) where { F } 
    
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
    f_input = f.memory.input
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
            output[k1,k2] = f_output[k1,k2]
        end
    end

    # return output 
    return output
    
end

function propagate_error!(layer::ARLayer{F,<:TrainMemory}) where { F } 
    
    # fetch input and output gradients in layer
    gradient_input  = getmatgradientinput(layer)
    gradient_output = getmatgradientoutput(layer)
    f = layer.f
    dim_in = f.dim_in
    dim_out = f.dim_out
    (sz1, sz2) = size(gradient_input)

    # set gradient output of internal function (custom to prevent allocs)
    f_gradient_output = getmatgradientoutput(f)
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

update!(layer::ARLayer{F,<:TrainMemory}) where { F } = update!(layer.f)

setlr!(layer::ARLayer{F,<:TrainMemory}, lr) where { F } = setlr!(layer.f, lr)

isinvertible(::ARLayer) = true

nr_params(layer::ARLayer) = nr_params(layer.f)

function deploy(layer::ARLayer, start_dim)
    return ARLayer(
        layer.dim_in,
        layer.dim_out,
        deploy(layer.f),
        DeployMemory(layer.dim_in, layer.dim_out, start_dim)
    )
end

function print_info(layer::ARLayer, level::Int, io)

    # print layer
    write(io, string(["--" for _=1:level]..., " ARLayer(", layer.dim_in, ")\n"))

    # print internal model
    print_info(layer.f, level+1, io)

end