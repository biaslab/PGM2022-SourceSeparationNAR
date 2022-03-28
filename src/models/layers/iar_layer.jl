mutable struct iARLayer{F, M<:Union{Nothing, AbstractMemory}} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    f               :: F
    memory          :: M
end
function iARLayer(dim, f; batch_size::Int64=128)
    @assert dim == f.dim_in + f.dim_out
    return iARLayer(dim, dim, f, TrainMemory(dim, batch_size))
end

function forward(layer::iARLayer, input::Vector)
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
        output[k] = f_output[k] + input[k+dim_in]
    end

    # return output
    return output

end

function forward!(layer::iARLayer{F,<:AbstractMemory}, input) where { F }

    # set input
    copytoinput!(layer, input)

    # calculate and return output 
    return forward!(layer)

end

function forward!(layer::iARLayer{F,<:TrainMemory}) where { F } 
    
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
            output[k1,k2] = f_output[k1,k2] + input[k1+dim_in,k2]
        end
    end

    # return output 
    return output
    
end

function forward!(layer::iARLayer{F,<:DeployMemory}) where { F } 
    
    # fetch input and output in layer
    input  = getmatinput(layer)
    output = getmatoutput(layer)
    f = layer.f
    dim_in = f.dim_in
    dim_out = f.dim_out

    # update output of layer (shifted component)
    @turbo for k1 in 1:dim_in
        output[k1+dim_out] = input[k1]
    end 

    # set input of internal function (custom to prevent allocs)
    f_input = f.memory.input
    @turbo for k1 in 1:dim_in
        f_input[k1] = input[k1]
    end

    # run current function forward
    f_output = forward!(f)::Vector{Float64}

    # process current output
    @turbo for k1 in 1:dim_out
        output[k1] = f_output[k1] + input[k1+dim_in]
    end

    # return output 
    return output
    
end

function backward(layer::iARLayer, output::Vector)

    input = similar(output)
    f = layer.f
    dim_in = f.dim_in
    dim_out = f.dim_out

    # update input of layer (shifted component)
    @turbo for k in 1:dim_in
        input[k] = output[k+dim_out]
    end 

    # run current function forward
    f_output = forward(f, input[1:dim_in])::Vector{Float64}

    # process current input
    @turbo for k in 1:dim_out
        input[k+dim_in] = - f_output[k] + output[k]
    end

    # return input
    return input

end

function jacobian(layer::iARLayer, input::Vector{T}) where { T <: Real }

    # fetch information
    len = length(input)
    f = layer.f

    # initialize jacobian
    J = zeros(T, len, len)

    # set identity diagonal
    @inbounds for k = 1:len-f.dim_out
        J[k+f.dim_out,k] = 1
    end
    @inbounds for k = 1:len-f.dim_in
        J[k,k+f.dim_in] = 1
    end

    # fetch jacobian of internal function
    J_internal = jacobian(f, input[1:f.dim_in])

    # update jacobian of layer with internal jacobian
    J[1:f.dim_out, 1:f.dim_in] .= J_internal

    # return jacobian
    return J

end

function invjacobian(layer::iARLayer, output::Vector{T}) where { T <: Real }

    # fetch information
    len = length(output)
    f = layer.f

    # initialize inverse jacobian
    invJ = zeros(T, len, len)

    # set identity diagonal
    @inbounds for k = 1:len-f.dim_out
        invJ[k,k+f.dim_out] = 1
    end
    @inbounds for k = 1:len-f.dim_in
        invJ[k+f.dim_in,k] = 1
    end

    # fetch jacobian of internal function
    J_internal = jacobian(f, output[f.dim_out+1:end])

    # update inverse jacobian of layer with internal jacobian
    invJ[f.dim_in+1:end, f.dim_out+1:end] .= -J_internal

    # return inverse jacobian
    return invJ

end


function propagate_error!(layer::iARLayer{F,<:TrainMemory}) where { F } 
    
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

update!(layer::iARLayer{F,<:TrainMemory}) where { F } = update!(layer.f)

setlr!(layer::iARLayer{F,<:TrainMemory}, lr) where { F } = setlr!(layer.f, lr)

isinvertible(::iARLayer) = true

nr_params(layer::iARLayer) = nr_params(layer.f)

function print_info(layer::iARLayer, level::Int, io)

    # print layer
    write(io, string(["--" for _=1:level]..., " iARLayer(", layer.dim_in, ")\n"))

    # print internal model
    print_info(layer.f, level+1, io)

end