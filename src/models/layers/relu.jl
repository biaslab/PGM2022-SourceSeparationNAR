mutable struct ReluLayer{M <: Union{Nothing, AbstractMemory}} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    memory          :: M
end
function ReluLayer(dim; batch_size::Int64=128)
    return ReluLayer(dim, dim, TrainMemory(dim,batch_size))
end

function forward(::ReluLayer, input)
    
    # update output of layer and return
    return relu.(input)
    
end

function forward!(layer::ReluLayer{<:AbstractMemory}, input)

    # set input
    copytoinput!(layer, input)

    # call forward function and return output
    return forward!(layer)

end

function forward!(layer::ReluLayer{<:TrainMemory}) 
    
    # fetch input and output in layer
    input  = getmatinput(layer)
    output = getmatoutput(layer)
    (ax1, ax2) = axes(input)

    # update output of layer
    @turbo for k1 in ax1
        for k2 in ax2
            output[k1,k2] = relu(input[k1,k2])
        end
    end

    # return output 
    return output
    
end

function forward!(layer::ReluLayer{<:DeployMemory}) 
    
    # fetch input and output in layer
    input  = getmatinput(layer)
    output = getmatoutput(layer)
    dim    = layer.dim_in

    # update output of layer
    @turbo for k in 1:dim
        output[k] = relu(input[k])
    end

    # return output 
    return output
    
end

function jacobian(::ReluLayer, input::Vector{T}) where { T <: Real }

    # create jacobian
    tmp = Vector{Float64}(undef, length(input))
    @turbo for k in 1:length(input)
        tmp[k] = !signbit(input[k])
    end
    J = Diagonal(tmp)

    # return jacobian
    return J
end

function propagate_error!(layer::ReluLayer{<:TrainMemory}) 
    
    # fetch input and output gradients in layer
    input           = getmatinput(layer)
    gradient_input  = getmatgradientinput(layer)
    gradient_output = getmatgradientoutput(layer)
    (ax1, ax2) = axes(input)

    # update input gradient of layer
    @turbo for k1 in ax1
        for k2 in ax2
            gradient_input[k1,k2] = !signbit(input[k1,k2]) * gradient_output[k1,k2]
        end
    end

    # return gradient input 
    return gradient_input
    
end

update!(::ReluLayer{<:TrainMemory}) = return

setlr!(::ReluLayer{<:TrainMemory}, lr) = return

isinvertible(::ReluLayer) = false

nr_params(::ReluLayer) = 0

function print_info(layer::ReluLayer, level::Int, io)

    # print layer
    write(io, string(["--" for _=1:level]..., " ReluLayer(", layer.dim_in, ")\n"))

end

relu(x) = max(0.0, x)
drelu(x) = x > 0 ? 1.0 : 0.0