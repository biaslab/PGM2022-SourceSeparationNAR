mutable struct ReluLayer{M <: Memory} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    memory          :: M
end
function ReluLayer(dim; batch_size::Int64=128)
    return ReluLayer(dim, dim, Memory(dim,batch_size))
end

function forward(layer::ReluLayer{Nothing}, input)
    
    # fetch input and output in layer
    output = similar(input)
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

function forward!(layer::ReluLayer{<:Memory}) 
    
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

function propagate_error!(layer::ReluLayer{<:Memory}) 
    
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

update!(::ReluLayer{<:Memory}) = return

setlr!(::ReluLayer{<:Memory}, lr) = return

isinvertible(::ReluLayer) = false

nr_params(::ReluLayer) = 0

function print_info(layer::ReluLayer, level::Int, io)

    # print layer
    write(io, string(["--" for _=1:level]..., " ReluLayer(", layer.dim_in, ")\n"))

end

relu(x) = max(0.0, x)
drelu(x) = x > 0 ? 1.0 : 0.0