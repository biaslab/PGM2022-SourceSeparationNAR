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

function forward!(layer::ARLayer{F,<:AbstractMemory}, input) where { F }

    # set input in model
    copytoinput!(layer, input)

    # run model forward and return output
    return forward!(layer)

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

function forward!(layer::ARLayer{F,<:DeployMemory}) where { F } 
    
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
        output[k1] = f_output[k1]
    end

    # return output 
    return output
    
end

# todo: create specialized companion matrix jacobian when shift = 1
function forward_jacobian!(layer::ARLayer)
    
    # fetch internal
    input_layer = getmatinput(layer)
    output_layer = getmatoutput(layer)
    jacobian_layer = getmatjacobian(layer)
    f = layer.f
    dim_in = f.dim_in
    dim_out = f.dim_out

    # copy input to internal
    for k = 1:dim_in
        f.memory.input[k] = layer.memory.input[k]
    end

    # copy gradient to internal
    linktojacobianinput!(f, IdentityMatrix())

    # propagate internal
    f_output, f_jacobian = forward_jacobian!(f)

    # update output from internal output
    @turbo for k1 in 1:dim_out
        output_layer[k1] = f_output[k1]
    end
    @turbo for k in 1:dim_in
        output_layer[k+dim_out] = input_layer[k]
    end 

    # update jacobian from internal jacobian
    for k = 1:dim_in
        jacobian_layer.θ[k] = f_jacobian[k]
    end
    for k = dim_in+1:layer.dim_out
        jacobian_layer.θ[k] = 0.0
    end

    # calculate jacobian at output
    jacobian_output_layer = custom_mul!(getmatjacobianoutput(layer), jacobian_layer, getmatjacobianinput(layer))

    # return output and jacobian
    return output_layer, jacobian_output_layer

end

# todo: create specialized companion matrix jacobian when shift = 1
function jacobian(layer::ARLayer, input::Vector{<:Real})

    # fetch information
    len = length(input)
    f = layer.f

    # initialize jacobian
    # J = zeros(T, len, len)

    # set identity diagonal
    # @inbounds for k = 1:len-f.dim_out
    #     J[k+f.dim_out,k] = 1
    # end

    # fetch jacobian of internal function
    J_internal = jacobian(f, input[1:f.dim_in])

    # update jacobian of layer with internal jacobian
    # J[1:f.dim_out, 1:f.dim_in] .= J_internal
    J = CompanionMatrix(vcat(J_internal[1:f.dim_in], 0))

    # return jacobian
    return J

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

function deploy(layer::ARLayer; jacobian_start=IdentityMatrix())

    jacobian_layer = jacobian(layer, randn(layer.dim_in))
    jacobian_layer_out = jacobian_layer * jacobian_start

    return ARLayer(
        layer.dim_in,
        layer.dim_out,
        deploy(layer.f; jacobian_start=IdentityMatrix()),
        DeployMemory(
            randn(layer.dim_in),
            randn(layer.dim_out),
            jacobian_layer,
            jacobian_start,
            jacobian_layer_out
        )
    )
end

function print_info(layer::ARLayer, level::Int, io)

    # print layer
    write(io, string(["--" for _=1:level]..., " ARLayer(", layer.dim_in, ")\n"))

    # print internal model
    print_info(layer.f, level+1, io)

end