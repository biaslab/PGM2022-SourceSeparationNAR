using StatsFuns: softmax!

import Base: length, size, getindex, setindex!

mutable struct SoftmaxOutput{T <: Real} <: AbstractMatrix{T}
    mat :: Matrix{T}
end
Base.length(x::SoftmaxOutput) = length(x.mat)
Base.size(x::SoftmaxOutput) = size(x.mat)
Base.getindex(x::SoftmaxOutput, i, ii) = x.mat[i,ii]
Base.setindex!(x::SoftmaxOutput, y, i, ii) = (x.mat[i,ii] = y)
getmat(A::SoftmaxOutput) = A.mat

mutable struct SoftmaxGradientOutput{T <: Real} <: AbstractMatrix{T}
    mat :: Matrix{T}
end
Base.length(x::SoftmaxGradientOutput) = length(x.mat)
Base.size(x::SoftmaxGradientOutput) = size(x.mat)
Base.getindex(x::SoftmaxGradientOutput, i, ii) = x.mat[i,ii]
Base.setindex!(x::SoftmaxGradientOutput, y, i, ii) = (x.mat[i,ii] = y)
getmat(A::SoftmaxGradientOutput) = A.mat

mutable struct SoftmaxLayer{M <: Union{Nothing,Memory}} <: AbstractLayer
    dim_in          :: Int64
    dim_out         :: Int64
    memory          :: M
end
function SoftmaxLayer(dim; batch_size::Int64=128)
    return SoftmaxLayer(dim, dim, Memory(zeros(dim,batch_size), SoftmaxOutput(zeros(dim,batch_size)), zeros(dim,batch_size), SoftmaxGradientOutput(zeros(dim,batch_size))))
end

function setoutput!(f::SoftmaxLayer, output)
    f.memory.output.mat = output
end
function setgradientoutput!(f::SoftmaxLayer, gradient_output)
    f.memory.gradient_output.mat = gradient_output
end

function forward(layer::SoftmaxLayer{Nothing}, input)
    
    # create output in layer
    output = similar(input)

    # update output of layer
    @turbo output .= input
    softmax!(output) # for numerical stability use custom vector type instead.

    # return output 
    return output
    
end

function forward!(layer::SoftmaxLayer{<:Memory}) 
    
    # fetch input and output in layer
    input  = getmatinput(layer)
    output = getmatoutput(layer)

    # update output of layer
    @turbo output .= input
    # softmax!(output) # for numerical stability use custom vector type instead.

    # return output 
    return output
    
end

function propagate_error!(layer::SoftmaxLayer{<:Memory}) 
    
    # fetch input and output gradients in layer
    input           = getmatinput(layer)
    output          = getmatoutput(layer)
    gradient_input  = getmatgradientinput(layer)
    gradient_output = getmatgradientoutput(layer)
    (ax1, ax2) = axes(input)

    # update input gradient of layer
    @turbo for k2 in ax2
        Z = logsumexp_column(output, k2)
        for k1 in ax1
            gradient_input[k1,k2] = exp(output[k1,k2] - Z) + gradient_output[k1,k2] # compensate for the factor output[k] with loss function
        end
    end

    # return gradient input 
    return gradient_input
    
end

update!(::SoftmaxLayer{<:Memory}) = return

setlr!(::SoftmaxLayer{<:Memory}, lr) = return

isinvertible(::SoftmaxLayer) = false

nr_params(::SoftmaxLayer) = 0

function print_info(layer::SoftmaxLayer, level::Int, io)

    # print layer
    write(io, string(["--" for _=1:level]..., " SoftmaxLayer(", layer.dim_in, ")\n"))

end