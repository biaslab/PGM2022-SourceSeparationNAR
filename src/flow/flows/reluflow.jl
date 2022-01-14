mutable struct ReluFlow{T <: Real, O<:AbstractOptimizer} <: AbstractFlow
    l :: T
    a :: Parameter{T,O}
    b :: Parameter{T,O}
    c :: Parameter{T,O}
    input           :: T
    output          :: T
    gradient_input  :: T
    gradient_output :: T
end

function ReluFlow(l::T, a::T, b::T, c::T) where { T <: Real }
    return ReluFlow(l, Parameter(a), Parameter(b), Parameter(c), randn(4)...)
end
function ReluFlow(l::T) where { T <: Real }
    return ReluFlow(l, randn(3)...)
end

getall(f::ReluFlow) = return f.l, f.a, f.b, f.c

function forward!(f::ReluFlow, x::T) where { T <: Real }

    # set input of layer
    f.input = x

    # calculate output of layer
    l, a, b, c = getall(f)
    z = x + c.value
    output = a*max(l*z, z) + b

    # set output of layer
    f.output = output

    # return output of layer 
    return output
    
end

function propagate_error!(f::ReluFlow, ∂L_∂y::T) where { T <: Real }

    # set gradient at output of layer
    f.gradient_output = ∂L_∂y

    # propagate gradient to input of layer
    l, a, b, c = getall(f)
    z = f.input + b
    if z >= 0
        ∂L_∂x = a * ∂L_∂y
    else
        ∂L_∂x = a * l * ∂L_∂y
    end
    f.gradient_input = ∂L_∂x
    
    # propagate gradient to the parameters
    a.gradient = ∂L_∂y*max(l*z, z)
    b.gradient = ∂L_∂x
    c.gradient = ∂L_∂y

    # return gradient at input of layer
    return ∂L_∂x

end

function update!(f::ReluFlow)
    update!(f.a)
    update!(f.b)
    update!(f.c)
end