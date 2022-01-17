import Base: +, -, *, /, ==, >=, <=, >, <

mutable struct Parameter{T, O <: AbstractOptimizer}
    value      :: T
    gradient   :: T
    optimizer  :: O
    batch_size :: Int64
    it         :: Int64
end

Parameter(value::T; batch_size::Int64=16) where { T <: Real } = Parameter(value, zero(T), Adam(), batch_size, 0)
function Parameter(value::AbstractVector{T}; batch_size::Int64=16) where { T <: Real }
    dim = length(value)
    return Parameter(value, zeros(T, dim), Adam(dim), batch_size, 0)
end
function Parameter(value::AbstractMatrix{T}; batch_size::Int64=16) where { T <: Real }
    sz = size(value)
    return Parameter(value, zeros(T, sz), Adam(sz), batch_size, 0)
end

# arithmetic operators
(+)(x::Parameter, y) = x.value + y
(+)(x, y::Parameter) = x + y.value
(-)(x::Parameter, y) = x.value - y
(-)(x, y::Parameter) = x - y.value
(*)(x::Parameter, y) = x.value * y
(*)(x, y::Parameter) = x * y.value
(/)(x::Parameter, y) = x.value / y
(/)(x, y::Parameter) = x / y.value

# conditional operators
(==)(x::Parameter, y) = x.value == y
(==)(x, y::Parameter) = x == y.value
(>=)(x::Parameter, y) = x.value >= y
(>=)(x, y::Parameter) = x >= y.value
(<=)(x::Parameter, y) = x.value <= y
(<=)(x, y::Parameter) = x <= y.value
(>)(x::Parameter, y) = x.value > y
(>)(x, y::Parameter) = x > y.value
(<)(x::Parameter, y) = x.value < y
(<)(x, y::Parameter) = x < y.value

function update!(x::Parameter{<:Real, <:AbstractOptimizer})

    # update batch counter
    x.it += 1
    batch_size = x.batch_size

    # if batch size is reached
    if x.it == batch_size

        # normalize gradient
        x.gradient /= batch_size

        # update value
        x.value = update!(x.value, x.optimizer, x.gradient)

        # reset gradient and batch counter
        x.gradient = 0.0
        x.it = 0

    end

end
function update!(x::Parameter)

    # update batch counter
    x.it += 1
    batch_size = x.batch_size

    # if batch size is reached
    if x.it == batch_size

        # fetch parameters
        gradient = x.gradient
        len = length(gradient)

        # normalize gradient
        @inbounds for k in 1:len
            gradient[k] /= batch_size
        end

        # update value
        update!(x.value, x.optimizer, x.gradient)

        # reset gradient and batch_counter
        @inbounds for k in 1:len
            gradient[k] = 0.0
        end
        x.it = 0

    end

end

function setlr!(x::Parameter, lr)
    x.optimizer.λ = lr
end