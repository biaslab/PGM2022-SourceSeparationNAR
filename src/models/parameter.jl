import Base: +, -, *, /, ==, >=, <=, >, <, length, size

mutable struct Parameter{T, O <: AbstractOptimizer}
    value      :: T
    gradient   :: T
    optimizer  :: O
    batch_size :: Int64
    it         :: Int64
end

Parameter(value::T, optimizer::Type{<:AbstractOptimizer}; batch_size::Int64=128) where { T <: Real } = Parameter(value, zero(T), optimizer(), batch_size, 0)
function Parameter(value::AbstractVector{T}, optimizer::Type{<:AbstractOptimizer}; batch_size::Int64=128) where { T <: Real }
    dim = length(value)
    return Parameter(value, zeros(T, dim), optimizer(dim), batch_size, 0)
end
function Parameter(value::AbstractMatrix{T}, optimizer::Type{<:AbstractOptimizer}; batch_size::Int64=128) where { T <: Real }
    sz = size(value)
    return Parameter(value, zeros(T, sz), optimizer(sz), batch_size, 0)
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

# other operators
length(x::Parameter) = length(x.value)
size(x::Parameter) = size(x.value)

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
function update!(x::Parameter{<:AbstractVector, <:AbstractOptimizer})

    # update batch counter
    x.it += 1
    ibatch_size = 1/x.batch_size

    # if batch size is reached
    if x.it == x.batch_size

        # fetch parameters
        gradient = x.gradient
        len = length(gradient)

        # normalize gradient
        @inbounds for k in 1:len
            gradient[k] *= ibatch_size
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
function update!(x::Parameter{<:AbstractMatrix, <:AbstractOptimizer})

    # update batch counter
    x.it += 1
    ibatch_size = 1/x.batch_size

    # if batch size is reached
    if x.it == x.batch_size

        # fetch parameters
        gradient = x.gradient
        sz = size(gradient)

        # normalize gradient
        @inbounds for k2 in 1:sz[2]
            @inbounds for k1 in 1:sz[1]
                gradient[k1,k2] *= ibatch_size
            end
        end

        # update value
        update!(x.value, x.optimizer, x.gradient)

        # reset gradient and batch_counter
        @inbounds for k2 in 1:sz[2]
            @inbounds for k1 in 1:sz[1]
                gradient[k1,k2] = 0.0
            end
        end
        x.it = 0

    end

end

function setlr!(x::Parameter, lr)
    x.optimizer.λ = lr
end

function setbatchsize!(x::Parameter, batch_size::Int64)
    x.batch_size = batch_size
end