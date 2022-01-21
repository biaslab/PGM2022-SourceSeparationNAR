import Base: +, -, *, /, ==, >=, <=, >, <, length, size

mutable struct Parameter{T, O <: AbstractOptimizer}
    value      :: T
    gradient   :: T
    optimizer  :: O
end

Parameter(value::T, optimizer::Type{<:AbstractOptimizer}) where { T <: Real } = Parameter(value, zero(T), optimizer())
function Parameter(value::AbstractVector{T}, optimizer::Type{<:AbstractOptimizer}) where { T <: Real }
    dim = length(value)
    return Parameter(value, zeros(T, dim), optimizer(dim))
end
function Parameter(value::AbstractMatrix{T}, optimizer::Type{<:AbstractOptimizer}) where { T <: Real }
    sz = size(value)
    return Parameter(value, zeros(T, sz), optimizer(sz))
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

update!(x::Parameter) = update!(x.value, x.optimizer, x.gradient)

function setlr!(x::Parameter, lr)
    x.optimizer.λ = lr
end