import Base: +, -, *, /, ==, >=, <=, >, <

mutable struct Parameter{T, O <: AbstractOptimizer}
    value     :: T
    gradient  :: T
    optimizer :: O
end

Parameter(value::T) where { T <: Real } = Parameter(value, zero(T), Adam())
function Parameter(value::AbstractVector{T}) where { T <: Real }
    dim = length(value)
    return Parameter(value, zeros(T, dim), Adam(dim))
end
function Parameter(value::AbstractMatrix{T}) where { T <: Real }
    sz = size(value)
    return Parameter(value, zeros(T, sz), Adam(sz))
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
    x.value = update!(x.value, x.optimizer, x.gradient)
end
function update!(x::Parameter)
    update!(x.value, x.optimizer, x.gradient)
end