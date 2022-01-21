using LoopVectorization: @turbo

function custom_mul(A::AbstractMatrix{T}, x::AbstractVector{T}) where { T <: Real }

    # allocate new vector
    y = Vector{T}(undef, size(A,1))

    # perform multiplication
    custom_mul!(y, A, x)

    # return output
    return y

end

function custom_mul(A::AbstractMatrix{T}, X::AbstractMatrix{T}) where { T <: Real }

    # allocate new vector
    Y = Matrix{T}(undef, (size(A,1), size(X,2)))

    # perform multiplication
    custom_mul!(Y, A, X)

    # return output
    return Y

end

function custom_mul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::Vector{T}) where { T <: Real }
    (ax1, ax2) = axes(A)
    @turbo for k1 ∈ ax1
        yk1 = zero(T)
        for k2 ∈ ax2
            yk1 += A[k1,k2]*x[k2]
        end
        y[k1] = yk1
    end
    return y
end

function custom_mul!(Y::AbstractMatrix{T}, A::AbstractMatrix{T}, X::AbstractMatrix{T}) where { T <: Real }
    @turbo for m ∈ axes(A,1), n ∈ axes(X,2)
        Ymn = zero(T)
        for k ∈ axes(A,2)
            Ymn += A[m,k] * X[k,n]
        end
        Y[m,n] = Ymn
    end
    return Y
end

function custom_mulp(A::AbstractMatrix{T}, x::AbstractVector{T}, b::AbstractVector{T}) where { T <: Real }

    # allocate new vector
    y = Vector{T}(undef, length(b))

    # perform multiplication
    custom_mulp!(y, A, x, b)

    # return output
    return y

end

function custom_mulp(A::AbstractMatrix{T}, X::AbstractMatrix{T}, b::AbstractVector{T}) where { T <: Real }

    # allocate new vector
    Y = Matrix{T}(undef, (length(b), size(X,2)))

    # perform multiplication
    custom_mulp!(Y, A, X, b)

    # return output
    return Y

end

function custom_mulp!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, b::AbstractVector{T}) where { T <: Real }
    (ax1, ax2) = axes(A)
    @turbo for k1 ∈ ax1
        yk1 = zero(T)
        yk1 += b[k1]
        for k2 ∈ ax2
            yk1 += A[k1,k2]*x[k2]
        end
        y[k1] = yk1
    end
    return y
end

function custom_mulp!(Y::AbstractMatrix{T}, A::AbstractMatrix{T}, X::AbstractMatrix{T}, b::AbstractVector{T}) where { T <: Real }
    @turbo for m ∈ axes(A,1), n ∈ axes(X,2)
        Ymn = zero(T)
        Ymn += b[m]   # do not change with Ymn = b[m]!
        for k ∈ axes(A,2)
            Ymn += A[m,k] * X[k,n]
        end
        Y[m,n] = Ymn
    end
    return Y
end