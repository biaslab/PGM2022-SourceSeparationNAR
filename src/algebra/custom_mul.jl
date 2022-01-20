using LoopVectorization: @turbo

function custom_mul(A::Matrix{T}, x::Vector{T}) where { T <: Real }

    # allocate new vector
    y = Vector{T}(undef, size(A,1))

    # perform multiplication
    custom_mul!(y, A, x)

    # return output
    return y

end

function custom_mul!(y::Vector{T}, A::AbstractMatrix{T}, x::Vector{T}) where { T <: Real }
    (ax1, ax2) = axes(A)
    @turbo for k1 ∈ ax1
        y[k1] = 0
        for k2 ∈ ax2
            y[k1] += A[k1,k2]*x[k2]
        end
    end
    return y
end

function custom_mulp(A::Matrix{T}, x::Vector{T}, b::Vector{T}) where { T <: Real }

    # allocate new vector
    y = Vector{T}(undef, length(b))

    # perform multiplication
    custom_mulp!(y, A, x, b)

    # return output
    return y

end

function custom_mulp!(y::Vector{T}, A::AbstractMatrix{T}, x::Vector{T}, b::Vector{T}) where { T <: Real }
    (ax1, ax2) = axes(A)
    @turbo for k1 ∈ ax1
        y[k1] = b[k1]
        for k2 ∈ ax2
            y[k1] += A[k1,k2]*x[k2]
        end
    end
    return y
end

function mygemmavx!(C, A, B)
    @turbo for m ∈ axes(A,1), n ∈ axes(B,2)
        Cmn = zero(eltype(C))
        for k ∈ axes(A,2)
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end
