export CompanionMatrix, CompanionMatrixTransposed

import Base: *
import LinearAlgebra: transpose, inv

using LinearAlgebra: checksquare
using LoopVectorization: @turbo, @tturbo


"""
    CompanionMatrix
Represents a matrix of the following structure:
θ1 θ2 θ3 ... θn-1 θn
 1  0  0  ...   0  0 
 0  1  0  ...   0  0 
 .  .  .  ...   .  .
 .  .  .  ...   .  .
 0  0  0  ...   0  0
 0  0  0  ...   1  0
"""
struct CompanionMatrix{ R <: Real, T <: AbstractVector{R} } <: AbstractMatrix{R}
    θ :: T
end

Base.eltype(::CompanionMatrix{R}) where R = R
Base.size(cmatrix::CompanionMatrix)       = (length(cmatrix.θ), length(cmatrix.θ)) 
Base.length(cmatrix::CompanionMatrix)     = prod(size(cmatrix))

Base.getindex(cmatrix::CompanionMatrix, i::Int) = getindex(cmatrix, map(r -> r + 1, reverse(divrem(i - 1, first(size(cmatrix)))))...)

function Base.getindex(cmatrix::CompanionMatrix, i::Int, j::Int)
    if i === 1
        return cmatrix.θ[j]
    elseif i === j + 1
        return one(eltype(cmatrix))
    else
        return zero(eltype(cmatrix))
    end
end

struct CompanionMatrixTransposed{ R <: Real, T <: AbstractVector{R} } <: AbstractMatrix{R}
    θ :: T
end

Base.eltype(::CompanionMatrixTransposed{R}) where R = R
Base.size(cmatrix::CompanionMatrixTransposed)       = (length(cmatrix.θ), length(cmatrix.θ)) 
Base.length(cmatrix::CompanionMatrixTransposed)     = prod(size(cmatrix))

Base.getindex(cmatrix::CompanionMatrixTransposed, i::Int) = getindex(cmatrix, map(r -> r + 1, reverse(divrem(i - 1, first(size(cmatrix)))))...)

function Base.getindex(cmatrix::CompanionMatrixTransposed, i::Int, j::Int)
    if j === 1
        return cmatrix.θ[i]
    elseif j === i + 1
        return one(eltype(cmatrix))
    else
        return zero(eltype(cmatrix))
    end
end

as_companion_matrix(θ::T) where { R, T <: AbstractVector{R} } = CompanionMatrix{R, T}(θ)
as_companion_matrix(θ::T) where { T <: Real }                 = θ

LinearAlgebra.transpose(cmatrix::CompanionMatrix)           = CompanionMatrixTransposed(cmatrix.θ)
LinearAlgebra.transpose(cmatrix::CompanionMatrixTransposed) = CompanionMatrix(cmatrix.θ)

LinearAlgebra.adjoint(cmatrix::CompanionMatrix)           = CompanionMatrixTransposed(cmatrix.θ)
LinearAlgebra.adjoint(cmatrix::CompanionMatrixTransposed) = CompanionMatrix(cmatrix.θ)

LinearAlgebra.inv(t::Union{CompanionMatrix, CompanionMatrixTransposed}) = inv(as_matrix(t))

function as_matrix(cmatrix::CompanionMatrix)
    dim     = first(size(cmatrix))
    S       = zeros(dim, dim)
    S[1, :] = cmatrix.θ
    for i in 2:dim
        S[i, i - 1] = one(eltype(cmatrix))
    end
    S
end

function as_matrix(cmatrix::CompanionMatrixTransposed)
    dim     = first(size(cmatrix))
    S       = zeros(dim, dim)
    S[:, 1] = cmatrix.θ
    for i in 2:dim
        S[i - 1, i] = one(eltype(cmatrix))
    end
    S
end

function Base.:*(tm::CompanionMatrix, v::AbstractVector)
    r = similar(v)
    r[1] = tm.θ' * v
    for i in 1:(length(v) - 1)
        r[i + 1] = v[i]
    end
    return r
end

function Base.:*(cM::CompanionMatrix, A::AbstractMatrix)

    # create output
    B = similar(A)

    # perform calculations
    mul!(B, cM, A)

    # return result
    return B

end

function mul!(B::AbstractMatrix, cM::CompanionMatrix, A::AbstractMatrix)

    # fetch values from companion matrix
    θ = cM.θ

    # calculate length
    len = length(θ)

    # asssert sizes
    dim1 = checksquare(A)
    dim2 = checksquare(B)
    @assert dim1 == dim2 == len

    # multiplied entries
    @turbo for l ∈ 1:len
        tmp = zero(Float64)
        for k ∈ axes(A,1)
            tmp += A[k,l] * θ[k]
        end
        B[1,l] = tmp
    end

    # shifted entries
    @turbo for l in 1:len
        for k in 1:len-1
            B[k+1, l] = A[k,l]
        end
    end

end

function Base.:*(A::AbstractMatrix, cM::CompanionMatrixTransposed)

    # create output
    B = similar(A)

    # perform calculations
    mul!(B, A, cM)

    # return result
    return B

end

function mul!(B::AbstractMatrix, A::AbstractMatrix, cM::CompanionMatrixTransposed)

    # fetch values from companion matrix
    θ = cM.θ

    # calculate length
    len = length(θ)

    # asssert sizes
    dim1 = checksquare(A)
    dim2 = checksquare(B)
    @assert dim1 == dim2 == len

    # multiplied entries
    @turbo for l ∈ 1:len
        tmp = zero(Float64)
        for k ∈ axes(A,1)
            tmp += A[l,k] * θ[k]
        end
        B[l,1] = tmp
    end

    # shifted entries
    @turbo for l in 1:len
        for k in 1:len-1
            B[l,k+1] = A[l,k]
        end
    end

end

function Base.:*(A::AbstractMatrix, cM::CompanionMatrix)

    # create output
    B = similar(A)

    # perform calculations
    mul!(B, A, cM)

    # return result
    return B

end

function mul!(B::AbstractMatrix, A::AbstractMatrix, cM::CompanionMatrix)

    # fetch values from companion matrix
    θ = cM.θ

    # calculate length
    len = length(θ)

    # asssert sizes
    dim1 = checksquare(A)
    dim2 = checksquare(B)
    @assert dim1 == dim2 == len

    # perform computations
    @turbo for m in 1:len
        for n in 1:len-1
            B[m,n] = A[m,n+1] + θ[n] * A[m,1]
        end
        B[m,len] = θ[len] * A[m,1]
    end

end

function Base.:*(cM::CompanionMatrixTransposed, A::AbstractMatrix)

    # create output
    B = similar(A)

    # perform calculations
    mul!(B, cM, A)

    # return result
    return B

end

function mul!(B::AbstractMatrix, cM::CompanionMatrixTransposed, A::AbstractMatrix)

    # fetch values from companion matrix
    θ = cM.θ

    # calculate length
    len = length(θ)

    # asssert sizes
    dim1 = checksquare(A)
    dim2 = checksquare(B)
    @assert dim1 == dim2 == len

    # perform computations
    @turbo for m in 1:len
        for n in 1:len-1
            B[n,m] = A[n+1,m] + θ[n] * A[1,m]
        end
        B[len,m] = θ[len] * A[1,m]
    end

end

function tri_matmul!(D::AbstractMatrix, A::CompanionMatrix, B::AbstractMatrix, C::CompanionMatrixTransposed)
    dim = size(D,1)
    for ki = 1:dim-1
        for kii = 1:dim-1
            D[ki+1, kii+1] = B[ki,kii]
        end
    end

    for ki = 2:dim
        D[ki, 1] = 0.0
        for kii = 1:dim
            D[ki, 1] += B[ki-1,kii] * C[kii,1]
        end
    end

    for ki = 2:dim
        D[1,ki] = 0.0
        for kii = 1:dim
            D[1,ki] += B[kii,ki-1] * A[1,kii]
        end
    end

    D[1,1] = 0
    @inbounds for j in 1:dim
        yj = C[j,1]
        if !iszero(yj)
            temp = 0.0
            @simd for i in 1:dim
                temp += adjoint(B[i,j]) * A[1,i]
            end
            D[1,1] += dot(temp, yj)
        end
    end

end

function Base.:*(A::CompanionMatrix, B::AbstractMatrix, C::CompanionMatrixTransposed) 
    D = similar(B)
    tri_matmul!(D, A, B, C)
    return D
end