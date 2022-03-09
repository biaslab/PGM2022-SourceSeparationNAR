mutable struct SVDSpectralNormal{P, T} 
    A       :: P
    Asn     :: Matrix{T}
    σ       :: Float64
    u       :: Vector{T}
    v       :: Vector{T}
    changed :: Bool
end
function SVDSpectralNormal(A, C)
    Asn = copy(A.value)

    # perform truncated svd
    σ, u, v = fast_tsvd(A.value)

    # normalize matrix
    Asn ./= (σ / C)

    return SVDSpectralNormal(A, Asn, σ, u, v, false)
end
function normalize!(obj::SVDSpectralNormal{P,T}, C) where { P, T }
    if obj.changed
        # copy current matrix
        obj.Asn .= obj.A.value

        # perform truncated svd
        σ, u, v = fast_tsvd(obj.A.value)

        # normalize matrix
        obj.Asn ./= (σ / C)

        # save intermediate outputs
        obj.changed = false
        obj.σ       = σ
        obj.u       = u
        obj.v       = v
    end
    return obj.Asn
end
function update!(obj::SVDSpectralNormal)
    update!(obj.A)
    obj.changed = true
end
setlr!(obj::SVDSpectralNormal, lr)= setlr!(obj.A, lr)
Base.length(obj::SVDSpectralNormal) = length(obj.A)

function fast_tsvd(A; epsilon=1e-10)

    n, m = size(A)

    if n > m

        v = svd_dominant_eigen(A, epsilon=epsilon)
        u = custom_mul(A, v)
        sigma = norm(u)
        u ./= sigma

    else

        u = svd_dominant_eigen(A, epsilon=epsilon)
        v = custom_mul(A', u)
        sigma = norm(v)
        v ./= sigma

    end

    return sigma, u, v

end

function svd_dominant_eigen(A; epsilon=1e-10)
    n, m = size(A)
    current_v = randn(min(n, m))
    current_v ./= norm(current_v)
    last_v = randn(min(n, m))
    last_v ./= norm(last_v)

    if n > m
        B = custom_mul(A', A)
    else
        B = custom_mul(A, A')
    end

    while abs(dot(current_v, last_v)) < 1 - epsilon
        last_v .= current_v
        custom_mul!(current_v, B, last_v)
        current_v ./= norm(current_v)
    end

    return current_v

end