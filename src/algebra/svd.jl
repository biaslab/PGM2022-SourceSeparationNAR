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