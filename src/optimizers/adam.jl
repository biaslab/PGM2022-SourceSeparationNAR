mutable struct Adam{T} <: AbstractOptimizer
    s    :: T
    r    :: T
    shat :: T
    rhat :: T
    ρ1   :: Float64
    ρ2   :: Float64
    itρ1 :: Float64
    itρ2 :: Float64
    λ    :: Float64
    it   :: Int64
    diff :: T
end
function Adam(;ρ1::T=0.9, ρ2::T=0.999, λ::T=1e-4)  where { T <: Real }
    return  Adam(zero(T), zero(T), zero(T), zero(T), ρ1, ρ2, ρ1, ρ2, λ, 1, zero(T))
end
function Adam(len::Int; ρ1::T=0.9, ρ2::T=0.999, λ::T=1e-4)  where { T <: Real }
    return Adam(zeros(T, len), zeros(T, len), zeros(T, len), zeros(T, len), ρ1, ρ2, ρ1, ρ2, λ, 1, zeros(T, len))
end
function Adam(size::Tuple; ρ1::T=0.9, ρ2::T=0.999, λ::T=1e-4) where { T <: Real }
    return Adam(zeros(T, size), zeros(T, size), zeros(T, size), zeros(T, size), ρ1, ρ2, ρ1, ρ2, λ, 1, zeros(T, size))
end

getall(optimizer::Adam) = return optimizer.s, optimizer.r, optimizer.shat, optimizer.rhat, optimizer.ρ1, optimizer.ρ2, optimizer.itρ1, optimizer.itρ2, optimizer.λ, optimizer.it, optimizer.diff

function update!(x::T, optimizer::Adam{ T }, ∇::T) where { T <: Real }

    # fetch parameters
    s, r, _, _, ρ1, ρ2, itρ1, itρ2, λ, _, diff = getall(optimizer)

    # update (biased) first moment
    s *= ρ1
    diff = ∇
    diff *= (1.0-ρ1)
    s += diff
    optimizer.s = s

    # update (unbiased) first moment
    shat = s / (1.0 - itρ1)
    optimizer.shat = shat

    # update (biased) second moment
    r *= ρ2
    diff = ∇
    diff ^= 2
    diff *= (1.0-ρ2)
    r += diff
    optimizer.r = r

    # update (unbiased) second moment
    rhat = r / (1.0 - itρ2)
    optimizer.rhat = rhat

    # perform accelerated gradient step
    rhat = sqrt(rhat)
    rhat += 1e-20
    rhat /= λ
    shat /= rhat
    optimizer.diff = shat

    # update iteration count
    optimizer.it   += 1
    optimizer.itρ1 *= ρ1
    optimizer.itρ2 *= ρ2

    # update x and return
    x -= shat
    return x

end

function update!(x::T, optimizer::Adam{ T }, ∇::T) where { T <: AbstractVector }

    # fetch parameters
    s, r, shat, rhat, ρ1, ρ2, itρ1, itρ2, λ, it, diff = getall(optimizer)
    iρ1  = 1.0 - ρ1
    iρ2  = 1.0 - ρ2
    ditρ1 = 1.0 / (1.0 - itρ1)
    ditρ2 = 1.0 / (1.0 - itρ2)

    @turbo for k in 1:length(x)

        # update (biased) first moment
        s[k] = ρ1*s[k] + iρ1*∇[k]

        # update (unbiased) first moment
        shat[k] = s[k] * ditρ1

        # update (biased) second moment
        r[k] = ρ2*r[k] + iρ2*∇[k]^2

        # update (unbiased) second moment
        rhat[k] = r[k] * ditρ2 

        # perform accelerated gradient step
        diff[k] = λ*shat[k]/(sqrt(rhat[k]) + 1e-20)

        # update x
        x[k] -= diff[k]

    end

    # update iteration count
    optimizer.it   += 1
    optimizer.itρ1 *= ρ1
    optimizer.itρ2 *= ρ2

    # return x
    return x

end

function update!(x::T, optimizer::Adam{ T }, ∇::T) where { T <: AbstractMatrix }

    # fetch parameters
    s, r, shat, rhat, ρ1, ρ2, itρ1, itρ2, λ, it, diff = getall(optimizer)
    iρ1  = 1.0 - ρ1
    ditρ1 = 1.0/(1.0 - itρ1)
    iρ2  = 1.0 - ρ2
    ditρ2 = 1.0/(1.0 - itρ2)

    (ax1, ax2) = axes(x)

    @turbo for k1 in ax1
        for k2 in ax2

            # tmp access
            ∇k1k2 = ∇[k1,k2]

            # update (biased) first moment
            s[k1,k2] = ρ1*s[k1,k2] + iρ1*∇k1k2

            # update (unbiased) first moment
            shat[k1,k2] = s[k1,k2] * ditρ1

            # update (biased) second moment
            r[k1,k2] = ρ2*r[k1,k2] + iρ2*∇k1k2^2

            # update (unbiased) second moment
            rhat[k1,k2] = r[k1,k2] * ditρ2 

            # perform accelerated gradient step
            diff[k1,k2] = λ*shat[k1,k2]/(sqrt(rhat[k1,k2]) + 1e-20)

            # update x
            x[k1,k2] -= diff[k1,k2]

        end
    end

    # update iteration count
    optimizer.it   += 1
    optimizer.itρ1 *= ρ1
    optimizer.itρ2 *= ρ2

    # return x
    return x

end