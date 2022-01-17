mutable struct GradientDescent{T} <: AbstractOptimizer
    λ    :: Float64
    it   :: Int64
    diff :: T
end
function GradientDescent(;λ::T=1e-8)  where { T <: Real }
    return  GradientDescent(λ, 1, zero(T))
end
function GradientDescent(len::Int; λ::T=1e-8)  where { T <: Real }
    return GradientDescent(λ, 1, zeros(T, len))
end
function GradientDescent(size::Tuple{<:Int}; λ::T=1e-8) where { T <: Real }
    return GradientDescent(λ, 1, zeros(T, size))
end

getall(optimizer::GradientDescent) = return optimizer.λ, optimizer.it, optimizer.diff


function update!(x::T, optimizer::GradientDescent{ T }, ∇::T) where { T <: Real }

    # fetch parameters
    λ, _, diff = getall(optimizer)

    # calculate gradient step
    diff = λ * ∇
    optimizer.diff = diff

    # update iteration count
    optimizer.it   += 1

    # update x and return
    x -= diff
    return x

end

function update!(x::T, optimizer::GradientDescent{ T }, ∇::T) where { T <: AbstractVector }

    # fetch parameters
    λ, _, diff = getall(optimizer)

     @inbounds for k in 1:length(x)

        # perform accelerated gradient step
        diff[k] = λ*∇[k]

        # update x
        x[k] -= diff[k]

    end

    # update iteration count
    optimizer.it   += 1

    # return x
    return x

end

function update!(x::T, optimizer::GradientDescent{ T }, ∇::T) where { T <: AbstractMatrix }

    # fetch parameters
    λ, _, diff = getall(optimizer)

    sz = size(x)

     @inbounds for k2 in 1:sz[2]
         @inbounds for k1 in 1:sz[1]

            # perform accelerated gradient step
            diff[k1,k2] = λ*∇[k2,k1]

            # update x
            x[k1,k2] -= diff[k1,k2]

        end
    end

    # update iteration count
    optimizer.it   += 1

    # return x
    return x

end