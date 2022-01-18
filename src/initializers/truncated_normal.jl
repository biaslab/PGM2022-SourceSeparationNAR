import Random: rand

struct TruncatedNormal <: AbstractInitializer
    μ :: Float64
    σ :: Float64
end

function rand(init::TruncatedNormal, dims::Tuple{Vararg{Int64, N}}) where { N }
    return init.μ .+ init.σ .* map(x -> sample(init), zeros(dims))
end
function rand(init::TruncatedNormal, dim::Int64)
    return init.μ .+ init.σ .* map(x -> sample(init), zeros(dim))
end
rand(init::TruncatedNormal) = return init.μ + init.σ * sample(init)

function sample(init::TruncatedNormal)
    x = randn()
    if abs(x) > 2.0
        x = sample(init)
    end
    return x
end