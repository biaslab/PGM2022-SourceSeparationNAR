import Random: rand

struct RandomNormal <: AbstractInitializer
    μ :: Float64
    σ :: Float64
end

function rand(init::RandomNormal, dims::Tuple{Vararg{Int64, N}}) where { N }
    return init.μ .+ init.σ .* randn(dims) 
end
function rand(init::RandomNormal, dim::Int64)
    return init.μ .+ init.σ .* randn(dim) 
end
rand(init::RandomNormal) = return init.μ + init.σ * randn()