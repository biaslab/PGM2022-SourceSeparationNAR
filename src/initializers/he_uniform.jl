import Random: rand

struct HeUniform <: AbstractInitializer
    dim_in :: Int64
end

function rand(init::HeUniform, dims::Tuple{Vararg{Int64, N}}) where { N }
    limit = sqrt(6 / init.dim_in)
    return (2 * limit) .* (rand(dims) .- 0.5)
end
function rand(init::HeUniform, dim::Int64)
    limit = sqrt(6 / init.dim_in)
    return (2 * limit) .* (rand(dim) .- 0.5)
end
function rand(init::HeUniform)
    limit = sqrt(6 / init.dim_in)
    return (2 * limit) * (rand() - 0.5)
end