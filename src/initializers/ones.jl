import Random: rand

struct Ones <: AbstractInitializer end

function rand(init::Ones, dims::Tuple{Vararg{Int64, N}}) where { N }
    return ones(dims)
end
function rand(init::Ones, dim::Int64)
    return ones(dim)
end
rand(init::Ones) = return 1.0