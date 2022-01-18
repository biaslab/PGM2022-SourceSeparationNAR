import Random: rand

struct Zeros <: AbstractInitializer end

function rand(init::Zeros, dims::Tuple{Vararg{Int64, N}}) where { N }
    return zeros(dims)
end
function rand(init::Zeros, dim::Int64)
    return zeros(dim)
end
rand(init::Zeros) = return 0.0