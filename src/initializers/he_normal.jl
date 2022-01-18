import Random: rand

struct HeNormal <: AbstractInitializer
    dim_in :: Int64
end

function rand(init::HeNormal, dims::Tuple{Vararg{Int64, N}}) where { N }
    correction_factor = sqrt(2 / init.dim_in)
    return randn(dims) .* correction_factor 
end
function rand(init::HeNormal, dim::Int64)
    correction_factor = sqrt(2 / init.dim_in)
    return randn(dim) .* correction_factor 
end
rand(init::HeNormal) = return randn() * sqrt(2)