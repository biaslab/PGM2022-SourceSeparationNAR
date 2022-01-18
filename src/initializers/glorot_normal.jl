import Random: rand

struct GlorotNormal <: AbstractInitializer
    dim_in  :: Int64
    dim_out :: Int64
end

function rand(init::GlorotNormal, dims::Tuple{Vararg{Int64, N}}) where { N }
    correction_factor = sqrt(2 / (init.dim_in + init.dim_out))
    return correction_factor .* map(x -> sample(init), zeros(dims))
end
function rand(init::GlorotNormal, dim::Int64)
    correction_factor = sqrt(2 / (init.dim_in + init.dim_out))
    return correction_factor .* map(x -> sample(init), zeros(dim))
end
rand(init::GlorotNormal) = return sqrt(2 / (init.dim_in + init.dim_out)) * sample(init)

function sample(init::GlorotNormal)
    x = randn()
    if abs(x) > 2.0
        x = sample(init)
    end
    return x
end