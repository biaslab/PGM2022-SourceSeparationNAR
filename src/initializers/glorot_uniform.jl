import Random: rand

struct GlorotUniform <: AbstractInitializer
    dim_in  :: Int64
    dim_out :: Int64
end

function rand(init::GlorotUniform, dims::Tuple{Vararg{Int64, N}}) where { N }
    limit = 2*sqrt(6 / (init.dim_in + init.dim_out))
    return (rand(Float64, dims) .- 0.5) .* limit 
end
function rand(init::GlorotUniform, dim::Int64)
    limit = 2*sqrt(6 / (init.dim_in + init.dim_out))
    return (rand(Float64, dim) .- 0.5) .* limit 
end
rand(init::GlorotUniform) = return (rand(Float64) - 0.5) * 2 * sqrt(3)