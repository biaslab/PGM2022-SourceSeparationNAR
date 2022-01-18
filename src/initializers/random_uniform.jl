import Random: rand

struct RandomUniform <: AbstractInitializer
    minval :: Float64
    maxval :: Float64
end

function rand(init::RandomUniform, dims::Tuple{Vararg{Int64, N}}) where { N }
    return init.minval .+ (init.maxval - init.minval) .* rand(dims)
end
function rand(init::RandomUniform, dim::Int64)
    return init.minval .+ (init.maxval - init.minval) .* rand(dim)
end
rand(init::RandomUniform) = return init.minval + (init.maxval - init.minval) * rand()