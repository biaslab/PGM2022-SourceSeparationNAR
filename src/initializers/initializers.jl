export AbstractInitializer
export GlorotNormal, GlorotUniform, HeNormal, HeUniform, Ones, RandomNormal, RandomUniform, TruncatedNormal, Zeros

abstract type AbstractInitializer end

include("glorot_normal.jl")
include("glorot_uniform.jl")
include("he_normal.jl")
include("he_uniform.jl")
include("ones.jl")
include("random_normal.jl")
include("random_uniform.jl")
include("truncated_normal.jl")
include("zeros.jl")