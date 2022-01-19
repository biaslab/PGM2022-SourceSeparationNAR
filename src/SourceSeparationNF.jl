module SourceSeparationNF

    include("algebra/permutation_matrix.jl")
    
    include("optimizers/optimizers.jl")

    include("initializers/initializers.jl")

    include("models/model.jl")

    include("losses/losses.jl")

    include("train.jl")

    include("data.jl")

end