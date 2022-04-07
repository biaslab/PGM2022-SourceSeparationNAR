module SourceSeparationINN

    include("algebra/cholesky.jl")
    include("algebra/identity_matrix.jl")
    include("algebra/companion_matrix.jl")
    include("algebra/custom_mul.jl")
    include("algebra/logsumexp.jl")
    include("algebra/permutation_matrix.jl")
    include("algebra/svd.jl")
    
    include("optimizers/optimizers.jl")

    include("initializers/initializers.jl")

    include("models/model.jl")

    include("losses/losses.jl")
    
    include("data.jl")

    include("ekf.jl")
    
    include("run.jl")

    include("log.jl")

    include("ReactiveMP.jl")
    
end