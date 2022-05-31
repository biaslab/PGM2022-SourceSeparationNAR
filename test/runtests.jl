module SourceSeparationNARTest

    using SourceSeparationNAR
    using Test

    @testset "Algebra" begin
        include("algebra/test_custom_mul.jl")
    end

end