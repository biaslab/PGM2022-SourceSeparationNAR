@testset "Custom multiplication" begin

    using SourceSeparationNAR: custom_mul, custom_mulp

    @testset "custom_mul(A::AbstractMatrix, x::AbstractVector)" begin
        
        for k = 5:10
            A = randn(rand(2:20),k)
            x = randn(k)
            @test custom_mul(A,x) ≈ A*x
        end

    end

    @testset "custom_mul(A::AbstractMatrix, X::AbstractMatrix)" begin
        
        for k = 5:10
            A = randn(rand(2:20), k)
            X = randn(k, rand(2:20))
            @test custom_mul(A,X) ≈ A*X
        end

    end

    
    @testset "custom_mulp(A::AbstractMatrix, x::AbstractVector, b::AbstractVector)" begin
        
        for k = 5:10
            dim = rand(2:20)
            A = randn(dim,k)
            x = randn(k)
            b = randn(dim)
            @test custom_mulp(A,x,b) ≈ A*x .+ b
        end

    end

    @testset "custom_mulp(A::AbstractMatrix, X::AbstractMatrix, b::AbstractVector)" begin
        
        for k = 5:10
            dim = rand(2:20)
            A = randn(dim,k)
            X = randn(k, rand(2:20))
            b = randn(dim)
            @test custom_mulp(A,X,b) ≈ A*X .+ b
        end

    end

end