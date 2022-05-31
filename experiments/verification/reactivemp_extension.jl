# this file extends upon the basic functionality of ReactiveMP.
export Nar, NarMeta

using SourceSeparationNAR: fastcholesky

# create Nar node
struct Nar end
@node Nar Deterministic [ out, in ]

# create Nar meta data
struct NarMeta{T <: SourceSeparationNAR.Model, A <: ReactiveMP.AbstractNonLinearApproximation}
    model         :: T
    approximation :: A
end

# specify rule towards the input of the layer
@rule Nar(:in, Marginalisation) (m_out::Any, meta::NarMeta) = begin

    # return distribution
    return missing

end

# specify rule towards the output of the layer (linearization)
@rule Nar(:out, Marginalisation) (m_in::MultivariateNormalDistributionsFamily, meta::NarMeta{M,Linearization}) where { M } = begin
    
    # get statistics of incoming message
    μ_in, Σ_in = mean_cov(m_in)

    # fetch model
    model = meta.model

    # run models forward and compute jacobian
    μ_out, J = forward_jacobian!(model, μ_in)

    # compute output covariance
    Σ_out = J * Σ_in * J'

    # return message
    return MvNormalMeanCovariance(μ_out, Σ_out)

end

@rule Nar(:out, Marginalisation) (m_in::MultivariateNormalDistributionsFamily, meta::NarMeta{M,Unscented}) where { M } = begin
    
    # get statistics of incoming message
    μ_in, Σ_in = mean_cov(m_in)

    # extract model
    model = meta.model

    # extract parameters of unscented transform
    approximation = meta.approximation
    λ  = approximation.λ
    L  = approximation.L
    Wm = approximation.Wm
    Wc = approximation.Wc

    # calculate sigma points/vectors    
    sqrtΣ = sqrt((L + λ)*Σ_in)
    # sqrtΣ = sqrt(L + λ) * fastcholesky(Σ_in).L
    χ = Vector{Vector{Float64}}(undef, 2*L + 1)
    for k = 1:length(χ)
        χ[k] = copy(μ_in)
    end
    for l = 2:L+1
        χ[l]     .+= sqrtΣ[l-1,:]
        χ[L + l] .-= sqrtΣ[l-1,:]
    end
 
    # transform sigma points
    Y = [ forward(model, x) for x in χ ]

    # calculate new parameters
    μ_out = zeros(L)
    Σ_out = zeros(L, L)
    for k = 1:2*L+1
        μ_out .+= Wm[k] .* Y[k]
    end
    for k = 1:2*L+1
        Σ_out .+= Wc[k] .* ( Y[k] - μ_out ) *  ( Y[k] - μ_out )'
    end

    # return message
    return MvNormalMeanCovariance(μ_out, Σ_out)

end

struct SphericalSimplex <: ReactiveMP.AbstractNonLinearApproximation 
    dim     :: Int64
    scale   :: Float64
    W0      :: Float64
    Wi      :: Float64
    W       :: Vector{Float64}
    X       :: Matrix{Float64}
    Z       :: Matrix{Float64}
    Y       :: Matrix{Float64}
end

function SphericalSimplex(dim::Int64, W0::Float64; scale::Real=1.0)

    # compute weight sequence
    Wi = (1.0 - W0) / (dim + 1)
    W = vcat(W0, Wi*ones(dim+1))

    # initiliaze matrix and vector sequence
    X = zeros(dim, dim + 2)
    X[1,1] = 0.0
    X[1,2] = -1 / sqrt(2*Wi)
    X[1,3] = 1 / sqrt(2*Wi)

    # expand vector sequence
    for j in 2:dim
        for i in 0:j+1
            if i == 0
                # do nothing as everything is initilialized with zeros
            elseif 1 ≤ i ≤ j
                X[j,i+1] = -1 / sqrt(j*(j+1)*Wi)
            elseif i == j + 1
                X[j,i+1] = j / sqrt(j*(j+1)*Wi)
            end
        end
    end

    # apply scaling
    X .*= scale         # no mean compensation required as X is zero-mean and will be transformed during inference
    W[1]      = W[1]/scale^2 + (1 - 1/scale^2)
    W[2:end] ./= scale^2 

    # return structure
    return SphericalSimplex(dim, float(scale), W0, Wi, W, X, zeros(dim, dim+2), zeros(dim, dim+2))

end

@rule Nar(:out, Marginalisation) (m_in::MultivariateNormalDistributionsFamily, meta::NarMeta{M,SphericalSimplex}) where { M } = begin
    
    # get statistics of incoming message
    μ_in, Σ_in = mean_cov(m_in)

    # extract model
    model = meta.model

    # extract parameters of unscented transform
    approximation = meta.approximation
    W = approximation.W
    X = approximation.X
    Z = approximation.Z
    Y = approximation.Y
    dim = approximation.dim

    # calculate sigma points
    sqrtΣ = fastcholesky(Σ_in).L
    SourceSeparationNAR.custom_mulp!(Z, sqrtΣ, X, μ_in)
 
    # transform sigma points
    for i in 1:dim+2

        # manually copy data
        for k in 1:dim
            model.memory.input[k] = Z[k,i]
        end

        # run model forward
        forward!(model)

        # extract data
        for k in 1:dim
            Y[k,i] = model.memory.output[k]
        end

    end

    # calculate new parameters
    μ_out = zeros(dim)
    Σ_out = zeros(dim, dim)
    for i = 1:dim+2
        for k = 1:dim
            μ_out[k] += W[i] * Y[k,i]
        end
    end
    for i = 1:dim+2
        for k1 = 1:dim
            for k2 = 1:dim
                Σ_out[k1,k2] += W[i] * ( Y[k1,i] - μ_out[k1] ) * ( Y[k2,i] - μ_out[k2] )
            end
        end
    end
    # γ = approximation.W0 + 3 - approximation.scale^2
    for k1 = 1:dim
        for k2 = 1:dim
            Σ_out[k1,k2] += (1 - approximation.scale^2) * ( Y[k1,1] - μ_out[k1] ) * ( Y[k2,1] - μ_out[k2] )
        end
    end  

    # return message
    return MvNormalMeanCovariance(μ_out, Σ_out)

end


# specify node for equalityMultiply
struct EqualityMultiply end
@node EqualityMultiply Deterministic [y, x, z]

# create EqualityMultiply meta data
struct EqualityMultiplyMeta{T}
    A :: T
end

# specify rule towards y
@rule EqualityMultiply(:y, Marginalisation) (m_x::MultivariateNormalDistributionsFamily, m_z::Missing, meta::EqualityMultiplyMeta{<:Vector}) = begin

    # fetch multiplier
    A = meta.A

    # fetch statistics of incoming message
    μ_x, Σ_x = mean_cov(m_x)

    # calculate output statistics
    μ_y = dot(A, μ_x)
    Σ_y = dot(A, Σ_x, A)

    # return message
    return NormalMeanVariance(μ_y, Σ_y)

end

# specify rule towards x
@rule EqualityMultiply(:x, Marginalisation) (m_y::UnivariateNormalDistributionsFamily, m_z::Missing, meta::EqualityMultiplyMeta{<:Vector}) = begin

    # fetch multiplier
    A = meta.A

    # fetch statistics of incoming message
    ξ_y, Λ_y = weightedmean_precision(m_y)

    # calculate output statistics
    ξ_x = A * ξ_y
    Λ_x = Λ_y * A * A'

    # return message
    return MvNormalWeightedMeanPrecision(ξ_x, Λ_x)

end

# specify rule towards z
@rule EqualityMultiply(:z, Marginalisation) (m_y::UnivariateNormalDistributionsFamily, m_x::MultivariateNormalDistributionsFamily, meta::EqualityMultiplyMeta{<:Vector}) = begin

    # fetch multiplier
    A = meta.A

    # fetch statistics of incoming messages
    μ_y, Σ_y = mean_cov(m_y)
    μ_x, Σ_x = mean_cov(m_x)

    # calculate output statistics
    G = 1 / (Σ_y + dot(A, Σ_x, A))
    K = Σ_x * A * G
    μ_z = μ_x + K * (μ_y - dot(A, μ_x)) 
    Σ_z = (I - K * A') * Σ_x


    # return message
    return MvNormalMeanCovariance(μ_z, Σ_z)

end
