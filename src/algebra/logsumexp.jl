using LogExpFunctions
using StatsFuns

function logsumexp_column(X::AbstractMatrix, idx::Int)
    # Do not use log(zero(eltype(X))) directly to avoid issues with ForwardDiff (#82)
    FT = float(eltype(X))
    xmax_r = (FT(-Inf), zero(FT))
    for k in 1:size(X,1)
        xmax_r = LogExpFunctions._logsumexp_onepass_op(X[k,idx], xmax_r)
    end
    # xmax_r = reduce(_logsumexp_onepass_op, X; dims=dims, init=(FT(-Inf), zero(FT)))
    return first(xmax_r) + log1p(last(xmax_r))
end

function logsumexp_row(X::AbstractMatrix, idx::Int)
    # Do not use log(zero(eltype(X))) directly to avoid issues with ForwardDiff (#82)
    FT = float(eltype(X))
    xmax_r = (FT(-Inf), zero(FT))
    for k in 1:size(X,2)
        xmax_r = LogExpFunctions._logsumexp_onepass_op(X[idx,k], xmax_r)
    end
    # xmax_r = reduce(_logsumexp_onepass_op, X; dims=dims, init=(FT(-Inf), zero(FT)))
    return first(xmax_r) + log1p(last(xmax_r))
end