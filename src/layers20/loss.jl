using Statistics
"""
    CrossEntropyLoss(dims=1)
    (l::CrossEntropyLoss)(scores, answers; average=true)
Calculates negative log likelihood error on your predicted scores.
`answers` should be integers corresponding to correct class indices.
If an answer is 0, loss from that answer will not be included.
This is the masking feature when you are working with unequal length sequences.

if dims==1,
    First dimension is assumed to be predicted logits.
* size(scores) = D,[BATCH,T1,T2,...]
* size(answers)= [BATCH,T1,T2,...]

elseif dims==2,
    Last dimension is assumed to be predicted logits.
* size(scores) = [BATCH,T1,T2,...],D
* size(answers)= [BATCH,T1,T2,...]
"""
struct CrossEntropyLoss <: Loss
    dims::Integer
end
CrossEntropyLoss(;dims=1) = CrossEntropyLoss(dims)
@inline (l::CrossEntropyLoss)(y,answers::Array{<:Integer}; average=true) = nllmask(y, answers; dims=l.dims, average=average)

"""
    BCELoss(average=true)
    (l::BCELoss)(scores, answers)
    Computes binary cross entropy given scores(predicted values) and answer labels. answer values should be {0,1}, then it returns negative of
    mean|sum(answers * log(p) + (1-answers)*log(1-p)) where p is equal to 1/(1 + exp.(scores)). see `LogisticLoss`.
"""
struct BCELoss <: Loss end
@inline (l::BCELoss)(y,answers::Array{<:Integer})=bce(y,answers)

"""
    LogisticLoss(average=true)
    (l::LogisticLoss)(scores, answers)
    Computes logistic loss given scores(predicted values) and answer labels. answer values should be {-1,1}, then it returns mean|sum(log(1 +
    exp(-answers*scores))). See also `BCELoss`.
"""
struct LogisticLoss <: Loss end
@inline (l::LogisticLoss)(y,answers::Array{<:Integer}) = logistic(y,answers)


"""
    SigmoidCrossEntropyLoss()
    (l::SigmoidCrossEntropyLoss)(scores, labels)

Measures the error in discrete classification tasks in which each class is independent and not mutually exclusive.
`labels` should be same size with scores where existing classes pointed by one others zero.

if dims==1, First dimension is assumed to be predicted logits.

elseif dims==2, Last dimension is assumed to be predicted logits.
"""
struct SigmoidCrossEntropyLoss <: Loss
    dims::Int
end
SigmoidCrossEntropyLoss(;dims=1) = SigmoidCrossEntropyLoss(dims)
function (l::SigmoidCrossEntropyLoss)(x, z; average=true)
    y = sum(relu.(x) .- x .* z - log.(sigm.(abs.(x))), dims=:)
    average ? y ./ (prod(size(x))/prod(size(x)[l.dims])) : y
end

####
#### Utils
####
@inline function nllmask(y,a::AbstractArray{<:Integer}; dims=1, average=true)
    indices = findindices(y, a, dims=dims)
    lp = logp(y,dims=dims)[indices]
    average ? -mean(lp) : -sum(lp)
end

function findindices(y,a::AbstractArray{<:Integer}; dims=1)
    n       = length(a)
    nonmask = a .> 0
    indices = Vector{Int}(undef,sum(nonmask))
    if dims == 1                   # instances in first dimension
        y1 = size(y,1)
        y2 = div(length(y),y1)
        if n != y2; throw(DimensionMismatch()); end
        k = 1
        @inbounds for (j,v) in enumerate(nonmask)
            !v && continue
            indices[k] = (j-1)*y1 + a[j]
            k += 1
        end
    elseif dims == 2               # instances in last dimension
        y2 = size(y,ndims(y))
        y1 = div(length(y),y2)
        if n != y1; throw(DimensionMismatch()); end
        k = 1
        @inbounds for (j,v) in enumerate(nonmask)
            !v && continue
            indices[k] = (a[j]-1)*y1 + j
            k += 1
        end
    else
        error("findindices only supports dims = 1 or 2")
    end
    return indices
end
