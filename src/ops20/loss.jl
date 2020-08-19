export accuracy, bce, logistic, nll, zeroone


"""
    nll(scores, labels; dims=1, average=true)

Return the negative log likelihood for a single batch of data given an unnormalized `scores`
matrix and an `Integer` array of correct `labels`. The `scores` matrix should have size
`(classes,instances)` if `dims=1` or `(instances,classes)` if `dims=2`. `labels[i]` should be
in `1:classes` to indicate the correct class for instance i, or 0 to skip instance i.

The return value is `(total/count)` if `average=true` and `(total,count)` if `average=false`
where `count` is the number of instances not skipped (i.e. `label != 0`) and `total` is their
total negative log likelihood.

## Example

Let's assume that there are three classes (cat, dog, ostrich) and just 2 instances with
the unnormalized score `scores[:,1]` and `scores[:,2]` respectively. The first instance
is actually a cat and the second instance a dog:

    scores = [12.2    0.3;
               2.0   21.5;
               0.0  -21.0]
    labels = [1, 2]
    nll(scores,labels)
    # returns 2.1657e-5

The probabilites are derived from the scores and the negative log-probabilities corresponding
to the labels are averaged:

    probabilites = exp.(scores) ./ sum(exp.(scores),dims=1)
    -(log(probabilites[labels[1],1]) + log(probabilites[labels[2],2]))/2
    # returns 2.1657e-5

"""
function nll(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
    indices = findindices(scores,labels,dims=dims)
    lp = logsoftmax(scores,dims=dims)[indices]
    average ? (-sum(lp) / length(lp)) : (-sum(lp), length(lp))
end


"""
    accuracy(scores, labels; dims=1, average=true)

Given an unnormalized `scores` matrix and an `Integer` array of correct `labels`, return the
ratio of instances where the correct label has the maximum score. `dims=1` means instances are
in columns, `dims=2` means instances are in rows. Use `average=false` to return the pair
(ncorrect,count) instead of the ratio (ncorrect/count). The valid labels should be integers in
the range `1:numclasses`, if `labels[i] == 0`, instance i is skipped.

"""
function accuracy(scores,labels::AbstractArray{<:Integer}; dims=1, average=true)
    indices = findindices(scores,labels,dims=dims)
    ycpu = convert(Array,value(scores))
    (maxval,maxind) = findmax(ycpu,dims=dims)
    maxind = LinearIndices(ycpu)[maxind]
    maxind = vec(maxind)[vec(labels) .!= 0]
    correct = (maxind .== indices)
    average ? (sum(correct) / length(correct)) : (sum(correct), length(correct))
end


"""
    logistic(scores, labels; average=true)

Computes logistic loss given predicted unnormalized scores and answer labels for a binary
prediction task.

    log.(1 .+ exp.(-labels .* scores))

Label values should be {-1,1}. Scores are unrestricted.  The return value is `(total/count)`
if `average=true` and `(total,count)` if `average=false` where `count` is the number of
instances and `total` is their total loss.

See also `bce` which computes the same loss with {0,1} labels.

Reference: https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c
"""
function logistic(scores, labels::AbstractVector{<:Integer}; average=true)
    labels = ((labels .+ 1) .÷ 2) # (-1,1)->(0,1)
    bce(scores,labels; average=average)
end


"""
    bce(scores, labels; average=true)

Computes binary cross entropy loss given predicted unnormalized scores and answer labels for a
binary prediction task. Label values should be in {0,1}. Scores are unrestricted and will be
converted to probabilities using

    probs = 1 ./ (1 .+ exp.(-scores))

The loss calculated is

    -(labels .* log.(probs) .+ (1 .- labels) .* log.(1 .- probs))

The return value is `(total/count)` if `average=true` and `(total,count)` if `average=false`
where `count` is the number of instances and `total` is their total loss.

See also `logistic` which computes the same loss with {-1,1} labels.

Reference: https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c
"""
function bce(scores, labels::AbstractVector{<:Integer}; average=true) 
    labels = oftype(scores,labels)
    l = max.(0, scores) .- labels .* scores .+ log.(1 .+ exp.(-abs.(scores)))
    average ? (sum(l)/length(l)) : (sum(l),length(l))
end


# Indexing help for nll and accuracy

function findindices(scores, labels::AbstractArray{<:Integer}; dims=1)
    ninstances = length(labels)
    nindices = 0
    indices = Vector{Int}(undef,ninstances)
    if dims == 1                   # instances in first dimension
        y1 = size(scores,1)
        y2 = div(length(scores),y1)
        if ninstances != y2; throw(DimensionMismatch()); end
        @inbounds for j=1:ninstances
            if labels[j] == 0; continue; end
            indices[nindices+=1] = (j-1)*y1 + labels[j]
        end
    elseif dims == 2               # instances in last dimension
        y2 = size(scores,ndims(scores))
        y1 = div(length(scores),y2)
        if ninstances != y1; throw(DimensionMismatch()); end
        @inbounds for j=1:ninstances
            if labels[j] == 0; continue; end
            indices[nindices+=1] = (labels[j]-1)*y1 + j
        end
    else
        error("findindices only supports dims = 1 or 2")
    end
    return (nindices == ninstances ? indices : view(indices,1:nindices))
end


"""
    nll(model; data, dims=1, average=true, o...)

Compute the negative log likelihood for a model over a dataset:

    nll(model(inputs; kwargs...), labels; dims) for (inputs,labels) in data

and return `(total/count)` if `average=true` or `(total,count)` if `average=false` where
`count` is the number of instances not skipped (instances with `label==0` are skipped) and
`total` is their total negative log likelihood.

The `model` should be a function returning scores given inputs, and data should be an iterable
of `(inputs,labels)` pairs. The valid labels should be integers in the range `1:numclasses`,
if `labels[i] == 0`, instance i is skipped.

"""
function nll(model; data, dims=1, average=true, o...)
    sum = cnt = 0
    for (x,y) in data
        (z,n) = nll(model(x; o...), y; dims=dims, average=false) 
        sum += z; cnt += n
    end
    average ? sum / cnt : (sum, cnt)
end


"""
    accuracy(model; data, dims=1, average=true, o...)

Compute the number of correct predictions of a model over a dataset:

    accuracy(model(inputs; kwargs...), labels; dims) for (inputs,labels) in data

and return `(ncorrect/count)` if `average=true` or `(ncorrect,count)` if `average=false` where
`count` is the number instances not skipped (instances with `label==0` are skipped) and
`ncorrect` is the number of them correctly labeled by the model.

The `model` should be a function returning scores given inputs, and data should be an iterable
of `(inputs,labels)` pairs. The valid labels should be integers in the range `1:numclasses`,
if `labels[i] == 0`, instance i is skipped.

"""
function accuracy(model; data, dims=1, average=true, o...)
    sum = cnt = 0
    for (x,y) in data
        (z,n) = accuracy(model(x; o...), y; dims=dims, average=false)
        sum += z
        cnt += n
    end
    average ? sum / cnt : (sum, cnt)
end


# DEPRECATE:

# The two-arg model,data calls cause mix-ups with the scores,labels calls:
function nll(model,data; o...)
    @warn "nll(model,data; o...) is deprecated, please use nll(model; data=data, o...)" maxlog=1
    nll(model; data=data, o...)
end

function accuracy(model,data; o...)
    @warn "accuracy(model,data; o...) is deprecated, please use accuracy(model; data=data, o...)" maxlog=1
    accuracy(model; data=data, o...)
end

# We need the (model,x,y) interface to implement regularization:
function nll(f, x, y; dims=1, average=true, o...)
    @warn "nll(f,x,y; o...) is deprecated, please use nll(f(x),y; o...) instead." maxlog=1
    nll(f(x; o...), y; dims=dims, average=average)
end

function accuracy(f, x, y; dims=1, average=true, o...)
    @warn "accuracy(f,x,y; o...) is deprecated, please use accuracy(f(x),y; o...) instead." maxlog=1
    accuracy(f(x; o...), y; dims=dims, average=average)
end

# We need the (weights,data,predict) interface to support the old interface:
function nll(w, data, f::Function; dims=1, average=true, o...)
    @warn "nll(weights,data,func; o...) is deprecated, please use nll(x->func(weights,x); data=data, o...) instead." maxlog=1
    nll(x->f(w,x;o...), data; dims=dims, average=average)
end

function accuracy(w, data, f::Function; dims=1, average=true, o...)
    @warn "accuracy(weights,data,func; o...) is deprecated, please use accuracy(x->func(weights,x); data=data, o...) instead." maxlog=1
    accuracy(x->f(w,x;o...), data; dims=dims, average=average)
end

"zeroone loss is equal to 1 - accuracy"
function zeroone(x...; o...)
    @warn "zeroone() is deprecated, please use 1-accuracy()" maxlog=1
    1 - accuracy(x...; o...)
end

