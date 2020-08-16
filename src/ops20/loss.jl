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
function nll(y,a; dims=1, average=true)
    indices = findindices(y,a,dims=dims)
    lp = logsoftmax(y,dims=dims)[indices]
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
function accuracy(y,a; dims=1, average=true)
    indices = findindices(y,a,dims=dims)
    ycpu = convert(Array,value(y))
    (maxval,maxind) = findmax(ycpu,dims=dims)
    maxind = LinearIndices(ycpu)[maxind]
    maxind = vec(maxind)[vec(a) .!= 0]
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
function logistic(z, y; average=true)
    y = ((y .+ 1) .÷ 2) # (-1,1)->(0,1)
    bce(z,y; average=average)
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
function bce(z, y; average=true) 
    y = oftype(z,y)
    l = max.(0, z) .- y .* z .+ log.(1 .+ exp.(-abs.(z)))
    average ? (sum(l)/length(l)) : (sum(l),length(l))
end


# Indexing help for nll and accuracy

function findindices(y,a; dims=1)
    ninstances = length(a)
    nindices = 0
    indices = Vector{Int}(undef,ninstances)
    if dims == 1                   # instances in first dimension
        y1 = size(y,1)
        y2 = div(length(y),y1)
        if ninstances != y2; throw(DimensionMismatch()); end
        @inbounds for j=1:ninstances
            if a[j] == 0; continue; end
            indices[nindices+=1] = (j-1)*y1 + a[j]
        end
    elseif dims == 2               # instances in last dimension
        y2 = size(y,ndims(y))
        y1 = div(length(y),y2)
        if ninstances != y1; throw(DimensionMismatch()); end
        @inbounds for j=1:ninstances
            if a[j] == 0; continue; end
            indices[nindices+=1] = (a[j]-1)*y1 + j
        end
    else
        error("findindices only supports dims = 1 or 2")
    end
    return (nindices == ninstances ? indices : view(indices,1:nindices))
end


# DEPRECATE:

"zeroone loss is equal to 1 - accuracy"
zeroone(x...; o...) = 1 - accuracy(x...; o...)

# We need the (model,x,y) interface to implement regularization:
nll(f, x, y; dims=1, average=true, o...)=nll(f(x; o...), y; dims=dims, average=average)
accuracy(f, x, y; dims=1, average=true, o...)=accuracy(f(x; o...), y; dims=dims, average=average)

# We need the (weights,data,predict) interface to support the old interface:
nll(w, data, f::Function; dims=1, average=true, o...)=nll(x->f(w,x;o...), data; dims=dims, average=average)
accuracy(w, data, f::Function; dims=1, average=true, o...)=accuracy(x->f(w,x;o...), data; dims=dims, average=average)
