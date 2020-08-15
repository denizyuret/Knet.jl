import Knet.Ops20: nll, accuracy
#include("data.jl") ## Data

"""
    nll(model, data; dims=1, average=true, o...)

Compute the negative log likelihood for a model over a dataset:

    nll(model(inputs; kwargs...), labels; dims) for (inputs,labels) in data

and return `(total/count)` if `average=true` or `(total,count)` if `average=false` where
`count` is the number of instances not skipped (instances with `label==0` are skipped) and
`total` is their total negative log likelihood.

The `model` should be a function returning scores given inputs, and data should be an iterable
of `(inputs,labels)` pairs. The valid labels should be integers in the range `1:numclasses`,
if `labels[i] == 0`, instance i is skipped.

"""
function nll(model, data::Data; dims=1, average=true, o...)
    sum = cnt = 0
    for (x,y) in data
        (z,n) = nll(model(x; o...), y; dims=dims, average=false) 
        sum += z; cnt += n
    end
    average ? sum / cnt : (sum, cnt)
end


"""
    accuracy(model, data; dims=1, average=true, o...)

Compute the number of correct predictions of a model over a dataset:

    accuracy(model(inputs; kwargs...), labels; dims) for (inputs,labels) in data

and return `(ncorrect/count)` if `average=true` or `(ncorrect,count)` if `average=false` where
`count` is the number instances not skipped (instances with `label==0` are skipped) and
`ncorrect` is the number of them correctly labeled by the model.

The `model` should be a function returning scores given inputs, and data should be an iterable
of `(inputs,labels)` pairs. The valid labels should be integers in the range `1:numclasses`,
if `labels[i] == 0`, instance i is skipped.

"""
function accuracy(model, data::Data; dims=1, average=true, o...)
    sum = cnt = 0
    for (x,y) in data
        (z,n) = accuracy(model(x; o...), y; dims=dims, average=false)
        sum += z
        cnt += n
    end
    average ? sum / cnt : (sum, cnt)
end

