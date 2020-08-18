import Knet.Ops20: nll, accuracy
using Knet.KnetArrays: DevArray, KnetArray
using Knet.Ops20: findindices, logsoftmax

## Using GPU arrays for scores is inefficient

function nll(scores, labels::DevArray{<:Integer}; dims=1, average=true)
    @warn "nll(scores, answers::$(typeof(labels)) is inefficient, nll(scores, answers::Array{<:Integer}) is better." maxlog=1
    nll(scores, Array(labels); dims=dims, average=average)
end

function accuracy(scores, labels::DevArray{<:Integer}; dims=1, average=true)
    @warn "accuracy(scores, answers::$(typeof(labels)) is inefficient, nll(scores, answers::Array{<:Integer}) is better." maxlog=1
    accuracy(scores, Array(labels); dims=dims, average=average)
end
