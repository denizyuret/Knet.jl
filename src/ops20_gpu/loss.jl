import Knet.Ops20: nll
using Knet.KnetArrays: DevArray

function nll(y,a::DevArray{<:Integer}; dims=1, average=true)
    @warn "nll(scores, answers::$(typeof(a)) is inefficient, nll(scores, answers::Array{<:Integer}) is better." maxlog=1
    nll(y, Array(a); dims=dims, average=average)
end
