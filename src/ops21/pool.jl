import NNlib
using AutoGrad
using Statistics: mean
using LinearAlgebra: lmul!
export pool, mean

#= NOTES
- maxpool/meanpool vs op=maximum, op=mean (would require Statistics)
- adaptivexxx vs specifying output=1 or window=Inf.
- torch.nn: MaxPoolNd, MaxUnpoolNd, AvgPoolNd, FractionalMaxPoolNd, LPPoolNd, AdaptiveMaxPoolNd, AdaptiveAvgPoolNd
- torch.nn.functional: avg_poolNd, max_poolNd, max_unpoolNd, lp_poolNd, adaptive_max_poolNd, adaptive_avg_poolNd
- adaptive versions take output_size as argument, regular versions take kernel_size.
- torch also supports dilation? cudnn does not.
- torch supports maxunpool but not avgunpool.
- existence of lp may make the op kwarg more preferrable (other ops in the future?)
- keras has MaxPoolingNd, AveragePoolingNd, GlobalMaxPoolingNd, GlobalAveragePoolingNd: only output=1 supported
- keras GlobalPooling includes reshape.
=#

"""
    pool(x; kwargs...)

Compute pooling of input values (i.e., the maximum or average of several adjacent values) to
produce an output with smaller height and/or width.

If `x` has dimensions `(X1,X2,...,Cx,N)`, the result `y` will have dimensions
`(Y1,Y2,...,Cx,N)` where

    Yi=1+floor((Xi+2*padding[i]-window[i])/stride[i])

Here `Cx` is the number of input channels, `N` is the number of instances, and `Xi,Yi` are
spatial dimensions. `window`, `padding`, `stride` are keyword arguments that can be
specified as a single number (in which case they apply to all dimensions), or an array/tuple
with entries for each spatial dimension. If `window[i] > Xi+2*padding[i]` it is truncated to
fit the padded input, e.g. one can use `window=typemax(Int)` to implement global pooling.

# Keywords:

* `op=maximum`: pooling operation, can be `maximum` or `mean`
* `window=2`: the pooling window size for each dimension.
* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=window`: the number of elements to slide to reach the next pooling window.
* `propagateNaN=false`: if false do not propagate NaN values in max pooling.
* `includePadding=false`: if false do not include padded values in the count for average.
* `channelmajor = false`: assume channel-major format tensors if true
* `alpha=1`: can be used to scale the result.

"""
function pool(x; op=maximum, window=2, padding=0, stride=window, propagateNaN=false, includePadding=false, channelmajor=false, alpha=1)
    checkpoolopts(x, op, window, padding, stride, propagateNaN, includePadding, channelmajor, alpha)
    if channelmajor; x = permutedims(x,(2,3,1,4)); end
    N = ndims(x)
    window = NNlib.expand(Val(N-2), window)
    stride = NNlib.expand(Val(N-2), stride)
    padding = NNlib.expand(Val(N-2), padding)
    window = min.(window, size(x)[1:N-2] .+ 2 .* padding)
    pdims = NNlib.PoolDims(x, window; padding = padding, stride = stride)
    y = (op == maximum ? NNlib.maxpool(x, pdims) : NNlib.meanpool(x, pdims))
    if channelmajor; y = permutedims(y, (3,1,2,4)); end
    alpha == 1 ? y : lmul!(eltype(y)(alpha), y)
end

function ∇pool(x,y,dy; op=maximum, window=2, padding=0, stride=window, propagateNaN=false, includePadding=false, channelmajor=false, alpha=1)
    checkpoolopts(x, op, window, padding, stride, propagateNaN, includePadding, channelmajor, alpha)
    if alpha != 1; y = y ./ eltype(y)(alpha); end
    if channelmajor; x,y,dy = (a->permutedims(a,(2,3,1,4))).((x,y,dy)); end
    N = ndims(x)
    window = NNlib.expand(Val(N-2), window)
    stride = NNlib.expand(Val(N-2), stride)
    padding = NNlib.expand(Val(N-2), padding)
    window = min.(window, size(x)[1:N-2] .+ 2 .* padding)
    pdims = NNlib.PoolDims(x, window; padding = padding, stride = stride)
    dx = (op == maximum ? NNlib.∇maxpool(dy, y, x, pdims) : NNlib.∇meanpool(dy, y, x, pdims))
    if channelmajor; dx = permutedims(dx, (3,1,2,4)); end
    alpha == 1 ? dx : lmul!(eltype(dx)(alpha), dx)
end

@primitive1 pool(x;o...),dy,y  ∇pool(x,y,dy;o...)
@primitive1 ∇pool(x,y,dy;o...),ddx,dx  nothing  nothing  pool(ddx;o...)


function checkpoolopts(x, op, window, padding, stride, propagateNaN, includePadding, channelmajor, alpha)
    @assert op ∈ (maximum, mean) "Pooling op=$op should be one of (maximum,mean)"
    if op == maximum && !propagateNaN
        @warn "propagateNaN=false not yet implemented in NNlib, NaNs will be propagated. See https://github.com/FluxML/NNlib.jl/issues/218" maxlog=1
    end
    if op == mean && any(padding .> 0) && !includePadding
        @warn "includePadding=false not yet implemented in NNlib, pads will be included. See https://github.com/FluxML/NNlib.jl/issues/218" maxlog=1
    end
    if channelmajor
        @warn "channelmajor is not yet implemented in NNlib, using slow permutedims, see NNlib#267" maxlog=1
    end
end    
