import NNlib
using AutoGrad
using Statistics: mean
export pool, mean

#= NOTES
- maxpool/meanpool vs op=maximum, op=mean (would require Statistics)
- adaptivexxx vs specifying output=1 as an alternative to window (can't specify both)
- torch.nn: MaxPoolNd, MaxUnpoolNd, AvgPoolNd, FractionalMaxPoolNd, LPPoolNd, AdaptiveMaxPoolNd, AdaptiveAvgPoolNd
- torch.nn.functional: avg_poolNd, max_poolNd, max_unpoolNd, lp_poolNd, adaptive_max_poolNd, adaptive_avg_poolNd
- adaptive versions take output_size as argument, regular versions take kernel_size.
- torch also supports dilation? cudnn does not.
- torch supports maxunpool but not avgunpool.
- existence of lp may make the op kwarg more preferrable (other ops in the future?)
- keras has MaxPoolingNd, AveragePoolingNd, GlobalMaxPoolingNd, GlobalAveragePoolingNd: only output=1 supported
=#

"""
    pool(x; kwargs...)

Compute pooling of input values (i.e., the maximum or average of several adjacent values) to
produce an output with smaller height and/or width.

If `x` has dimensions `(X1,X2,...,Cx,N)`, the result `y` will have dimensions
`(Y1,Y2,...,Cx,N)` where

    Yi=1+floor((Xi+2*padding[i]-window[i])/stride[i])

Here `Cx` is the number of input channels, `N` is the number of instances, and `Xi,Yi` are
spatial dimensions.  If `output` is specified the `window` is computed adaptively to result
in `Yi` sizes given by `output`.  `window`, `padding`, `stride`, `output` are keyword
arguments that can be specified as a single number (in which case they apply to all
dimensions), or an array/tuple with entries for each spatial dimension.

# Keywords:

* `op=maximum`: pooling operation, can be `maximum` or `mean`
* `window=2`: the pooling window size for each dimension.
* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=window`: the number of elements to slide to reach the next pooling window.
* `output=nothing`: if specified use a window size that will result in this output size.
* `propagateNaN=false`: if false do not propagate NaN values in max pooling.
* `includePadding=false`: if false do not include padded values in the count for average.
* `alpha=1`: can be used to scale the result.

"""
function pool(x; op=maximum, window=2, padding=0, stride=window, output=nothing, propagateNaN=false, includePadding=false, alpha=1)
    checkpoolopts(x, op, window, padding, stride, output, propagateNaN, includePadding, alpha)
    N = ndims(x)
    if output === nothing
        window = expand(Val(N-2), window)
        stride = expand(Val(N-2), stride)
        padding = expand(Val(N-2), padding)
    else
        # TODO: figure out semantics of output for window/stride/padding
    end
    pdims = PoolDims(x, window; padding = padding, stride = stride)
    y = (op == maximum ? maxpool(x, pdims) : meanpool(x, pdims))
    alpha == 1 ? y : lmul!(alpha, y)
end

function poolx(x,y,dy; window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=1, alpha=1)
    mode, maxpoolingNanOpt = checkpoolopts(x, window, padding, stride, mode, maxpoolingNanOpt, alpha)
    if alpha != 1
        y = y ./ eltype(y)(alpha)
    end
    N = ndims(x)
    window = expand(Val(N-2), window)
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    pdims = PoolDims(x, window; padding = padding, stride = stride)
    dx = (mode == 0 ? ∇maxpool(dy, y, x, pdims) :
          mode == 1 ? ∇meanpool(dy, y, x, pdims) :
          mode == 2 ? error("Pool mode=2 not yet implemented in NNlib. See https://github.com/FluxML/NNlib.jl/issues/218") :
          mode == 3 ? ∇maxpool(dy, y, x, pdims) :
          error("mode=$mode is not supported for CPU pool."))
    alpha == 1 ? dx : lmul!(alpha, dx)
end

@primitive1 pool(x;o...),dy,y  poolx(x,y,dy;o...)
@primitive1 poolx(x,y,dy;o...),ddx,dx  nothing  nothing  pool(ddx;o...)


function checkpoolopts(x, op, window, padding, stride, output, propagateNaN, includePadding, alpha)
    @assert op ∈ (maximum, mean) "Pooling op=$op should be one of (maximum,mean)"
    if op == maximum && !propagateNaN
        @warn "propagateNaN=false not yet implemented in NNlib, NaNs will be propagated. See https://github.com/FluxML/NNlib.jl/issues/218" maxlog=1
    end
    if op == mean && any(padding .> 0) && !includePadding
        @warn "includePadding=false not yet implemented in NNlib, pads will be included. See https://github.com/FluxML/NNlib.jl/issues/218" maxlog=1
    end
end    


"""
    unpool(x; o...)

Perform the reverse of pooling: `x == pool(unpool(x;o...); o...)`
"""
function unpool(x; window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=1, alpha=1)
    if mode == 1 && x isa Array
        @warn "unpool(mode=1), which uses poolx(mode=2) is not supported on the CPU; performing unpool(mode=2) instead, see https://github.com/FluxML/NNlib.jl/issues/218" maxlog=1
    end
    w = prod(psize(window,x))
    y = similar(x,updims(x; window, padding, stride, mode, maxpoolingNanOpt, alpha))
    # pool0=>unpool1, pool1=>unpool2, pool2=>unpool1
    mode = (mode==0 ? 1 : mode==1 ? 2 : mode==2 ? 1 : mode==3 ? 1 : error("Unknown unpool mode $mode"))
    alpha = 1/alpha
    # Leave unpool as a non-primitive, it is just a poolx call
    poolx(y,x,x.*w; window, padding, stride, mode, maxpoolingNanOpt, alpha)
end
