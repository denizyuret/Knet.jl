using LinearAlgebra: lmul!
import NNlib
expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

"""

    conv4(w, x; kwargs...)

Execute convolutions or cross-correlations using filters specified
with `w` over tensor `x`.

Currently KnetArray{Float32/64,4/5} and Array{Float32/64,4} are
supported as `w` and `x`.  If `w` has dimensions `(W1,W2,...,I,O)` and
`x` has dimensions `(X1,X2,...,I,N)`, the result `y` will have
dimensions `(Y1,Y2,...,O,N)` where

    Yi=1+floor((Xi+2*padding[i]-Wi)/stride[i])

Here `I` is the number of input channels, `O` is the number of output
channels, `N` is the number of instances, and `Wi,Xi,Yi` are spatial
dimensions.  `padding` and `stride` are keyword arguments that can be
specified as a single number (in which case they apply to all
dimensions), or an array/tuple with entries for each spatial
dimension.

# Keywords

* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=1`: the number of elements to slide to reach the next filtering window.
* `dilation=1`: dilation factor for each dimension.
* `mode=0`: 0 for convolution and 1 for cross-correlation.
* `alpha=1`: can be used to scale the result.
* `group=1`: can be used to perform grouped convolutions.

"""
function conv4(w::AbstractArray{T,N}, x::AbstractArray{T,N};
               padding=0, stride=1, dilation=1, mode=0, alpha=1, group=1) where {T,N}
    @assert group == 1 "Grouped convolutions not implemented for CPU yet." #TODO
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    dilation = expand(Val(N-2), dilation)
    cdims = NNlib.DenseConvDims(x, w; stride = stride, padding = padding, dilation = dilation, flipkernel = (mode!=0))
    y = NNlib.conv(x, w, cdims)
    alpha == 1 ? y : lmul!(alpha, y)
end

function conv4w(w::AbstractArray{T,N},x::AbstractArray{T,N},dy::AbstractArray{T,N};
                padding=0, stride=1, dilation=1, mode=0, alpha=1, group=1) where {T,N}
    @assert group == 1 "Grouped convolutions not implemented for CPU yet." #TODO
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    dilation = expand(Val(N-2), dilation)
    cdims = NNlib.DenseConvDims(x, w; stride = stride, padding = padding, dilation = dilation, flipkernel = (mode!=0))
    dw = NNlib.∇conv_filter(x, dy, cdims)
    alpha == 1 ? dw : lmul!(alpha, dw)
end

function conv4x(w::AbstractArray{T,N},x::AbstractArray{T,N},dy::AbstractArray{T,N};
                padding=0, stride=1, dilation=1, mode=0, alpha=1, group=1) where {T,N}
    @assert group == 1 "Grouped convolutions not implemented for CPU yet." #TODO
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    dilation = expand(Val(N-2), dilation)
    cdims = NNlib.DenseConvDims(x, w; stride = stride, padding = padding, dilation = dilation, flipkernel = (mode!=0))
    dx = NNlib.∇conv_data(dy, w, cdims)
    alpha == 1 ? dx : lmul!(alpha, dx)
end

@primitive conv4(w,x; o...),dy  conv4w(w,x,dy;o...)  conv4x(w,x,dy;o...)
@zerograd  conv4x(w,x,dy;o...)
@zerograd  conv4w(w,x,dy;o...)


"""

    pool(x; kwargs...)

Compute pooling of input values (i.e., the maximum or average of
several adjacent values) to produce an output with smaller height
and/or width.

Currently 4 or 5 dimensional KnetArrays with `Float32` or `Float64`
entries are supported.  If `x` has dimensions `(X1,X2,...,I,N)`, the
result `y` will have dimensions `(Y1,Y2,...,I,N)` where

    Yi=1+floor((Xi+2*padding[i]-window[i])/stride[i])

Here `I` is the number of input channels, `N` is the number of
instances, and `Xi,Yi` are spatial dimensions.  `window`, `padding`
and `stride` are keyword arguments that can be specified as a single
number (in which case they apply to all dimensions), or an array/tuple
with entries for each spatial dimension.

# Keywords:

* `window=2`: the pooling window size for each dimension.
* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=window`: the number of elements to slide to reach the next pooling window.
* `mode=0`: 0 for max, 1 for average including padded values, 2 for average excluding padded values.
* `maxpoolingNanOpt=1`: Nan numbers are not propagated if 0, they are propagated if 1.
* `alpha=1`: can be used to scale the result.

"""
function pool(x::AbstractArray{T,N}; 
              alpha=1, mode=0, window=2, padding=0, stride=window, maxpoolingNanOpt=1) where {T,N}
    @assert maxpoolingNanOpt==1 "maxpoolingNanOpt not implemented for the CPU, see https://github.com/FluxML/NNlib.jl/issues/218"
    @assert mode==0 || mode==1 "mode=$mode not implemented for the CPU, see https://github.com/FluxML/NNlib.jl/issues/218"
    window = expand(Val(N-2), window)
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    pdims = NNlib.PoolDims(x, window; padding = padding, stride = stride)
    y = (mode == 0 ? NNlib.maxpool(x, pdims) :
         mode == 1 ? NNlib.meanpool(x, pdims) :
         # mode == 2 ? meanpool(x, pdims) : ## CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING is missing in NNlib Issue #218
         error("mode=$mode is not supported for CPU pool."))
    alpha == 1 ? y : lmul!(alpha, y)
end

function poolx(x::AbstractArray{T,N},y::AbstractArray{T,N},dy::AbstractArray{T,N}; 
               alpha=1, mode=0, window=2, padding=0, stride=window, maxpoolingNanOpt=1) where {T,N}
    @assert maxpoolingNanOpt==1 "maxpoolingNanOpt not implemented for the CPU, see https://github.com/FluxML/NNlib.jl/issues/218"
    @assert mode==0 || mode==1 "mode=$mode not implemented for the CPU, see https://github.com/FluxML/NNlib.jl/issues/218"
    if alpha != 1
        y = y ./ T(alpha)
    end
    window = expand(Val(N-2), window)
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    pdims = NNlib.PoolDims(x, window; padding = padding, stride = stride)
    dx = (mode == 0 ? NNlib.∇maxpool(dy, y, x, pdims) :
          mode == 1 ? NNlib.∇meanpool(dy, y, x, pdims) :
          # mode == 2 ? ∇meanpool(dy, y, x, pdims) : ## CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING is missing in NNlib Issue #218
          error("mode=$mode is not supported for CPU pool."))
    alpha == 1 ? dx : lmul!(alpha, dx)
end

@primitive pool(x;o...),dy,y  poolx(x,y,dy;o...)
@zerograd  poolx(x,y,dy;o...)


"""

Unpooling; `reverse` of pooling. 

Warning: Does not work correctly for every window, padding, mode combination. Test before use. #TODO

    x == pool(unpool(x;o...); o...)

"""
function unpool(x; window=2, alpha=1, o...) # padding=0, stride=window, mode=0, maxpoolingNanOpt=0
    w = prod(psize(window,x))
    y = similar(x,updims(x; window=window, o...))
    poolx(y,x,x.*w; o..., window=window, mode=1, alpha=1/alpha)
end

function unpoolx(dy; window=2, alpha=1, o...) # padding=0, stride=window, mode=0, maxpoolingNanOpt=0
    w = prod(psize(window,dy))
    pool(dy; o..., window=window, mode=1, alpha=1/alpha) * w
end

@primitive  unpool(x;o...),dy,y  unpoolx(dy;o...)


"""

    y = deconv4(w, x; kwargs...)

Simulate 4-D deconvolution by using _transposed convolution_ operation. Its forward pass is
equivalent to backward pass of a convolution (gradients with respect to input
tensor). Likewise, its backward pass (gradients with respect to input tensor) is equivalent
to forward pass of a convolution. Since it swaps forward and backward passes of convolution
operation, padding and stride options belong to output tensor. See [this
report](https://arxiv.org/abs/1603.07285) for further explanation.

Currently KnetArray{Float32/64,4} and Array{Float32/64,4} are supported as `w` and `x`.  If
`w` has dimensions `(W1,W2,...,O,I)` and `x` has dimensions `(X1,X2,...,I,N)`, the result
`y` will have dimensions `(Y1,Y2,...,O,N)` where

Yi = Wi+stride[i]*(Xi-1)-2*padding[i]

Here I is the number of input channels, O is the number of output channels, N is the number
of instances, and Wi,Xi,Yi are spatial dimensions. padding and stride are keyword arguments
that can be specified as a single number (in which case they apply to all dimensions), or an
array/tuple with entries for each spatial dimension.

# Keywords

* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=1`: the number of elements to slide to reach the next filtering window.
* `mode=0`: 0 for convolution and 1 for cross-correlation.
* `alpha=1`: can be used to scale the result.
* `handle`: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.

"""
function deconv4(w,x; o...)
    y = similar(x,dcdims(w,x;o...))
    return conv4x(w,y,x;o...)
end

function deconv4w(w,x,dy; o...)
    return conv4w(w,dy,x;o...)
end

function deconv4x(w,x,dy; o...)
    return conv4(w,dy;o...)
end


@primitive deconv4(w,x; o...),dy,y  deconv4w(w,x,dy; o...)  deconv4x(w,x,dy; o...)
@zerograd deconv4w(w,x,dy; o...)
@zerograd deconv4x(w,x,dy; o...)


## Dimension helpers:

function dcdims(w,x; padding=0, stride=1, dilation=1, o...)
    # TODO: handle dilation here
    dilation != 1 && error("deconv4 cannot handle dilation!=1 yet.")
    N = ndims(x)
    @assert size(x,N-1) == size(w,N)
    ntuple(N) do i
        if i < N-1
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            # di = (if isa(dilation,Number); dilation; else dilation[i]; end)
            si*(size(x,i)-1) + size(w,i) - 2*pi
        elseif i == N-1
            size(w,N-1)
        else
            size(x,N)
        end
    end
end

function updims(x; window=2, padding=0, stride=window, o...)
    window = psize(window,x)
    stride = psize(stride,x)
    padding = psize(padding,x)
    N = ndims(x)
    ntuple(N) do i
        if i < N-1
            (size(x,i)-1)*stride[i]+window[i]-2*padding[i]
        else
            size(x,i)
        end
    end
end

# convolution padding size that preserves the input size when filter size is odd and stride=1
padsize(w)=ntuple(i->div(size(w,i)-1,2), ndims(w)-2)

