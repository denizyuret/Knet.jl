export conv4, deconv4, mat, pool, unpool
using NNlib: conv, DenseConvDims, maxpool, meanpool, PoolDims, ∇conv_data, ∇conv_filter, ∇maxpool, ∇meanpool
using LinearAlgebra: lmul!
using AutoGrad: AutoGrad, @primitive1


"""
    conv4(w, x; kwargs...)

Execute convolutions or cross-correlations using filters specified with `w` over tensor `x`.

If `w` has dimensions `(W1,W2,...,Cx,Cy)` and `x` has dimensions `(X1,X2,...,Cx,N)`, the
result `y` will have dimensions `(Y1,Y2,...,Cy,N)` where `Cx` is the number of input channels,
`Cy` is the number of output channels, `N` is the number of instances, and `Wi,Xi,Yi` are
spatial dimensions with `Yi` determined by:

    Yi = 1 + floor((Xi + 2*padding[i] - ((Wi-1)*dilation[i] + 1)) / stride[i])

`padding`, `stride` and `dilation` are keyword arguments that can be specified as a single
number (in which case they apply to all dimensions), or an array/tuple with entries for each
spatial dimension.

# Keywords

* `padding=0`: the number of extra zeros implicitly concatenated at the start and end of each dimension.
* `stride=1`: the number of elements to slide to reach the next filtering window.
* `dilation=1`: dilation factor for each dimension.
* `mode=0`: 0 for convolution and 1 for cross-correlation (which flips the filter).
* `alpha=1`: can be used to scale the result.
* `group=1`: can be used to perform grouped convolutions.

"""
function conv4(w::AbstractArray{T,N}, x::AbstractArray{T,N};
               padding=0, stride=1, dilation=1, mode=0, alpha=1, group=1) where {T,N}
    @assert group == 1 "Grouped convolutions not yet implemented in NNlib, see https://github.com/JuliaGPU/CuArrays.jl/pull/523" #TODO
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    dilation = expand(Val(N-2), dilation)
    cdims = DenseConvDims(x, w; stride = stride, padding = padding, dilation = dilation, flipkernel = (mode!=0))
    y = conv(x, w, cdims)
    alpha == 1 ? y : lmul!(alpha, y)
end

function conv4w(w::AbstractArray{T,N},x::AbstractArray{T,N},dy::AbstractArray{T,N};
                padding=0, stride=1, dilation=1, mode=0, alpha=1, group=1) where {T,N}
    @assert group == 1 "Grouped convolutions not yet implemented in NNlib, see https://github.com/JuliaGPU/CuArrays.jl/pull/523"
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    dilation = expand(Val(N-2), dilation)
    cdims = DenseConvDims(x, w; stride = stride, padding = padding, dilation = dilation, flipkernel = (mode!=0))
    dw = ∇conv_filter(x, dy, cdims)
    alpha == 1 ? dw : lmul!(alpha, dw)
end

function conv4x(w::AbstractArray{T,N},x::AbstractArray{T,N},dy::AbstractArray{T,N};
                padding=0, stride=1, dilation=1, mode=0, alpha=1, group=1) where {T,N}
    @assert group == 1 "Grouped convolutions not yet implemented in NNlib, see https://github.com/JuliaGPU/CuArrays.jl/pull/523"
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    dilation = expand(Val(N-2), dilation)
    cdims = DenseConvDims(x, w; stride = stride, padding = padding, dilation = dilation, flipkernel = (mode!=0))
    dx = ∇conv_data(dy, w, cdims)
    alpha == 1 ? dx : lmul!(alpha, dx)
end

@primitive1 conv4(w,x; o...),dy,y       conv4w(w,x,dy;o...)   conv4x(w,x,dy;o...)
@primitive1 conv4w(w,x,dy;o...),ddw,dw  nothing               conv4x(ddw,x,dy;o...)  conv4(ddw,x;o...)
@primitive1 conv4x(w,x,dy;o...),ddx,dx  conv4w(w,ddx,dy;o...) nothing                conv4(w,ddx;o...)


"""
    deconv4(w, x; kwargs...)

Simulate 4-D deconvolution by using _transposed convolution_ operation. Its forward pass is
equivalent to backward pass of a convolution (gradients with respect to input
tensor). Likewise, its backward pass (gradients with respect to input tensor) is equivalent to
forward pass of a convolution. Since it swaps forward and backward passes of convolution
operation, padding and stride options belong to output tensor. See [this
report](https://arxiv.org/abs/1603.07285) for further explanation.

If `w` has dimensions `(W1,W2,...,Cy,Cx)` and `x` has dimensions `(X1,X2,...,Cx,N)`, the
result `y=deconv4(w,x)` will have dimensions `(Y1,Y2,...,Cy,N)` where

    Yi = (Xi - 1)*stride[i] + ((Wi-1)*dilation[i] + 1) - 2*padding[i]

Here Cx is the number of x channels, Cy is the number of y channels, N is the number of
instances, and Wi,Xi,Yi are spatial dimensions. Padding and stride are keyword arguments that
can be specified as a single number (in which case they apply to all dimensions), or an
array/tuple with entries for each spatial dimension.

# Keywords

* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=1`: the number of elements to slide to reach the next filtering window.
* `mode=0`: 0 for convolution and 1 for cross-correlation.
* `alpha=1`: can be used to scale the result.
* `handle`: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.
* `group=1`: can be used to perform grouped convolutions.

"""
function deconv4(w,x; o...)
    y = similar(x,dcdims(w,x;o...))
    return conv4x(w,y,x;o...)
end

@primitive1 deconv4(w,x;o...),dy  conv4w(w,dy,x;o...)  conv4(w,dy;o...)


"""
    pool(x; kwargs...)

Compute pooling of input values (i.e., the maximum or average of several adjacent values) to
produce an output with smaller height and/or width.

If `x` has dimensions `(X1,X2,...,Cx,N)`, the result `y` will have dimensions
`(Y1,Y2,...,Cx,N)` where

    Yi=1+floor((Xi+2*padding[i]-window[i])/stride[i])

Here `Cx` is the number of input channels, `N` is the number of instances, and `Xi,Yi` are
spatial dimensions.  `window`, `padding` and `stride` are keyword arguments that can be
specified as a single number (in which case they apply to all dimensions), or an array/tuple
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
    @assert maxpoolingNanOpt==1 "maxpoolingNanOpt not yet implemented in NNlib, see https://github.com/FluxML/NNlib.jl/issues/218" # TODO
    @assert mode != 2 "Pool mode=2 not yet implemented in NNlib, see https://github.com/FluxML/NNlib.jl/issues/218" # TODO
    @assert mode==0 || mode==1 "Bad pooling mode=$mode"
    window = expand(Val(N-2), window)
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    pdims = PoolDims(x, window; padding = padding, stride = stride)
    y = (mode == 0 ? maxpool(x, pdims) :
         mode == 1 ? meanpool(x, pdims) :
         # mode == 2 ? meanpool(x, pdims) : ## CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING is missing in NNlib Issue #218
         error("mode=$mode is not supported for CPU pool."))
    alpha == 1 ? y : lmul!(alpha, y)
end

function poolx(x::AbstractArray{T,N},y::AbstractArray{T,N},dy::AbstractArray{T,N}; 
               alpha=1, mode=0, window=2, padding=0, stride=window, maxpoolingNanOpt=1) where {T,N}
    @assert maxpoolingNanOpt==1 "maxpoolingNanOpt not yet implemented in NNlib, see https://github.com/FluxML/NNlib.jl/issues/218"
    @assert mode != 2 "Pool mode=2 not yet implemented in NNlib, see https://github.com/FluxML/NNlib.jl/issues/218"
    @assert mode==0 || mode==1 "Bad pooling mode=$mode"
    if alpha != 1
        y = y ./ T(alpha)
    end
    window = expand(Val(N-2), window)
    stride = expand(Val(N-2), stride)
    padding = expand(Val(N-2), padding)
    pdims = PoolDims(x, window; padding = padding, stride = stride)
    dx = (mode == 0 ? ∇maxpool(dy, y, x, pdims) :
          mode == 1 ? ∇meanpool(dy, y, x, pdims) :
          # mode == 2 ? ∇meanpool(dy, y, x, pdims) : ## CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING is missing in NNlib Issue #218
          error("mode=$mode is not supported for CPU pool."))
    alpha == 1 ? dx : lmul!(alpha, dx)
end

@primitive1 pool(x;o...),dy,y  poolx(x,y,dy;o...)
@primitive1 poolx(x,y,dy;o...) #TODO


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

@primitive1  unpool(x;o...),dy,y  unpoolx(dy;o...)
@primitive1  unpoolx(dy;o...)  # TODO


"""
    mat(x; dims = ndims(x) - 1)

Reshape `x` into a two-dimensional matrix by joining the first dims dimensions, i.e. 
`reshape(x, prod(size(x,i) for i in 1:dims), :)`

`dims=ndims(x)-1` (default) is typically used when turning the output of a 4-D convolution
result into a 2-D input for a fully connected layer.

`dims=1` is typically used when turning the 3-D output of an RNN layer into a 2-D input for
a fully connected layer.

`dims=0` will turn the input into a row vector, `dims=ndims(x)` will turn it into a column
vector.

"""
mat(x; dims::Int=ndims(x)-1)=reshape(x, (dims > 0 ? prod(size(x,i) for i in 1:dims) : 1), :)


## Dimension helpers:

# outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride;
# inputDim = (outputDim - 1) * convolutionStride + (((filterDim-1)*dilation)+1) - 2*pad
function dcdims(w,x; padding=0, stride=1, dilation=1, o...)
    N = ndims(x)
    @assert size(x,N-1) == size(w,N)
    ntuple(N) do i
        if i < N-1
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            di = (if isa(dilation,Number); dilation; else dilation[i]; end)
            si*(size(x,i)-1) + (((size(w,i)-1)*di)+1) - 2*pi
        elseif i == N-1
            size(w,N-1)
        else
            size(x,N)
        end
    end
end

# convert padding etc. size to an Int array of the right dimension
function psize(p, x)
    nd = ndims(x)-2
    if isa(p,Number)
        fill(Int(p),nd)
    elseif length(p)==nd
        collect(Int,p)
    else
        throw(DimensionMismatch("psize: $p $nd"))
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

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)



# TODO:
# Grouped convolutions not yet implemented in NNlib, see https://github.com/JuliaGPU/CuArrays.jl/pull/523
# Gradient for poolx, unpoolx
# Test for pool, poolx, unpool, unpoolx
# maxpoolingNanOpt not yet implemented in NNlib, see https://github.com/FluxML/NNlib.jl/issues/218
# Pool mode=2 not yet implemented in NNlib, see https://github.com/FluxML/NNlib.jl/issues/218
# unpool Does not work correctly for every window, padding, mode combination.

