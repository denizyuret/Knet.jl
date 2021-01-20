export conv
import NNlib
using AutoGrad

"""
    conv(w, x; kwargs...)

Return the convolution of filter `w` with tensor `x`, optionally with bias/residual
addition, activation and/or scaling:

    y = activation.(alpha * conv(w,x) + beta * z .+ bias) 

All tensors should have the same number of dimensions. 3-5 dimensional tensors are
supported. The sizes should be:

    tensor  default       channelmajor
    ------  -------       ------------
    x       (X...,Cx,N)   (Cx,X...,N)
    y       (Y...,Cy,N)   (Cy,Y...,N)
    w       (W...,Cw,Cy)  (Cw,W...,Cy) # Cw=Cx÷group
    bias    (1...,Cy,1)   (Cy,1...)
    z       (Y...,Cy,N)   (Cy,Y...,N)  # same as y

where `(X...)`, `(Y...)`, `(W...)` are spatial dimensions, `Cx`,`Cy`,`Cw` are the number of
input/output/filter channels, and `N` is the number of instances. Both `Cx` and `Cy` have to
be an exact multiple of `group` and `Cw` must be `Cx÷group`. The `channelmajor` option
corresponds to `CUDNN_TENSOR_NHWC` format in cuDNN.

The arguments `padding`, `stride` and `dilation` can be specified as `n-2` dimensional
integer vectors, tuples or a single integer which is assumed to be repeated `n-2` times
where `n` is the number of `x` dimensions. If any of the entries is larger than the
corresponding `x` dimension, the `x` dimension is used instead. For a description of
different types of convolution see:
https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215

Keyword arguments:
* `activation = nothing`: apply activation function if provided
* `alpha = 1, beta = 0`: scaling parameters
* `bias = nothing`: add bias if provided
* `channelmajor = false`: assume channel-major format tensors if specified
* `crosscorrelation = false`: apply cross-correlation rather than convolution if true
* `dilation = 1`: dilation factor
* `group = 1`: number of groups to be used
* `padding = 0`: padding assumed around `x`
* `stride = 1`: how far to shift the convolution window at each step
* `z = nothing`: add `beta*z` to the result if specified
"""
function conv(
    w, x;
    activation = nothing,
    alpha = 1,
    beta = 0,
    bias = nothing,
    channelmajor = false,
    crosscorrelation = false,
    dilation = 1,
    group = 1,
    padding = 0,
    stride = 1,
    z = nothing,
    o...
)
    if channelmajor
        @warn "channelmajor is not implemented on the CPU yet, using slow permutedims, see NNlib#267" maxlog=1
        w,x,z,bias = (x->(x===nothing ? nothing : permutedims(x,(2,3,1,4)))).((w,x,z,bias))
    end
    N = ndims(w)
    stride = NNlib.expand(Val(N-2), stride)
    padding = NNlib.expand(Val(N-2), padding)
    dilation = NNlib.expand(Val(N-2), dilation)
    if group == 1
        cdims = NNlib.DenseConvDims(size(x), size(w); stride, padding, dilation, flipkernel=crosscorrelation)
        y = NNlib.conv(x, w, cdims)
    else
        @assert size(x,3) == size(w,3)*group
        @assert size(w,4) % group == 0
        ys,dx,dy = Array{Any}(undef,group), size(w,3), size(w,4)÷group
        for i = 1:group
            xi = view(x, :, :, (1+(i-1)*dx):(i*dx), :)
            wi = view(w, :, :, :, (1+(i-1)*dy):(i*dy))
            cdims = NNlib.DenseConvDims(size(xi), size(wi); stride, padding, dilation, flipkernel=crosscorrelation)
            ys[i] = NNlib.conv(xi, wi, cdims)
        end
        y = cat(ys...; dims=3)
    end
    if alpha != 1; y = alpha * y; end
    if beta != 0 && z !== nothing; y = y + beta * z; end
    if bias !== nothing; y = y .+ bias; end
    if activation !== nothing && activation !== identity; y = activation.(y); end
    if channelmajor; y = permutedims(y, (3,1,2,4)); end
    return y
end


@primitive1 NNlib.conv(x, w, cdims; o...),dy,y  NNlib.∇conv_data(dy, w, cdims; o...)  NNlib.∇conv_filter(x, dy, cdims; o...)
@primitive1 NNlib.∇conv_data(dy, w, cdims; o...),ddx,dx  NNlib.conv(ddx, w, cdims; o...)  NNlib.∇conv_filter(ddx, dy, cdims; o...)
@primitive1 NNlib.∇conv_filter(x, dy, cdims; o...),ddw,dw  NNlib.∇conv_data(dy, ddw, cdims; o...)  NNlib.conv(x, ddw, cdims; o...)
