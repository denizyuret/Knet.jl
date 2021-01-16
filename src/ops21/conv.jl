export conv
import NNlib
using AutoGrad

"""
    conv(w, x; activation, alpha, beta, bias, dilation, flipped, group, padding, stride, z)

Return the convolution of filter `w` with tensor `x`, overwriting `y` if provided, according
to keyword arguments. Optionally perform bias/residual addition, activation and/or scaling:

    activation.(alpha * conv(w,x) + beta * z .+ bias) 

All tensors should have the same number of dimensions. If they are less than 4-D their
dimensions are assumed to be padded on the left with 1's. `x` has size `(X...,Cx,N)` where
`(X...)` are the spatial dimensions, `Cx` is the number of input channels, and `N` is the
number of instances. `y,z` have size `(Y...,Cy,N)` where `(Y...)` are the spatial dimensions
and `Cy` is the number of output channels. Both `Cx` and `Cy` have to be an exact multiple
of `group`.  `w` has size `(W...,Cx÷group,Cy)` where `(W...)` are the filter
dimensions. `bias` has size `(1...,Cy,1)`.

The arguments `padding`, `stride` and `dilation` can be specified as `n-2` dimensional
vectors, tuples or a single integer which is assumed to be repeated `n-2` times. If any of
the entries is larger than the corresponding `x` dimension, the `x` dimension is used
instead. For a description of different types of convolution see:
https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215

Keyword arguments:
* `activation = nothing`: apply activation function if provided
* `alpha = 1, beta = 0`: scaling parameters
* `bias = nothing`: add bias if provided
* `dilation = 1`: dilation factor
* `flipkernel = false`: apply cross-correlation rather than convolution if true
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
    dilation = 1,
    flipkernel = false,
    group = 1,
    padding = 0,
    stride = 1,
    z = nothing,
    o...
)
    if group != 1; error("group != 1 is not supported on the CPU yet, see NNlib#267"); end
    N = ndims(w)
    stride = NNlib.expand(Val(N-2), stride)
    padding = NNlib.expand(Val(N-2), padding)
    dilation = NNlib.expand(Val(N-2), dilation)
    cdims = NNlib.DenseConvDims(size(x), size(w); stride, padding, dilation, flipkernel)
    y = NNlib.conv(x, w, cdims)
    if alpha != 1; y = alpha * y; end
    if beta != 0 && z !== nothing; y = y + beta * z; end
    if bias !== nothing; y = y .+ bias; end
    if activation !== nothing && activation !== identity; y = activation.(y); end
    return y
end


@primitive1 NNlib.conv(x, w, cdims; o...),dy,y  NNlib.∇conv_data(dy, w, cdims; o...)  NNlib.∇conv_filter(x, dy, cdims; o...)
@primitive1 NNlib.∇conv_data(dy, w, cdims; o...),ddx,dx  NNlib.conv(ddx, w, cdims; o...)  NNlib.∇conv_filter(ddx, dy, cdims; o...)
@primitive1 NNlib.∇conv_filter(x, dy, cdims; o...),ddw,dw  NNlib.∇conv_data(dy, ddw, cdims; o...)  NNlib.conv(x, ddw, cdims; o...)
