export Conv
using Knet.Ops21: conv

using CUDA.CUDNN:
    cudnnConvolutionForward,
    cudnnConvolutionForward!,
    cudnnConvolutionBackwardFilter,
    cudnnConvolutionBackwardData,
    cudnnGetConvolutionNdForwardOutputDim,
    cudnnSetConvolutionMathType,
    cudnnSetConvolutionReorderType,
    cudnnSetConvolutionGroupCount,
    cudnnFindConvolutionForwardAlgorithmEx,
        cudnnConvolutionFwdAlgoPerf_t,
    cudnnFindConvolutionBackwardFilterAlgorithmEx,
        cudnnConvolutionBwdFilterAlgoPerf_t,
    cudnnFindConvolutionBackwardDataAlgorithmEx,
        cudnnConvolutionBwdDataAlgoPerf_t,
    cudnnConvolutionDescriptor,
        cudnnConvolutionDescriptor_t,
        cudnnCreateConvolutionDescriptor,
        cudnnSetConvolutionNdDescriptor,
        cudnnDestroyConvolutionDescriptor,
    cudnnConvolutionMode_t,
        CUDNN_CONVOLUTION,       # 0
        CUDNN_CROSS_CORRELATION, # 1
    cudnnActivationMode_t,
        CUDNN_ACTIVATION_SIGMOID,      # 0
        CUDNN_ACTIVATION_RELU,         # 1
        CUDNN_ACTIVATION_TANH,         # 2
        CUDNN_ACTIVATION_CLIPPED_RELU, # 3
        CUDNN_ACTIVATION_ELU,          # 4
        CUDNN_ACTIVATION_IDENTITY,     # 5
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN, # 0
        CUDNN_PROPAGATE_NAN,     # 1
    cudnnMathType_t,
        CUDNN_DEFAULT_MATH,                    # 0
        CUDNN_TENSOR_OP_MATH,                  # 1
        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION, # 2
        CUDNN_FMA_MATH,                        # 3
    cudnnReorderType_t,
        CUDNN_DEFAULT_REORDER, # 0
        CUDNN_NO_REORDER,      # 1
    cudnnConvolutionFwdAlgo_t,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,         # 0
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, # 1
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,                  # 2
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,                # 3
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,                   # 4
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,            # 5
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,              # 6
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,     # 7
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT,                 # 8
    cudnnConvolutionBwdFilterAlgo_t,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,                 # 0, /* non-deterministic */
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,                 # 1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,               # 2,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,                 # 3, /* non-deterministic */
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,          # 4, /* not implemented */
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED, # 5,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,        # 6,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,             # 7
    cudnnConvolutionBwdDataAlgo_t,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,                 # 0, /* non-deterministic */
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,                 # 1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,               # 2,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,        # 3,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,          # 4,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED, # 5,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,             # 6
    cudnnTensorFormat_t,
        CUDNN_TENSOR_NCHW,        # 0, /* row major (wStride = 1, hStride = w) */
        CUDNN_TENSOR_NHWC,        # 1, /* feature maps interleaved ( cStride = 1 )*/
        CUDNN_TENSOR_NCHW_VECT_C, # 2, /* each image point is vector of element of C, vector length in data type */
    cudnnDataType,
    convdims,
    math_mode,
    handle


"""
    c = Conv(dims...; kwargs...)
    c = Conv(weights; kwargs...)
    y = c(x, z=nothing)

Return a convolution layer that can perform convolution and optionally bias/residual
addition, activation and/or scaling. 

    y = activation.(alpha * conv(w,x) + beta * z .+ bias) 

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
mutable struct Conv
    wdims
    bdims

    w
    bias
    y

    dw
    dbias
    dx
    dz

    padding::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}}
    stride::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}}
    dilation::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}}
    group::Integer
    alpha::Real
    beta::Real

    convDesc::cudnnConvolutionDescriptor
    activation::cudnnActivationMode_t
    format::cudnnTensorFormat_t
end


Conv(w; o...) = Conv(size(w)...; w, o...)

function Conv(
    wdims::Integer...;
    w = nothing,
    activation = nothing,
    alpha = 1,
    beta = 0,
    bias = nothing,
    dilation = 1,
    flipkernel = false,
    group = 1,
    padding = 0,
    stride = 1,
)
end


function initconv(c::Conv, x, z)
    issimilar(u,v,s=size(v))=(typeof(value(u)) === typeof(value(v)) && size(u) === s)
    if c.w === nothing
        # TODO: init w
    end
    @assert typeof(x) === typeof(c.w)
    @assert (c.format === CUDNN_TENSOR_NHWC ? size(x,1) === size(w,1) : size(x)[end-1] === size(w)[end-1])
    if c.bias === nothing && (c.activation === relu || z !== nothing)
        c.bias = fill!(similar(c.w, c.bdims), 0)
    end
    @assert c.bias === nothing || issimilar(c.bias, c.w, c.bdims)
    # TODO: calculate ydims: do it with Refs? what if same Conv run multiple times in an iteration? same in batchnorm?
    if !issimilar(c.y, x, ydims)
        c.y = similar(x, ydims)
    end
    @assert z === nothing || issimilar(z, c.y)
    if c.convDesc === nothing
        mode = flipkernel ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION
        reorderType, mathType = CUDNN_DEFAULT_REORDER, math_mode()
        c.convDesc = cudnnConvolutionDescriptor(convdims(c.padding,size(x),c.format), convdims(c.stride,size(x),c.format), convdims(c.dilation,size(x),c.format), mode, cudnnDataType(eltype(x)), mathType, reorderType, Cint(c.group))
    end
end


function (c::Conv)(x, z=nothing)
    initConv(c, x, z)
    conv(c.w, x; z, c.bias, c.y,
         c.activation, c.alpha, c.beta,
         c.dilation, c.flipkernel, c.group, c.padding, c.stride,
         c.convDesc, c.format, c.dw, c.dx, c.dz, c.dbias)
end
