export Conv
using Knet.Ops21: conv
import CUDA

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
    c = Conv(wdims...; winit, kwargs...)
    c = Conv(w; kwargs...)
    y = c(x, [z=nothing])

Return a layer that can perform convolution and optionally scaling, bias/residual addition,
normalization and/or activation. The constructor can take the convolution weight tensor `w`
or its dimensions `wdims`. If `wdims` is used a weight tensor will be initialized using the
`winit` function which defaults to uniform random in `±√(6/(fanin+fanout))`. The layer can
be called with a single input `x`, or two inputs `x,z` where `z` has the same size as the
output `y`. The computed result is:

    y = activation(normalization(alpha * conv(w,x) + beta * z .+ bias))

For tensor sizes and keyword arguments with their defaults see `@doc Knet.Ops21.conv`.
"""
mutable struct Conv
    w  # No type here, we do not want to restrict the type of array
    wdims::Dims
    winit

    bias  # Can be nothing
    bdims::Dims

    # y; dw; dbias; dx; dz; These are overwritten if multiple calls to same layer. 

    padding::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Integer}}}
    stride::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Integer}}}
    dilation::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Integer}}}
    groups::Integer
    crosscorrelation::Bool
    channelmajor::Bool
    activation::Union{Nothing,Function}
    normalization
    convDesc::Union{Nothing,cudnnConvolutionDescriptor}

    alpha::Real
    beta::Real
end


# If the user provides size, we can't immediately initialize w because we do not know the
# array type until we see the first input
function Conv(
    wdims::Integer...;
    w = nothing,
    bias = nothing,
    activation::Union{Nothing,Function} = nothing,
    normalization = nothing,

    padding::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{<:Integer}}} = 0,
    stride::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{<:Integer}}} = 1,
    dilation::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{<:Integer}}} = 1,
    groups::Integer = 1,
    crosscorrelation::Bool = true,

    channelmajor::Bool = false, # CUDNN_TENSOR_NHWC if true.
    winit = 𝑼(√(6/(fanin(wdims; channelmajor)+fanout(wdims; channelmajor)))),

    alpha::Real = 1,
    beta::Real = 0,
)
    wdims = Int.(wdims)
    dw = Ref{Any}(nothing)
    dbias = Ref{Any}(nothing)
    ndims = length(wdims)
    cdim = channelmajor ? 1 : ndims-1
    bdims = ntuple(i->(i===cdim ? wdims[end] : 1), ndims)
    convDesc = nothing          # To be initialized at first call
    Conv(w, wdims, winit, bias, bdims, padding, stride, dilation, groups, crosscorrelation, channelmajor, activation, normalization, convDesc, alpha, beta)
end


Conv(w; o...) = Conv(size(w)...; w, o...)


# Some of the initialization can only be done at the first call when we know the type of x
# TODO: are we sure about this? should we use atype() instead?
function initconv(c::Conv, x, z)
    issimilar(u,v,s=size(v))=(typeof(value(u)) === typeof(value(v)) && size(u) === s)
    if c.w === nothing
        c.w = Param(oftype(value(x), c.winit(c.wdims...)))
    end
    @assert issimilar(c.w, x, c.wdims)
    @assert (c.channelmajor ? size(x,1) === size(c.w,1)*c.groups : size(x)[end-1] === size(c.w)[end-1]*c.groups)
    if c.bias === nothing && ((c.activation === relu && c.normalization ∈ (nothing, identity)) || (z !== nothing && c.beta != 0))
        # will call cudnnConvolutionBiasActivationForward, must have bias
        c.bias = fill!(similar(c.w, c.bdims), 0)
    end
    @assert c.bias === nothing || issimilar(c.bias, c.w, c.bdims)
    if CUDA.functional() && (c.convDesc === nothing || c.convDesc.ptr === C_NULL)
        mode = c.crosscorrelation ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION
        format = c.channelmajor ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW
        reorderType, mathType = CUDNN_DEFAULT_REORDER, math_mode()
        c.convDesc = cudnnConvolutionDescriptor(convdims(c.padding,size(x),format), convdims(c.stride,size(x),format), convdims(c.dilation,size(x),format), mode, cudnnDataType(eltype(x)), mathType, reorderType, Cint(c.groups))
    end
end


function (c::Conv)(x, z=nothing)
    initconv(c, x, z)
    conv(c.w, x; z, c.bias, c.activation, c.normalization, c.alpha, c.beta, c.channelmajor,
         c.convDesc, c.crosscorrelation, c.dilation, c.groups, c.padding, c.stride)
end
