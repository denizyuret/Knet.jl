import Knet.Ops21: conv, relu

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
    cudnnConvolutionForwardOutput,
    cudnnTensorDescriptor,
    cudnnFilterDescriptor,
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


function conv(
    w::GPUVal, x::GPUVal;
    z = nothing,
    bias = nothing,
    activation = nothing,
    normalization = nothing,

    alpha::Real = 1,
    beta::Real = 0,

    padding::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 0,  # >= 0
    stride::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 1,   # >= 1
    dilation::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 1, # >= 1
    groups::Integer = 1,
    crosscorrelation::Bool = false,
    channelmajor::Bool = false,

    format::cudnnTensorFormat_t = channelmajor ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
    mode::cudnnConvolutionMode_t = crosscorrelation ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION,
    mathType::cudnnMathType_t = math_mode(),
    reorderType::cudnnReorderType_t = CUDNN_DEFAULT_REORDER,
    convDesc::cudnnConvolutionDescriptor = cudnnConvolutionDescriptor(convdims(padding,size(x),format), convdims(stride,size(x),format), convdims(dilation,size(x),format), mode, cudnnDataType(eltype(x)), mathType, reorderType, Cint(groups)),
    y = cudnnConvolutionForwardOutput(x, cudnnTensorDescriptor(x;format), cudnnFilterDescriptor(w;format), convDesc, format),

    dw = Ref{Any}(nothing),
    dx = Ref{Any}(nothing),
    dz = Ref{Any}(nothing),
    dbias = Ref{Any}(nothing),
)
    a = (activation === relu && normalization ∈ (nothing, identity) ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY)
    y .= 0
    global _preconv_gpux = deepcopy(x)
    global _preconv_gpuy = deepcopy(y)
    r = cudnnConvolutionForward!(y, w, x, convDesc; activation=a, bias, z, alpha, beta, format, dw, dx, dz, dbias)
    global _prenorm_gpu = deepcopy(r)
    if normalization ∉ (nothing, identity); r = normalization(r); end
    global _postnorm_gpu = deepcopy(r)
    if a === CUDNN_ACTIVATION_IDENTITY && activation ∉ (nothing, identity); r = activation.(r); end
    global _postact_gpu = deepcopy(r)
    return r
end
