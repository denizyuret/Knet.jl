using AutoGrad: AutoGrad, @primitive1, value, recording

using CUDA.CUDNN:
    #cudnnConvolutionForward,
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
    cudnnConvolutionDescriptor_t,
        cudnnCreateConvolutionDescriptor,
        cudnnSetConvolutionNdDescriptor,
        cudnnDestroyConvolutionDescriptor,
    cudnnConvolutionMode_t,
        CUDNN_CONVOLUTION,       # 0
        CUDNN_CROSS_CORRELATION, # 1
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
    handle


"""
TODO

"""
cudnnConvolutionForward, cudnnConvolutionForward!


cudnnConvolutionForward(w, x; o...)               = cudnnConvolutionForwardWithDefaults(w, x; o...)
cudnnConvolutionForward(w, x, convDesc; o...)     = cudnnConvolutionForwardWithDefaults(w, x; convDesc, o...)
cudnnConvolutionForward!(y, w, x; o...)           = cudnnConvolutionForwardWithDefaults(w, x; y, o...)
cudnnConvolutionForward!(y, w, x, convDesc; o...) = cudnnConvolutionForwardWithDefaults(w, x; y, convDesc, o...)


function cudnnConvolutionForwardWithDefaults(
    w,
    x;
    padding::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 0,  # >= 0
    stride::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 1,   # >= 1
    dilation::Union{Integer,Vector{<:Integer},Tuple{<:Integer,Vararg{Int}}} = 1, # >= 1
    mode::cudnnConvolutionMode_t = CUDNN_CONVOLUTION,
    dataType::DataType = cudnnConvolutionDataType(eltype(x)),
    mathType::cudnnMathType_t = cudnnConvolutionMathType(eltype(x)),
    reorderType::cudnnReorderType_t = cudnnConvolutionReorderType(),
    group::Integer = 1,
    convDesc = cudnnConvolutionDescriptor(convdims(padding,size(x)), convdims(stride,size(x)), convdims(dilation,size(x)), mode, cudnnDataType(dataType), mathType, reorderType, Cint(group)),
    format::cudnnTensorFormat_t = CUDNN_TENSOR_NCHW,
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x; format),
    wDesc::cudnnFilterDescriptor = cudnnFilterDescriptor(w; format),
    y = cudnnConvolutionForwardOutput(x, xDesc, wDesc, convDesc),
    yDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(y; format),
    alpha::Real = 1,
    beta::Real = 0,
)
    alpha, beta = scalr(alpha,x), scalr(beta,x)
    cudnnConvolutionForwardAutoGrad(w, x; convDesc, wDesc, xDesc, yDesc, y, alpha, beta)
end


function cudnnConvolutionForwardAutoGrad(w, x; convDesc, wDesc, xDesc, yDesc, y, alpha, beta)
    p = cudnnConvolutionFwdAlgoPerf(xDesc, x, wDesc, w, convDesc, yDesc, y)
    workspace = cudnnConvolutionWorkspace(p.memory)
    CUDA.CUDNN.cudnnConvolutionForward(handle(), alpha, xDesc, x, wDesc, w, convDesc, p.algo, cu_null(workspace), sizeof(workspace), beta, yDesc, y)
    return y
end


# Define gradients
@primitive1((cudnnConvolutionForwardAutoGrad(w, x; convDesc, wDesc, xDesc, yDesc, y, alpha, beta),
             _dy,_y),
            ((x,dy,dw) = (value(x),value(_dy),similar(w));
             p = cudnnConvolutionBwdFilterAlgoPerf(xDesc, x, yDesc, dy, convDesc, wDesc, dw);
             workspace = cudnnConvolutionWorkspace(p.memory);
             cudnnConvolutionBackwardFilter(handle(), alpha, xDesc, x, yDesc, dy, convDesc, p.algo, cu_null(workspace), sizeof(workspace), beta, wDesc, dw);
             dw),
            ((w,dy,dx) = (value(w),value(_dy),similar(x));
             p = cudnnConvolutionBwdDataAlgoPerf(wDesc, w, yDesc, dy, convDesc, xDesc, dx);
             workspace = cudnnConvolutionWorkspace(p.memory);
             cudnnConvolutionBackwardData(handle(), alpha, wDesc, w, yDesc, dy, convDesc, p.algo, cu_null(workspace), sizeof(workspace), beta, xDesc, dx);
             dx))


function cudnnSetConvolutionDescriptor(
    ptr::cudnnConvolutionDescriptor_t,
    padding::Vector{Cint},
    stride::Vector{Cint},
    dilation::Vector{Cint},
    mode::cudnnConvolutionMode_t,
    dataType::cudnnDataType_t,
    mathType::cudnnMathType_t,
    reorderType::cudnnReorderType_t,
    groupCount::Cint,
)
    cudnnSetConvolutionNdDescriptor(ptr, Cint(length(padding)), padding, stride, dilation, mode, dataType)
    mathType != CUDNN_DEFAULT_MATH       && cudnnSetConvolutionMathType(ptr, mathType)
    reorderType != CUDNN_DEFAULT_REORDER && cudnnSetConvolutionReorderType(ptr, reorderType)
    groupCount != 1                      && cudnnSetConvolutionGroupCount(ptr, groupCount)
end


function cudnnConvolutionForwardOutput(x, xDesc, wDesc, convDesc)
    d = Array{Cint}(undef, max(4, ndims(x)))
    cudnnGetConvolutionNdForwardOutputDim(convDesc, xDesc, wDesc, length(d), d)
    if length(d) > ndims(x) # This happens when x is (X,C,N), xDesc is [N,C,X,1]
        @assert all(d[ndims(x)+1:end] .== 1)
        d = d[1:ndims(x)]
    end
    return similar(x, reverse(d)...)
end


# Convert the integer, tuple or array to convolution dims compatible with array size
function convdims(d, s::Dims{N}) where N
    if d isa Integer || length(d) == N-2
        Cint[reverse(min.(d,s[1:N-2]))...]
    else
        throw(DimensionMismatch("Cannot conv $(Base.dims2string(s)) array with $d convdims."))
    end
end

convdims(d, s::Dims{3}) = convdims(d, (1,s...))
convdims(d, s::Dims{2}) = convdims(d, (1,1,s...))
convdims(d, s::Dims{1}) = convdims(d, (1,1,1,s...))
convdims(d, s::Dims{0}) = convdims(d, (1,1,1,1))


# datatype: Selects the data type in which the computation will be done.
# Note:CUDNN_DATA_HALF in cudnnSetConvolutionNdDescriptor() with HALF_CONVOLUTION_BWD_FILTER is not recommended as it is known to not be useful for any practical use case for training and will be considered to be blocked in a future cuDNN release. The use of CUDNN_DATA_HALF for input tensors in cudnnSetTensorNdDescriptor() and CUDNN_DATA_FLOAT in cudnnSetConvolutionNdDescriptor() with HALF_CONVOLUTION_BWD_FILTER is recommended and is used with the automatic mixed precision (AMP) training in many well known deep learning frameworks.

cudnnConvolutionDataType(::Type{Float16}) = Float32
cudnnConvolutionDataType(::Type{Float32}) = Float32
cudnnConvolutionDataType(::Type{Float64}) = Float64


# cudnnMathType_t is an enumerated type used to indicate if the use of Tensor Core operations is permitted in a given library routine.
# CUDNN_DEFAULT_MATH: Tensor Core operations are not used on pre-NVIDIA A100 GPU devices. On A100 GPU devices, Tensor Core TF32 operation is permitted.
# CUDNN_TENSOR_OP_MATH: The use of Tensor Core operations is permitted but will not actively perform datatype down conversion on tensors in order to utilize Tensor Cores.
# CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION: The use of Tensor Core operations is permitted and will actively perform datatype down conversion on tensors in order to utilize Tensor Cores.
# CUDNN_FMA_MATH: Restricted to only kernels that use FMA instructions.
# On pre-NVIDIA A100 GPU devices, CUDNN_DEFAULT_MATH and CUDNN_FMA_MATH have the same behavior: Tensor Core kernels will not be selected. With NVIDIA Ampere GPU architecture and CUDA Toolkit 11, CUDNN_DEFAULT_MATH permits TF32 Tensor Core operation and CUDNN_FMA_MATH does not. The TF32 behavior for CUDNN_DEFAULT_MATH can be explicitly disabled by the environment variable NVIDIA_TF32_OVERRIDE=0.

cudnnConvolutionMathType(::Type) = CUDNN_DEFAULT_MATH  # TODO: test different options


cudnnConvolutionReorderType() = CUDNN_DEFAULT_REORDER  # cudnn 8.0.3 provides no meaningful documentation for this!


## Utilities to find a fast algorithm

const cudnnConvolutionFwdAlgoPerfCache = Dict{Tuple,cudnnConvolutionFwdAlgoPerf_t}()
function cudnnConvolutionFwdAlgoPerf(xDesc, x, wDesc, w, convDesc, yDesc, y)
    get!(cudnnConvolutionFwdAlgoPerfCache, (xDesc, wDesc, convDesc)) do 
        requestedAlgoCount = Int(CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
        returnedAlgoCount = Cint[0]
        perfResults = Array{cudnnConvolutionFwdAlgoPerf_t}(undef,requestedAlgoCount)
        workSpace = cudnnFindConvolutionAlgorithmWorkspace(x)
        cudnnFindConvolutionForwardAlgorithmEx(handle(),xDesc,x,wDesc,w,convDesc,yDesc,y,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,sizeof(workSpace))
        perfChoose(perfResults, returnedAlgoCount[1])
    end
end

const cudnnConvolutionBwdDataAlgoPerfCache = Dict{Tuple,cudnnConvolutionBwdDataAlgoPerf_t}()
function cudnnConvolutionBwdDataAlgoPerf(wDesc, w, dyDesc, dy, convDesc, dxDesc, dx)
    get!(cudnnConvolutionBwdDataAlgoPerfCache, (wDesc, dyDesc, convDesc)) do 
        requestedAlgoCount = Int(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT)
        returnedAlgoCount = Cint[0]
        perfResults = Array{cudnnConvolutionBwdDataAlgoPerf_t}(undef,requestedAlgoCount)
        workSpace = cudnnFindConvolutionAlgorithmWorkspace(dy)
        cudnnFindConvolutionBackwardDataAlgorithmEx(handle(),wDesc,w,dyDesc,dy,convDesc,dxDesc,dx,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,sizeof(workSpace))
        perfChoose(perfResults, returnedAlgoCount[1])
    end
end

const cudnnConvolutionBwdFilterAlgoPerfCache = Dict{Tuple,cudnnConvolutionBwdFilterAlgoPerf_t}()
function cudnnConvolutionBwdFilterAlgoPerf(xDesc, x, dyDesc, dy, convDesc, dwDesc, dw)
    get!(cudnnConvolutionBwdFilterAlgoPerfCache, (xDesc, dyDesc, convDesc)) do 
        requestedAlgoCount = Int(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT)
        returnedAlgoCount = Cint[0]
        perfResults = Array{cudnnConvolutionBwdFilterAlgoPerf_t}(undef,requestedAlgoCount)
        workSpace = cudnnFindConvolutionAlgorithmWorkspace(x)
        cudnnFindConvolutionBackwardFilterAlgorithmEx(handle(),xDesc,x,dyDesc,dy,convDesc,dwDesc,dw,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,sizeof(workSpace))
        perfChoose(perfResults, returnedAlgoCount[1])
    end
end


# Return algorithm with best memory that is within 10% of best time
function perfChoose(ps, n)
    (ibest,mbest,tbest) = (0,Inf,Inf)
    for i = 1:n
        # These metrics are written in a sorted fashion where the first element has the lowest compute time.
        if ps[i].status == 0 && ps[i].memory < mbest && ps[i].time < tbest * 1.1
            (ibest,mbest,tbest) = (i,ps[i].memory,ps[i].time)
        end
    end
    @assert ibest > 0 "No valid algorithm found for convolution."
    return ps[ibest]
end


# Allocate the maximum reasonable amount of memory for algorithm discovery
function cudnnFindConvolutionAlgorithmWorkspace(x)
    gpufree = Mem.info()[1] + (isdefined(CUDA,:pool) ? CUDA.pool[].cached_memory() : CUDA.cached_memory())
    nbytes = min(gpufree ÷ 10, sizeof(x) * 100)
    cudnnConvolutionWorkspace(nbytes)
end


# Use 128 to avoid alignment issues
function cudnnConvolutionWorkspace(nbytes)
    nbytes == 0 ? nothing : CuArray{Int128}(undef, (nbytes-1)÷sizeof(Int128)+1)
end


# TODO: convbiasact
