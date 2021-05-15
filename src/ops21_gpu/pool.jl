import Knet.Ops21: pool
using Statistics: mean

using CUDA.CUDNN:
    cudnnPoolingForward,
    cudnnPoolingForward!,
    cudnnPoolingBackward,
    cudnnGetPoolingNdForwardOutputDim,
    cudnnPoolingDescriptor,
        cudnnPoolingDescriptor_t,
        cudnnCreatePoolingDescriptor,
        cudnnSetPoolingNdDescriptor,
        cudnnDestroyPoolingDescriptor,
    cudnnPoolingMode_t,
        CUDNN_POOLING_MAX,                           # 0,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, # 1, /* count for average includes padded values */
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, # 2, /* count for average does not include padded values */
        CUDNN_POOLING_MAX_DETERMINISTIC,             # 3
    cudnnNanPropagation_t,
        CUDNN_NOT_PROPAGATE_NAN, # 0
        CUDNN_PROPAGATE_NAN,     # 1
    cudnnTensorFormat_t,
        CUDNN_TENSOR_NCHW,        # 0, /* row major (wStride = 1, hStride = w) */
        CUDNN_TENSOR_NHWC,        # 1, /* feature maps interleaved ( cStride = 1 )*/
        CUDNN_TENSOR_NCHW_VECT_C, # 2, /* each image point is vector of element of C, vector length in data type */
    pooldims


function pool(x::GPUVal; op=maximum, window=2, padding=0, stride=window, propagateNaN=false, includePadding=false, channelmajor=false, alpha=1)
    mode = (op==maximum ? CUDNN_POOLING_MAX :
            op==mean ? (includePadding ?
                        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING :
                        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) :
            error("Unknown pooling op=$op"))
    nanOpt = (propagateNaN ? CUDNN_PROPAGATE_NAN : CUDNN_NOT_PROPAGATE_NAN)
    format = (channelmajor ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW)
    xsize = (channelmajor ? size(x)[2:end-1] : size(x)[1:end-2])
    window = min.(window, xsize .+ 2 .* padding)
    if ndims(x) == 3  # cudnnPoolingForward converts to 4D
        window, padding, stride = (x->x[1]).((window, padding, stride))
    end
    y = cudnnPoolingForward(x; mode, nanOpt, window, padding, stride, format, alpha)
end
