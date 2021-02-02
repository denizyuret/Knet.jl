import Knet.Ops21: batchnorm

using CUDA.CUDNN:
    cudnnNormalizationForward,
    cudnnNormalizationForward!,
    cudnnNormalizationForwardInference,
    cudnnNormalizationForwardTraining,
    cudnnNormalizationBackward,
    cudnnGetNormalizationForwardTrainingWorkspaceSize,
    cudnnGetNormalizationTrainingReserveSpaceSize,
    cudnnActivationDescriptor,
    cudnnTensorDescriptor,
    cudnnNormMode_t,
        CUDNN_NORM_PER_ACTIVATION, # 0, bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice)
        CUDNN_NORM_PER_CHANNEL,    # 1, bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors)
    cudnnNormOps_t,
        CUDNN_NORM_OPS_NORM,                # 0, /* do normalization only */
        CUDNN_NORM_OPS_NORM_ACTIVATION,     # 1, /* do Norm, then activation */
        CUDNN_NORM_OPS_NORM_ADD_ACTIVATION, # 2, /* do Norm, then elemWiseAdd, then activation */
    cudnnNormAlgo_t,
        CUDNN_NORM_ALGO_STANDARD, # 0
        CUDNN_NORM_ALGO_PERSIST,  # 1
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
    cudnnTensorFormat_t,
        CUDNN_TENSOR_NCHW,        # 0, /* row major (wStride = 1, hStride = w) */
        CUDNN_TENSOR_NHWC,        # 1, /* feature maps interleaved ( cStride = 1 )*/
        CUDNN_TENSOR_NCHW_VECT_C, # 2, /* each image point is vector of element of C, vector length in data type */
    handle


function batchnorm(
    x::GPUVal, xmean::GPUVal, xvar::GPUVal, bias::GPUVal, scale::GPUVal;
    out = similar(x),
    training = Knet.training(),
    epsilon = 1e-5,
    momentum = 0.9,
    mode = nothing,
    format = nothing,
    savedMean = nothing,
    savedVar = nothing,
    workspace = nothing,
    reserveSpace = nothing,
    dx = Ref{Any}(nothing),
    dscale = Ref{Any}(nothing),
    dbias = Ref{Any}(nothing),
    o...)
    @assert size(xmean) == size(xvar) == size(bias) == size(scale)
    n = ndims(x)
    if size(xmean) == ntuple(i->(i===n-1 ? size(x,i) : 1), n)
        mode === nothing ? mode = CUDNN_NORM_PER_CHANNEL : @assert mode === CUDNN_NORM_PER_CHANNEL
        format === nothing ? format = CUDNN_TENSOR_NCHW : @assert format === CUDNN_TENSOR_NCHW
    elseif size(xmean) == ntuple(i->(i===1 ? size(x,i) : 1), n)
        mode === nothing ? mode = CUDNN_NORM_PER_CHANNEL : @assert mode === CUDNN_NORM_PER_CHANNEL
        format === nothing ? format = CUDNN_TENSOR_NHWC : @assert format === CUDNN_TENSOR_NHWC
    elseif size(xmean) == ntuple(i->(i===n ? 1 : size(x,i)), n)
        mode === nothing ? mode = CUDNN_NORM_PER_ACTIVATION : @assert mode === CUDNN_NORM_PER_ACTIVATION
        format === nothing ? format = CUDNN_TENSOR_NCHW : @assert format === CUDNN_TENSOR_NCHW
    else
        error("Unsupported batchnorm size x=$(size(x)) m=$(size(m))")
    end
    cudnnNormalizationForward!(
        out, x, xmean, xvar, bias, scale;
        training, mode, format, epsilon, exponentialAverageFactor=1-momentum,
        savedMean, savedInvVariance=savedVar,
        workspace, reserveSpace,
        dx, dscale, dbias)
end

