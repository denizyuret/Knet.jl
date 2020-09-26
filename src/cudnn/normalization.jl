using CUDA.CUDNN:
    cudnnNormalizationForwardInference,
    cudnnNormalizationForwardTraining,
    cudnnNormalizationBackward,
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



function cudnnNormalizationForwardWithDefaults(
    x;
    format::cudnnTensorFormat_t = CUDNN_TENSOR_NCHW,
    mode::cudnnNormMode_t = CUDNN_NORM_PER_CHANNEL, # Per-channel layer is based on the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, S. Ioffe, C. Szegedy, 2015.
    normOps::cudnnNormOps_t = CUDNN_NORM_OPS_NORM,  # Currently CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are only supported in the NHWC layout.
    algo::cudnnNormAlgo_t = CUDNN_NORM_ALGO_STANDARD, # trigger the new semi-persistent NHWC kernel when CUDNN_NORM_ALGO_PERSIST
    alpha::Real = 1, # alpha[0] = result blend factor
    beta::Real = 0,  # beta[0] = dest layer blend factor
    normScale::DevArray, # in the original paper bias is referred to as beta and scale as gamma
    normBias::DevArray,  # in the original paper bias is referred to as beta and scale as gamma
    exponentialAverageFactor::Real,
    resultRunningMean::DevArray,
    resultRunningVariance::DevArray,
    resultSaveMean::Union{Nothing,DevArray}, # Optionally save intermediate results from the forward pass here - can be reused to speed up backward pass. NULL if unused.
    resultSaveInvVariance::Union{Nothing,DevArray},
    epsilon::Real, # Has to be >= 0. Should be the same in forward and backward functions.
    activationMode::cudnnActivationMode_t = CUDNN_ACTIVATION_IDENTITY,
    activationReluNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    activationCoef::Real = 1,
    activationDesc::Union{Nothing,cudnnActivationDescriptor} = (normOps == CUDNN_NORM_OPS_NORM ? C_NULL : cudnnActivationDescriptor(activationMode, activationReluNanOpt, Cdouble(activationCoef))),
    y::DevArray = similar(x),
    z::Union{Nothing,DevArray} = nothing, # for residual addition to the result of the normalization operation, prior to the activation
    xDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(x; format),
    yDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(y; format),
    zDesc::Union{Nothing,cudnnTensorDescriptor} = cudnnTensorDescriptor(z; format),
    normScaleBiasDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(normScale; format),
    normMeanVarDesc::cudnnTensorDescriptor = cudnnTensorDescriptor(resultRunningMean; format),
    workspace::DevArray,
    workSpaceSizeInBytes::Integer,
    reserveSpace::DevArray,
    reserveSpaceSizeInBytes::Integer,
    groupCnt::Cint = 1 # Place hold for future work, should be set to 1 now
)

end
