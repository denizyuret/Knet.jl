export BatchNorm
using AutoGrad: Param, value
using CUDA.CUDNN:
    cudnnNormalizationForward,
    cudnnNormalizationForward!,
    cudnnNormalizationForwardInference,
    cudnnNormalizationForwardTraining,
    cudnnNormalizationBackward,
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


"""
    b = BatchNorm(; kwargs...)
    b(x)

Return batch normalization applied to `x`:

    y = [b.bias + (b.scale * (x - b.mean)/sqrt(b.epsilon + b.variance)] 
    b.y = y * b.alpha + b.y * b.beta

Bias and scale are trainable parameters, mean and variance are modified to collect
statistics during training and treated as constants during inference. Note that during
inference the values given by mean and variance arguments are used in the formula whereas
during training the actual mean and variance of the minibatch are used in the formula: the
mean/variance fields are only used to collect statistics. In the original paper bias is
referred to as beta and scale as gamma (Batch Normalization: Accelerating Deep Network
Training by Reducing Internal Covariate Shift, S. Ioffe, C. Szegedy, 2015). `alpha` and
`beta` can be used to scale or blend the output.

Unless specified during construction, array fields of BatchNorm are initialized during the
first call based on the input `x` to inherit the right array type and size. In particular
the `bsize` used for mean, variance etc. is:

    (1,1,C,1) if mode==CUDNN_NORM_PER_CHANNEL and format==CUDNN_TENSOR_NCHW and size(x)==(W,H,C,N)
    (C,1,1,1) if mode==CUDNN_NORM_PER_CHANNEL and format==CUDNN_TENSOR_NHWC and size(x)==(C,W,H,N)
    (W,H,C,1) if mode==CUDNN_NORM_PER_ACTIVATION and format==CUDNN_TENSOR_NCHW and size(x)==(W,H,C,N)
    (C,W,H,1) if mode==CUDNN_NORM_PER_ACTIVATION and format==CUDNN_TENSOR_NHWC and size(x)==(C,W,H,N)

Keyword arguments for the constructor:
* `mean = fill!(similar(x,bsize),0)`: mean tensor
* `variance = fill!(similar(x,bsize),1)`: variance tensor
* `bias = Param(fill!(similar(x,bsize),0))`: bias parameter
* `scale = Param(fill!(similar(x,bsize),1))`: scale parameter
* `y = similar(x)`: optional storage for output tensor
* `mode::cudnnNormMode_t = CUDNN_NORM_PER_CHANNEL`: Per-channel layer is based on the paper. The other alternative is `CUDNN_NORM_PER_ACTIVATION`.
* `algo::cudnnNormAlgo_t = CUDNN_NORM_ALGO_STANDARD`: The other alternative, `CUDNN_NORM_ALGO_PERSIST`, triggers the new semi-persistent NHWC kernel when certain conditions are met (see cudnn docs).
* `epsilon = 1e-5`: epsilon value used in the normalization formula
* `exponentialAverageFactor = 0.1`: factor used in running mean/variance calculation: `runningMean = runningMean*(1-factor) + newMean*factor`
* `alpha = 1; beta = 0`: scaling parameters

"""
mutable struct BatchNorm
    # Inference parameters:
    y
    mean
    variance
    bias
    scale
    mode::cudnnNormMode_t
    normOps::cudnnNormOps_t
    algo::cudnnNormAlgo_t
    alpha::Float64
    beta::Float64
    epsilon::Float64
    groupCnt::Integer

    # Training-only parameters:
    exponentialAverageFactor::Float64
    savedMean
    savedInvVariance

    # Activation parameters:
    activationMode::cudnnActivationMode_t
    activationReluNanOpt::cudnnNanPropagation_t
    activationCoef::Float64
    activationDesc::Union{Nothing,cudnnActivationDescriptor}

    # Tensor descriptors:
    format::cudnnTensorFormat_t

    # Temporary space used in training:
    workspace
    reserveSpace
    dx::Ref{Any}
    dscale::Ref{Any}
    dbias::Ref{Any}
    dz::Ref{Any}
end


function BatchNorm(
    ;
    # Inference parameters:
    # x = nothing, # input
    # z = nothing, # for residual addition to the result of the normalization operation, prior to the activation
    y = nothing,
    mean = nothing,
    variance = nothing,
    bias = nothing,
    scale = nothing,
    mode::cudnnNormMode_t = CUDNN_NORM_PER_CHANNEL, # Per-channel layer is based on the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, S. Ioffe, C. Szegedy, 2015.
    normOps::cudnnNormOps_t = CUDNN_NORM_OPS_NORM,  # Currently CUDNN_NORM_OPS_NORM_ACTIVATION and CUDNN_NORM_OPS_NORM_ADD_ACTIVATION are only supported in the NHWC layout (training,backward), not supported (inference)
    algo::cudnnNormAlgo_t = CUDNN_NORM_ALGO_STANDARD, # trigger the new semi-persistent NHWC kernel when CUDNN_NORM_ALGO_PERSIST
    alpha::Real = 1,
    beta::Real = 0,
    epsilon::Real = Cdouble(1e-5), # Has to be >= 0. Should be the same in forward and backward functions.
    groupCnt::Integer = Cint(1),   # Place hold for future work, should be set to 1 now

    # Training-only parameters:
    exponentialAverageFactor::Real = Cdouble(0.1),
    savedMean = nothing, # Optionally save intermediate results from the forward pass here - can be reused to speed up backward pass. NULL if unused.
    savedInvVariance = nothing,

    # Activation parameters:
    activationMode::cudnnActivationMode_t = CUDNN_ACTIVATION_IDENTITY,
    activationReluNanOpt::cudnnNanPropagation_t = CUDNN_NOT_PROPAGATE_NAN,
    activationCoef::Real = 1,
    activationDesc::Union{Nothing,cudnnActivationDescriptor} = (normOps == CUDNN_NORM_OPS_NORM ? nothing : cudnnActivationDescriptor(activationMode, activationReluNanOpt, Cdouble(activationCoef))),

    # Tensor descriptors:
    format::cudnnTensorFormat_t = CUDNN_TENSOR_NCHW,

    # Temporary space used in training:
    workspace = nothing,
    reserveSpace = nothing,
    dx = Ref{Any}(nothing),
    dscale = Ref{Any}(nothing),
    dbias = Ref{Any}(nothing),
    dz = Ref{Any}(nothing),
)
    BatchNorm(y, mean, variance, bias, scale, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationMode, activationReluNanOpt, activationCoef, activationDesc, format, workspace, reserveSpace, dx, dscale, dbias, dz)
end


# Some fields can only be initialized after seeing the first input x

function initBatchNorm(b::BatchNorm, x, z)
    n = ndims(x)
    bsize = (b.mode === CUDNN_NORM_PER_ACTIVATION ? ntuple(i->(i===n ? 1 : size(x,i)), n) :
             b.format === CUDNN_TENSOR_NCHW ? ntuple(i->(i===n-1 ? size(x,i) : 1), n) :
             ntuple(i->(i===1 ? size(x,i) : 1), n))
    issimilar(u,v,s=size(v))=(typeof(value(u)) === typeof(value(v)) && size(u) === s)
    b.y === nothing ? b.y = similar(x) : @assert issimilar(b.y, x)
    b.mean === nothing ? b.mean = fill!(similar(x, bsize), 0) : @assert issimilar(b.mean, x, bsize)
    b.variance === nothing ? b.variance = fill!(similar(x, bsize), 1) : @assert issimilar(b.variance, x, bsize)
    b.bias === nothing ? b.bias = Param(fill!(similar(x, bsize), 0)) : @assert issimilar(b.bias, x, bsize)
    b.scale === nothing ? b.scale = Param(fill!(similar(x, bsize), 1)) : @assert issimilar(b.scale, x, bsize)
    if AutoGrad.recording()
        b.savedMean === nothing ? b.savedMean = similar(x, bsize) : @assert issimilar(b.savedMean, x, bsize)
        b.savedInvVariance === nothing ? b.savedInvVariance = similar(x, bsize) : @assert issimilar(b.savedInvVariance, x, bsize)
        xDesc, bDesc = (u->cudnnTensorDescriptor(value(u); b.format)).((x, b.mean))
        workspaceSize, reserveSpaceSize = Knet.CUDNN.cudnnNormalizationTempSpaceSizes(b.mode, b.normOps, b.algo, xDesc, xDesc, xDesc, bDesc, b.activationDesc, bDesc, b.groupCnt)
        if sizeof(b.reserveSpace) < reserveSpaceSize; b.reserveSpace = cudnnTempSpace(reserveSpaceSize); end
        if sizeof(b.workspace) < workspaceSize; b.workspace = cudnnTempSpace(workspaceSize); end
        b.dx[] === nothing ? b.dx[]     = similar(x) : @assert issimilar(x, b.dx[])
        b.dscale[] === nothing ? b.dscale[] = similar(b.scale) : @assert issimilar(b.scale, b.dscale[])
        b.dbias[] === nothing ? b.dbias[]  = similar(b.bias) : @assert issimilar(b.bias, b.dbias[])
        z === nothing ? b.dz[] = nothing : b.dz[] === nothing ? b.dz[] = zero(z) : (@assert issimilar(z, b.dz[]); b.dz[] .= 0) # z may not be used, dz may not be modified, should be zeroed
    end
end


function (b::BatchNorm)(x, z=nothing)
    initBatchNorm(b, x, z)
    cudnnNormalizationForward!(
        b.y, x, b.mean, b.variance, b.bias, b.scale; z,
        mode=b.mode, normOps=b.normOps, algo=b.algo, alpha=b.alpha, beta=b.beta,
        epsilon=b.epsilon, groupCnt=b.groupCnt, exponentialAverageFactor=b.exponentialAverageFactor,
        savedMean=b.savedMean, savedInvVariance=b.savedInvVariance, activationDesc=b.activationDesc,
        format=b.format, workspace=b.workspace, reserveSpace=b.reserveSpace,
        dx=b.dx, dscale=b.dscale, dbias=b.dbias, dz=b.dz)
end
