export BatchNorm
import Knet
using Knet.Ops21
using Knet.KnetArrays: DevArray
using AutoGrad: Param, value
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


"""
    b = BatchNorm(; kwargs...)
    b(x; training)

Return batch normalization applied to `x`:

    b.bias + b.scale * (x - mean(x; b.dims)) / sqrt(b.epsilon + var(x; b.dims))  # training
    b.bias + b.scale * (x - b.mean) / sqrt(b.epsilon + b.var)                    # inference

During training, the actual mean/var of the batch are used in the normalization and to
update the running mean/var kept in `b.mean` and `b.var`. During inference, `b.mean` and
`b.var` are used in the normalization calculation. The training mode defaults to
`Knet.training()` but can be overriden by the `training` keyword argument.  `b.bias` and
`b.scale` are trainable parameters, corresponding to beta and gamma in the original
paper. The common size of bias/scale/mean/var can be:

    (1,1,C,1) if size(x)==(W,H,C,N) and b.dims==(1,2,4) (default, NCHW, per-channel)
    (C,1,1,1) if size(x)==(C,W,H,N) and b.dims==(2,3,4) (NHWC tensor format)
    (W,H,C,1) if size(x)==(W,H,C,N) and b.dims==4       (per-activation mode)

Keyword arguments for the constructor:
* `dims`: mean/var is computed over the given dimensions, by default (1,2,4) for 4D and (1,2,3,5) for 5D
* `mean = fill!(similar(x,bsize),0)`: mean tensor where `bsize=size(mean(x; b.dims))`
* `var = fill!(similar(x,bsize),1)`: variance tensor
* `bias = Param(fill!(similar(x,bsize),0))`: bias parameter
* `scale = Param(fill!(similar(x,bsize),1))`: scale parameter
* `epsilon = 1e-5`: epsilon value used in the normalization formula
* `momentum = 0.9`: momentum for the moving average, e.g. `b.mean = b.mean*momentum + mean(x; b.dims)*(1-momentum)`

Reference: Batch Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift, S. Ioffe, C. Szegedy, 2015.

"""
mutable struct BatchNorm
    dims
    mean
    var
    bias
    scale
    epsilon::Float64
    momentum::Float64
    format::cudnnTensorFormat_t
    mode::cudnnNormMode_t
    savedMean
    savedVar
    workspace
    reserveSpace
    dx::Ref{Any}
    dscale::Ref{Any}
    dbias::Ref{Any}
end


function BatchNorm(
    ;
    dims = nothing,
    mean = nothing,
    var = nothing,
    bias = nothing,
    scale = nothing,
    epsilon::Real = Cdouble(1e-5),
    momentum::Real = Cdouble(0.9),
)
    format = CUDNN_TENSOR_NCHW
    mode = CUDNN_NORM_PER_CHANNEL
    savedMean = nothing
    savedVar = nothing
    workspace = nothing
    reserveSpace = nothing
    dx = Ref{Any}(nothing)
    dscale = Ref{Any}(nothing)
    dbias = Ref{Any}(nothing)
    BatchNorm(dims, mean, var, bias, scale, epsilon, momentum, format, mode, savedMean, savedVar, workspace, reserveSpace, dx, dscale, dbias)
end


# Some fields can only be initialized after seeing the first input x

function initBatchNorm(b::BatchNorm, x; training)
    n = ndims(x)
    if b.dims === nothing || b.dims === (1:(n-2)...,n)
        b.mode, b.format, bsize = CUDNN_NORM_PER_CHANNEL, CUDNN_TENSOR_NCHW, ntuple(i->(i===n-1 ? size(x,i) : 1), n)
    elseif b.dims === (2:n...,)
        b.mode, b.format, bsize = CUDNN_NORM_PER_CHANNEL, CUDNN_TENSOR_NHWC, ntuple(i->(i===1 ? size(x,i) : 1), n)
    elseif length(b.dims) === 1 && b.dims[1] === n
        b.mode, b.format, bsize = CUDNN_NORM_PER_ACTIVATION, CUDNN_TENSOR_NCHW, ntuple(i->(i===n ? 1 : size(x,i)), n)
    else
        error("x=$(size(x)) dims=$dims not supported")
    end
    issimilar(u,v,s=size(v))=(typeof(value(u)) === typeof(value(v)) && size(u) === s)
    b.mean === nothing ? b.mean = fill!(similar(x, bsize), 0) : @assert issimilar(b.mean, x, bsize)
    b.var === nothing ? b.var = fill!(similar(x, bsize), 1) : @assert issimilar(b.var, x, bsize)
    b.bias === nothing ? b.bias = Param(fill!(similar(x, bsize), 0)) : @assert issimilar(b.bias, x, bsize)
    b.scale === nothing ? b.scale = Param(fill!(similar(x, bsize), 1)) : @assert issimilar(b.scale, x, bsize)
    if training && x isa DevArray
        b.savedMean === nothing ? b.savedMean = similar(x, bsize) : @assert issimilar(b.savedMean, x, bsize)
        b.savedVar === nothing ? b.savedVar = similar(x, bsize) : @assert issimilar(b.savedVar, x, bsize)
        workspaceSize, reserveSpaceSize = cudnnNormalizationTempSpaceSizes(b, x)
        if sizeof(b.reserveSpace) < reserveSpaceSize; b.reserveSpace = cudnnTempSpace(reserveSpaceSize); end
        if sizeof(b.workspace) < workspaceSize; b.workspace = cudnnTempSpace(workspaceSize); end
        b.dx[] === nothing ? b.dx[] = similar(x) : @assert issimilar(x, b.dx[])
        b.dscale[] === nothing ? b.dscale[] = similar(b.scale) : @assert issimilar(b.scale, b.dscale[])
        b.dbias[] === nothing ? b.dbias[] = similar(b.bias) : @assert issimilar(b.bias, b.dbias[])
    end
end


function cudnnNormalizationTempSpaceSizes(b::BatchNorm, x)
    workspaceSize, reserveSpaceSize = Ref{Csize_t}(0), Ref{Csize_t}(0)
    xDesc, bDesc = (u->cudnnTensorDescriptor(value(u); format=b.format)).((x, b.mean))
    mode, normOps, algo, groupCnt = b.mode, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD, 1
    cudnnGetNormalizationForwardTrainingWorkspaceSize(handle(), mode, normOps, algo, xDesc, C_NULL, xDesc, bDesc, C_NULL, bDesc, workspaceSize, groupCnt)
    cudnnGetNormalizationTrainingReserveSpaceSize(handle(), mode, normOps, algo, C_NULL, xDesc, reserveSpaceSize, groupCnt)
    workspaceSize[], reserveSpaceSize[]
end


function (b::BatchNorm)(x; training=Knet.training())
    initBatchNorm(b, x; training)
    batchnorm(x, b.mean, b.var, b.bias, b.scale; training,
              b.epsilon, b.momentum, b.mode, b.format,
              b.savedMean, b.savedVar, b.workspace, b.reserveSpace,
              b.dx, b.dscale, b.dbias)
end


