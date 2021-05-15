import Knet.Ops21: batchnorm
using AutoGrad: value

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
    x::GPUVal, mean_estimate::GPUVal, var_estimate::GPUVal, bias::GPUVal, scale::GPUVal;
    use_estimates = !Knet.training(),
    update = Knet.training() ? 0.1 : 0.0,
    epsilon = 1e-5,
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
    @assert size(mean_estimate) == size(var_estimate) == size(bias) == size(scale)
    n = ndims(x)
    if size(mean_estimate) == ntuple(i->(i===n-1 ? size(x,i) : 1), n)
        mode === nothing ? mode = CUDNN_NORM_PER_CHANNEL : @assert mode === CUDNN_NORM_PER_CHANNEL
        format === nothing ? format = CUDNN_TENSOR_NCHW : @assert format === CUDNN_TENSOR_NCHW
    elseif size(mean_estimate) == ntuple(i->(i===1 ? size(x,i) : 1), n)
        mode === nothing ? mode = CUDNN_NORM_PER_CHANNEL : @assert mode === CUDNN_NORM_PER_CHANNEL
        format === nothing ? format = CUDNN_TENSOR_NHWC : @assert format === CUDNN_TENSOR_NHWC
    elseif size(mean_estimate) == ntuple(i->(i===n ? 1 : size(x,i)), n)
        mode === nothing ? mode = CUDNN_NORM_PER_ACTIVATION : @assert mode === CUDNN_NORM_PER_ACTIVATION
        format === nothing ? format = CUDNN_TENSOR_NCHW : @assert format === CUDNN_TENSOR_NCHW
    else
        error("Unsupported batchnorm size x=$(size(x)) m=$(size(m))")
    end
    # default training  => update > 0, use_estimates=false, gradients calculated
    # default inference => update = 0, use_estimates=true,  gradients not calculated
    # Other combinations must be manually implemented
    kw = (; mode, format, epsilon, savedMean, savedInvVariance=savedVar, workspace, reserveSpace, dx, dscale, dbias)
    if Knet.training() && !use_estimates && update == 0
        cudnnNormalizationForward(x, nothing, nothing, bias, scale; training=true, exponentialAverageFactor=0, kw...)
    elseif Knet.training() && !use_estimates && update > 0
        (mean_estimate, var_estimate) = value.((mean_estimate, var_estimate))
        cudnnNormalizationForward(x, mean_estimate, var_estimate, bias, scale; training=true, exponentialAverageFactor=update, kw...)
    elseif Knet.training() && use_estimates && update == 0
        ((x .- mean_estimate) ./ sqrt.(epsilon .+ var_estimate)) .* scale .+ bias
    elseif Knet.training() && use_estimates && update > 0
        update_estimates!(x, mean_estimate, var_estimate, update)
        ((x .- mean_estimate) ./ sqrt.(epsilon .+ var_estimate)) .* scale .+ bias
    elseif !Knet.training() && !use_estimates && update == 0
        cudnnNormalizationForward(x, nothing, nothing, bias, scale; training=true, exponentialAverageFactor=0, kw...)
    elseif !Knet.training() && !use_estimates && update > 0
        (mean_estimate, var_estimate) = value.((mean_estimate, var_estimate))
        cudnnNormalizationForward(x, mean_estimate, var_estimate, bias, scale; training=true, exponentialAverageFactor=update, kw...)
    elseif !Knet.training() && use_estimates && update == 0
        (mean_estimate, var_estimate) = value.((mean_estimate, var_estimate))
        cudnnNormalizationForward(x, mean_estimate, var_estimate, bias, scale; training=false, kw...)
    elseif !Knet.training() && use_estimates && update > 0
        (mean_estimate, var_estimate) = value.((mean_estimate, var_estimate))
        update_estimates!(x, mean_estimate, var_estimate, update)
        cudnnNormalizationForward(x, mean_estimate, var_estimate, bias, scale; training=false, kw...)
    end
end

function update_estimates!(x, mean_estimate, var_estimate, update)
    (x, mean_estimate, var_estimate, update) = value.((x, mean_estimate, var_estimate, update))
    dims = findall(size(mean_estimate) .== 1)
    xmean = mean(x; dims)
    xvar  = var(x; dims, mean=xmean, corrected=false)
    update = eltype(x)(update)
    mean_estimate .= xmean * update + mean_estimate * (1-update)
    var_estimate  .= xvar  * update + var_estimate  * (1-update)
end
