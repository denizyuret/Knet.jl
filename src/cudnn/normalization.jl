import CUDA.CUDNN: cudnnNormalizationForwardAutoGrad
import AutoGrad: forw
using CUDA.CUDNN: handle, CU_NULL, cudnnNormalizationBackward, cudnnNormalizationForwardTraining, cudnnNormalizationForwardInference, cudnnGetNormalizationForwardTrainingWorkspaceSize, cudnnGetNormalizationTrainingReserveSpaceSize, cudnnTempSpace, scalingParameter
using AutoGrad: AutoGrad, @primitive1, forwargs, recording, Result, value


@primitive1((cudnnNormalizationForwardAutoGrad(x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready), dy, _y),
            (dready[] || cudnnNormalizationBack(dy, x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready); dx[]),
            (dready[] || cudnnNormalizationBack(dy, x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready); dscale[]),
            (dready[] || cudnnNormalizationBack(dy, x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready); dbias[]),
            (dready[] || cudnnNormalizationBack(dy, x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready); dz[]))


function cudnnNormalizationBack(dy, x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready)
    # Make sure backward gets called only once
    dready[] && return
    dready[] = true
    (x, scale, bias, z) = value.((x, scale, bias, z))
    # Allocate gradient buffers if necessary
    !isassigned(dx) ?     dx[]     = similar(x) :     @assert issimilar(x, dx[])
    !isassigned(dscale) ? dscale[] = similar(scale) : @assert issimilar(scale, dscale[])
    !isassigned(dbias) ?  dbias[]  = similar(bias) :  @assert issimilar(bias, dbias[])
    z === nothing ? dz[] = nothing : !isassigned(dz) ? dz[] = zero(z) : (@assert issimilar(z, dz[]); dz[] .= 0) # z may not be used, dz may not be modified, should be zeroed
    s0,s1 = scalingParameter(eltype(x),0),scalingParameter(eltype(x),1)
    cudnnNormalizationBackward(handle(), mode, normOps, algo, alpha, s0, alpha, s0, xDesc, x, yDesc, y, yDesc, dy, something(zDesc,C_NULL), something(dz[],CU_NULL), xDesc, dx[], normScaleBiasDesc, scale, bias, dscale[], dbias[], epsilon, something(normMeanVarDesc,C_NULL), something(savedMean,CU_NULL), something(savedInvVariance,CU_NULL), something(activationDesc,C_NULL), something(workspace,CU_NULL), sizeof(workspace), something(reserveSpace,CU_NULL), sizeof(reserveSpace), groupCnt)
end


# This is where we specialize cudnnNormalizationForwardAutoGrad to call the Training method.
# forw is called when there are Tracked (Param or Result) arguments: either during training or during inference with Param args.
function forw(f::typeof(cudnnNormalizationForwardAutoGrad), x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready)
    args = (x, scale, bias, z)
    (f, nobcast, novalue) = forwargs(f, args)
    @assert nobcast === args    # we shouldn't need to handle broadcasting
    @assert novalue !== args    # there should be some tracked args for forw to be called (maybe Params)
    x, scale, bias, z = novalue
    if recording()              # we are taking gradients
        mean === nothing ? savedMean = nothing : savedMean === nothing ? savedMean = similar(mean) : @assert issimilar(mean, savedMean)
        variance === nothing ? savedInvVariance = nothing : savedInvVariance === nothing ? savedInvVariance = similar(variance) : @assert issimilar(variance, savedInvVariance)
        workspaceSize, reserveSpaceSize = cudnnNormalizationTempSpaceSizes(mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, groupCnt)
        if reserveSpaceSize > 0 && reserveSpace === nothing; reserveSpace = cudnnTempSpace(reserveSpaceSize); end
        @assert sizeof(reserveSpace) >= reserveSpaceSize  "reserveSpace should be at least $(reserveSpaceSize) bytes"
        if workspaceSize > 0 && workspace === nothing; workspace = cudnnTempSpace(workspaceSize); end
        @assert sizeof(workspace) >= workspaceSize  "workspace should be at least $(workspaceSize) bytes"
        cudnnNormalizationForwardTraining(handle(), mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, scale, bias, exponentialAverageFactor, something(normMeanVarDesc,C_NULL), something(mean,CU_NULL), something(variance,CU_NULL), epsilon, something(savedMean,CU_NULL), something(savedInvVariance,CU_NULL), something(activationDesc,C_NULL), something(zDesc,C_NULL), something(z,CU_NULL), yDesc, y, something(workspace,CU_NULL), sizeof(workspace), something(reserveSpace,CU_NULL), sizeof(reserveSpace), groupCnt)
        y = Result(y, f, args, (; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready))
    else                        # we are not taking gradients
        @assert mean !== nothing && variance !== nothing && normMeanVarDesc !== nothing "normalization mean and variance are required in inference mode."
        cudnnNormalizationForwardInference(handle(), mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, scale, bias, normMeanVarDesc, mean, variance, something(zDesc,C_NULL), something(z,CU_NULL), something(activationDesc,C_NULL), yDesc, y, epsilon, groupCnt)
    end
    return y
end


# Helper functions
function cudnnNormalizationTempSpaceSizes(mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, groupCnt)
    workspaceSize, reserveSpaceSize = Ref{Csize_t}(0), Ref{Csize_t}(0)
    cudnnGetNormalizationForwardTrainingWorkspaceSize(handle(), mode, normOps, algo, xDesc, something(zDesc,C_NULL), yDesc, normScaleBiasDesc, something(activationDesc,C_NULL), something(normMeanVarDesc,C_NULL), workspaceSize, groupCnt)
    cudnnGetNormalizationTrainingReserveSpaceSize(handle(), mode, normOps, algo, something(activationDesc,C_NULL), xDesc, reserveSpaceSize, groupCnt)
    workspaceSize[], reserveSpaceSize[]
end
