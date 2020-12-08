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
    if dx[] === nothing; dx[] = similar(x); end; @assert issimilar(x, dx[])
    if dscale[] === nothing; dscale[] = similar(scale); end; @assert issimilar(scale, dscale[])
    if dbias[] === nothing; dbias[] = similar(bias); end; @assert issimilar(bias, dbias[])
    if z !== nothing && dz[] === nothing; dz[] = similar(z); end; @assert issimilar(z, dz[])
    (alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff) = (a->scalingParameter(eltype(x),a)).((1,0,1,0))
    cudnnNormalizationBackward(handle(), mode, normOps, algo, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, yDesc, y, yDesc, dy, something(zDesc,C_NULL), something(dz[],CU_NULL), xDesc, dx[], normScaleBiasDesc, scale, bias, dscale[], dbias[], epsilon, normMeanVarDesc, something(savedMean[],CU_NULL), something(savedInvVariance[],CU_NULL), activationDesc, something(workspace[],CU_NULL), sizeof(workspace[]), something(reserveSpace[],CU_NULL), sizeof(reserveSpace[]), groupCnt)
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
        if savedMean[] === nothing; savedMean[] = similar(mean); end; @assert issimilar(mean, savedMean[])
        if savedInvVariance[] === nothing; savedInvVariance[] = similar(variance); end; @assert issimilar(variance, savedInvVariance[])
        workspaceSize, reserveSpaceSize = cudnnNormalizationTempSpaceSizes(mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, workspaceSize, groupCnt)
        if reserveSpaceSize > 0 && reserveSpace === nothing; reserveSpace = cudnnTempSpace(reserveSpaceSize); end
        @assert sizeof(reserveSpace) >= reserveSpaceSize  "reserveSpace should be at least $(reserveSpaceSize) bytes"
        if workspaceSize > 0 && workspace === nothing; workspace = cudnnTempSpace(workspaceSize); end
        @assert sizeof(workspace) >= workspaceSize  "workspace should be at least $(workspaceSize) bytes"
        cudnnNormalizationForwardTraining(handle(), mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, scale, bias, exponentialAverageFactor, normMeanVarDesc, something(mean,CU_NULL), something(variance,CU_NULL), epsilon, something(savedMean[],CU_NULL), something(savedInvVariance[],CU_NULL), activationDesc, zDesc, z, yDesc, y, something(workspace,CU_NULL), sizeof(workspace), something(reserveSpace,CU_NULL), sizeof(reserveSpace), groupCnt)
        y = Result(y, f, args, (; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready))
    else                        # we are not taking gradients
        cudnnNormalizationForwardInference(handle(), mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, scale, bias, normMeanVarDesc, something(mean,CU_NULL), something(variance,CU_NULL), zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt)
    end
    return y
end


# Helper functions
function cudnnNormalizationTempSpaceSizes(mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, workspaceSize, groupCnt)
    workspaceSize = Ref{Csize_t}(0); reserveSpaceSize = Ref{Csize_t}(0)
    cudnnGetNormalizationForwardTrainingWorkspaceSize(handle(), mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, workspaceSize, groupCnt)
    cudnnGetNormalizationTrainingReserveSpaceSize(handle(), mode, normOps, algo, activationDesc, xDesc, reserveSpaceSize, groupCnt)
    workspaceSize[], reserveSpaceSize[]
end
