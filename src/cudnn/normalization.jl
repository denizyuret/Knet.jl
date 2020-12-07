import CUDA.CUDNN: cudnnNormalizationForwardAutoGrad
import AutoGrad: forw
using CUDA.CUDNN: handle, CU_NULL, cudnnNormalizationBackward, cudnnNormalizationForwardTraining, cudnnNormalizationForwardInference, cudnnGetNormalizationForwardTrainingWorkspaceSize, cudnnGetNormalizationTrainingReserveSpaceSize, cudnnTempSpace, scalingParameter
using AutoGrad: AutoGrad, @primitive1, forwargs, recording, Result, value


@primitive1((cudnnNormalizationForwardAutoGrad(x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz), dy, _y),
            (dx[] === nothing && cudnnNormalizationBack(dy, x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz); dx[]),
            (dscale[] === nothing && cudnnNormalizationBack(dy, x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz); dscale[]),
            (dbias[] === nothing && cudnnNormalizationBack(dy, x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz); dbias[]),
            (dz[] === nothing && cudnnNormalizationBack(dy, x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz); dz[]))


function cudnnNormalizationBack(dy, x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz)
    (x, scale, bias, z) = value.((x, scale, bias, z))
    dx[], dscale[], dbias[] = similar.((x, scale, bias))
    dz[] = (z === CU_NULL ? z : similar(z))
    (alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff) = (a->scalingParameter(eltype(x),a)).((1,0,1,0))
    @info "calling backward";
    cudnnNormalizationBackward(handle(), mode, normOps, algo, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, yDesc, y, yDesc, dy, zDesc, dz[], xDesc, dx[], normScaleBiasDesc, scale, bias, dscale[], dbias[], epsilon, normMeanVarDesc, something(savedMean[],CU_NULL), something(savedInvVariance[],CU_NULL), activationDesc, something(workspace[],CU_NULL), sizeof(workspace[]), something(reserveSpace[],CU_NULL), sizeof(reserveSpace[]), groupCnt)
end


# This is where we specialize cudnnNormalizationForwardAutoGrad to call the Training method.
# forw is called when there are Tracked (Param or Result) arguments: either during training or during inference with Param args.
function forw(f::typeof(cudnnNormalizationForwardAutoGrad), x, scale, bias, z; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz)
    args = (x, scale, bias, z)
    (f, nobcast, novalue) = forwargs(f, args)
    @assert nobcast === args    # we shouldn't need to handle broadcasting
    @assert novalue !== args    # there should be some tracked args for forw to be called (maybe Params)
    x, scale, bias, z = novalue
    if recording()              # we are taking gradients
        savedMean[], savedInvVariance[] = similar.((mean, variance))
        workspaceSize = Ref{Csize_t}(0)
        cudnnGetNormalizationForwardTrainingWorkspaceSize(handle(), mode, normOps, algo, xDesc, zDesc, yDesc, normScaleBiasDesc, activationDesc, normMeanVarDesc, workspaceSize, groupCnt)
        workspace[] = cudnnTempSpace(workspaceSize[])
        reserveSpaceSize = Ref{Csize_t}(0)
        cudnnGetNormalizationTrainingReserveSpaceSize(handle(), mode, normOps, algo, activationDesc, xDesc, reserveSpaceSize, groupCnt)
        reserveSpace[] = cudnnTempSpace(reserveSpaceSize[])
        @info "calling forw->Training"
        cudnnNormalizationForwardTraining(handle(), mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, scale, bias, exponentialAverageFactor, normMeanVarDesc, something(mean,CU_NULL), something(variance,CU_NULL), epsilon, something(savedMean[],CU_NULL), something(savedInvVariance[],CU_NULL), activationDesc, zDesc, z, yDesc, y, something(workspace[],CU_NULL), sizeof(workspace[]), something(reserveSpace[],CU_NULL), sizeof(reserveSpace[]), groupCnt)
        y = Result(y, f, args, (; mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz))
    else                        # we are not taking gradients
        @info "calling forw->Inference"
        cudnnNormalizationForwardInference(handle(), mode, normOps, algo, alpha, beta, xDesc, x, normScaleBiasDesc, scale, bias, normMeanVarDesc, something(mean,CU_NULL), something(variance,CU_NULL), zDesc, z, activationDesc, yDesc, y, epsilon, groupCnt)
    end
    return y
end
