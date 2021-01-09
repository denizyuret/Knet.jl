import CUDA.CUDNN: cudnnNormalizationForwardAD
import AutoGrad: forw
using CUDA.CUDNN: handle, CU_NULL, cudnnNormalizationBackward, cudnnNormalizationForwardTraining, cudnnNormalizationForwardInference, cudnnGetNormalizationForwardTrainingWorkspaceSize, cudnnGetNormalizationTrainingReserveSpaceSize, cudnnTempSpace, scalingParameter
using AutoGrad: AutoGrad, @primitive1, forwargs, recording, Result, value


@primitive1((cudnnNormalizationForwardAD(x, scale, bias, z; training, mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready), dy, _y),
            (dready[] || cudnnNormalizationBack(dy, x, scale, bias, z; training, mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready); dx[]),
            (dready[] || cudnnNormalizationBack(dy, x, scale, bias, z; training, mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready); dscale[]),
            (dready[] || cudnnNormalizationBack(dy, x, scale, bias, z; training, mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready); dbias[]),
            (dready[] || cudnnNormalizationBack(dy, x, scale, bias, z; training, mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready); dz[]))


function cudnnNormalizationBack(dy, x, scale, bias, z; training, mean, variance, y, mode, normOps, algo, alpha, beta, epsilon, groupCnt, exponentialAverageFactor, savedMean, savedInvVariance, activationDesc, xDesc, yDesc, zDesc, normScaleBiasDesc, normMeanVarDesc, workspace, reserveSpace, dx, dscale, dbias, dz, dready)
    # Make sure backward gets called only once
    dready[] && return
    dready[] = true
    (x, scale, bias, z) = value.((x, scale, bias, z))
    # Allocate gradient buffers if necessary
    !isassigned(dx) || dx[]===nothing ? dx[] = similar(x) : @assert issimilar(x, dx[])
    !isassigned(dscale) || dscale[]===nothing ? dscale[] = similar(scale) : @assert issimilar(scale, dscale[])
    !isassigned(dbias) || dbias[]===nothing ? dbias[]  = similar(bias) : @assert issimilar(bias, dbias[])
    z === nothing ? dz[] = nothing : !isassigned(dz) || dz[]===nothing ? dz[] = zero(z) : (@assert issimilar(z, dz[]); dz[] .= 0) # z may not be used, dz may not be modified, should be zeroed
    s0,s1 = scalingParameter(eltype(x),0),scalingParameter(eltype(x),1)
    cudnnNormalizationBackward(handle(), mode, normOps, algo, alpha, s0, alpha, s0, xDesc, x, yDesc, y, yDesc, dy, something(zDesc,C_NULL), something(dz[],CU_NULL), xDesc, dx[], normScaleBiasDesc, scale, bias, dscale[], dbias[], epsilon, something(normMeanVarDesc,C_NULL), something(savedMean,CU_NULL), something(savedInvVariance,CU_NULL), something(activationDesc,C_NULL), something(workspace,CU_NULL), sizeof(workspace), something(reserveSpace,CU_NULL), sizeof(reserveSpace), groupCnt)
end


# CUDNN API controls training vs inference mode with a simple keyword argument.
# In Knet things are a bit more complicated: whether or not we are in @diff mode and we have Value args enter the picture.
# With Value args the AutoGrad.forw method is called regardless of the kwarg / @diff mode.
# The training keyword argument defaults to Knet.training() in Knet API, so by default @diff=training, but the user can override this.
# We have 8 cases to consider for Value, training, @diff:
# When there are no Value arguments, i.e. scale/bias not Params and x is not a Param or Result:
# cudnnNormalizationForwardAD is called, @diff=false, training=false: AD calls Inference: ok
# cudnnNormalizationForwardAD is called, @diff=false, training=true: AD calls Training: user is forcing training mode
# cudnnNormalizationForwardAD is called, @diff=true, training=false: AD calls Inference: user is forcing inference mode
# cudnnNormalizationForwardAD is called, @diff=true, training=true: AD calls Training: ok (but no Value args no back pass)
# When there are some Value arguments:
# forw is called, @diff=false, training=false: Standard test mode. forw->AD->Inference: ok
# forw is called, @diff=false, training=true: forw->AD->Training: We are not taking gradients but user is forcing training mode.
# forw is called, @diff=true, training=false: forw->AD->Inference: User wants to force inference mode?
# forw is called, @diff=true, training=true: Standard training mode. forw->AD->Training: ok

