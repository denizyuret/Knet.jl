export batchnorm
using Statistics
using AutoGrad: value
import Knet

"""
    batchnorm(x, xmean, xvar, bias, scale; epsilon, momentum, training)

Return batch normalization applied to `x`:

    ((x .- mean(x; dims)) ./ sqrt.(epsilon .+ var(x; dims))) .* scale .+ bias  # training
    ((x .- xmean) ./ sqrt.(epsilon .+ xvar)) .* scale .+ bias                  # inference

During inference, the xmean/xvar arguments are used in the normalization calculation. During
training, the actual mean/var of the batch are used in the normalization and the xmean/xvar
arguments are updated in-place with their moving average. The behavior can be controlled
with the `training` keyword argument, which defaults to `Knet.training()`. bias/scale are
trainable parameters, corresponding to beta and gamma in the original paper. The `momentum`
keyword argument is used in the moving average calculation:

    xmean .= xmean * momentum + mean(x; dims) * (1-momentum)
    xvar .= xvar * momentum + var(x; dims) * (1-momentum)

The relation between the common size of bias/scale/mean/var and the dims value in the
formula can be:

    (1,1,C,1) if size(x)==(W,H,C,N) and dims==(1,2,4) (default, NCHW, per-channel)
    (C,1,1,1) if size(x)==(C,W,H,N) and dims==(2,3,4) (NHWC tensor format)
    (W,H,C,1) if size(x)==(W,H,C,N) and dims==4       (per-activation mode)

Keyword arguments:
* `epsilon=1e-5`
* `momentum=0.9`
* `training=Knet.training()`

Reference: Batch Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift, S. Ioffe, C. Szegedy, 2015.

Note: PyTorch uses momentum to mean (1-momentum), the Knet use is consistent with TensorFlow and ONNX.

Note: To be consistent with the CUDNN implementation `var` is called with `corrected=false`.
"""
function batchnorm(x, xmean, xvar, bias, scale; epsilon=1e-5, momentum=0.9, training=Knet.training(), o...)
    epsilon, momentum = eltype(x).((epsilon, momentum))
    if training
        dims = findall(size(xmean) .== 1)
        m = mean(x; dims)
        v = var(x; dims, mean=m, corrected=false)
        # x and therefore m,v are AutoGrad.Value, xmean/xvar are regular arrays
        xmean .= xmean * momentum + value(m) * (1-momentum)
        xvar .= xvar * momentum + value(v) * (1-momentum)
        y = ((x .- m) ./ sqrt.(epsilon .+ v)) .* scale .+ bias
    else
        y = ((x .- xmean) ./ sqrt.(epsilon .+ xvar)) .* scale .+ bias
    end
    return y
end
