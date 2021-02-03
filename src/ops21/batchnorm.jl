export batchnorm
using Statistics
using AutoGrad: value
import Knet

"""
    batchnorm(x, mean_estimate, var_estimate, bias, scale; 
              epsilon = 1e-5, 
              update = Knet.training() ? 0.1 : 0
              use_estimates = !Knet.training())

Return batch normalization applied to `x`:

    ((x .- mean(x; dims)) ./ sqrt.(epsilon .+ var(x; dims))) .* scale .+ bias  # use_estimates=false
    ((x .- mean_estimate) ./ sqrt.(epsilon .+ var_estimate)) .* scale .+ bias  # use_estimates=true

If `use_estimates=true`, the mean_estimate/var_estimate arguments are used in the
normalization calculation, otherwise the actual mean/var of the batch are used. To be
consistent with the CUDNN implementation `var` is called with `corrected=false`.  If `update
> 0`, the mean_estimate/var_estimate arguments will be updated using batch statistics:

    mean_estimate .= mean(x; dims) * update + mean_estimate * (1-update)
    var_estimate  .= var(x; dims)  * update + var_estimate  * (1-update)

By default `use_estimates=false, update > 0` during training and `use_estimates=true,
update=0` during inference.

Bias and scale are trainable parameters, corresponding to beta and gamma in the original
paper.

The relation between the common size of bias/scale/mean/var and the dims value in the
formula can be:

    (1,1,C,1) if size(x)==(W,H,C,N) and dims==(1,2,4) (default, NCHW, per-channel)
    (C,1,1,1) if size(x)==(C,W,H,N) and dims==(2,3,4) (NHWC tensor format)
    (W,H,C,1) if size(x)==(W,H,C,N) and dims==4       (per-activation mode)

Reference: Batch Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift, S. Ioffe, C. Szegedy, 2015.
"""
function batchnorm(
    x, mean_estimate, var_estimate, bias, scale;
    epsilon = 1e-5,
    update = Knet.training() ? 0.1 : 0.0,
    use_estimates = !Knet.training(),
    o...
)
    update,epsilon = eltype(x).((update,epsilon))
    if update > 0 || !use_estimates
        dims = findall(size(mean_estimate) .== 1)
        xmean = mean(x; dims)
        xvar  = var(x; dims, mean=xmean, corrected=false)
    end
    if update > 0
        # x and therefore xmean,xvar may be AutoGrad.Value, mean_estimate/var_estimate are regular arrays
        mean_estimate .= value(xmean) * update + mean_estimate * (1-update)
        var_estimate  .= value(xvar)  * update + var_estimate  * (1-update)
    end        
    if use_estimates
        y = ((x .- mean_estimate) ./ sqrt.(epsilon .+ var_estimate)) .* scale .+ bias
    else
        y = ((x .- xmean) ./ sqrt.(epsilon .+ xvar)) .* scale .+ bias
    end
    return y
end
