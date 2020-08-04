import ..Ops20: batchnorm4, batchnorm4_back
using ..Ops20: _wsize, _lazy_init!

const BN_MODE_SPATIAL = 1
const BN_MODE_ACTIVATION = 0
const CUDNN_BN_MIN_EPS = 1e-5

batchnorm4(g::KnetArray{T}, b::KnetArray{T}, x::KnetArray{T}; o...) where T = _batchnorm4(T,g,b,x;o...)
batchnorm4(g::CuArray{T}, b::CuArray{T}, x::CuArray{T}; o...) where T = _batchnorm4(T,g,b,x;o...)

# Only spatial mode is supported
# TODO: support per-activation mode
function _batchnorm4(T, g, b, x; 
                     training=Knet.training(),
                     cache=nothing,
                     moments=nothing,
                     eps=1e-5,
                     alpha=1, beta=0,
                     handle = CUDNN.handle(),
                     cache_verbose=false, #reporting cache uses
                     o...)
    y = similar(x)
    weight_size = _wsize(y)
    # TODO: implement other bn mode
    bnmode = BN_MODE_SPATIAL
    # epsilon fix
    if eps < CUDNN_BN_MIN_EPS
        eps = CUDNN_BN_MIN_EPS
        warn("eps ", eps,
             " is too small for cudnn, so it is set to ",
             CUDNN_BN_MIN_EPS)
    end
    # moments
    if moments !== nothing
        _lazy_init!(moments, x)
        running_mean = moments.mean
        running_var = moments.var
        momentum = moments.momentum
    else
        running_mean, running_var = CU_NULL, CU_NULL
        momentum = .1
    end
    # The training mode
    if training
        # Cache the mean and ivar for later
        if cache !== nothing
            mean = similar(x,weight_size)
            ivar = similar(x,weight_size)
        else
            mean = CU_NULL
            ivar = CU_NULL
        end
        bnmode = CUDNN.cudnnBatchNormMode_t(bnmode)
        CUDNN.cudnnBatchNormalizationForwardTraining(handle, bnmode, Ref(T(alpha)), Ref(T(beta)), TD(x), x, TD(y), y, TD(g), g, b, momentum, running_mean, running_var, eps, mean, ivar)

        # Cache the resulting mean and inverse variance
        if cache != nothing
            cache_verbose && info("mean and ivar data saved to cache")
            cache.mean = mean
            cache.ivar = ivar
        end
    else
        @assert (moments!==nothing) "You must provide moments for the test mode!"
        bnmode = CUDNN.cudnnBatchNormMode_t(bnmode)
        CUDNN.cudnnBatchNormalizationForwardInference(handle, bnmode, Ref(T(alpha)), Ref(T(beta)), TD(x), x, TD(y), y, TD(g), g, b, running_mean, running_var, eps)
    end
    return y
end

function batchnorm4(x::Union{KnetArray,CuArray};o...)
    # Dummy buffers
    #  (cudnn doesn't support bn w/o affine
    #  although it is used in many applications)
    g = oftype(x, ones(_wsize(x)...))
    b = oftype(x, zeros(_wsize(x)...))
    return batchnorm4(g, b, x; o...)
end

batchnorm4_back(g::Union{KnetArray{T},Nothing}, x::KnetArray{T}, dy::KnetArray{T}; o...) where T = _batchnorm4_back(T,g,x,dy;o...)
batchnorm4_back(g::Union{CuArray{T},Nothing}, x::CuArray{T}, dy::CuArray{T}; o...) where T = _batchnorm4_back(T,g,x,dy;o...)

function _batchnorm4_back(T, g, x, dy;
                          training=Knet.training(),
                          cache=nothing,
                          moments=nothing,
                          grad_cache_disabled=false,
                          eps=1e-5, alpha=1, beta=0,
                          dalpha=1, dbeta=0,
                          handle = CUDNN.handle(),
                          cache_verbose=false,
                          o...)
    if training
        dx = similar(x)
        weight_size = _wsize(dy)
        if g==nothing; g=oftype(x, ones(weight_size)); end
        dg = similar(x,weight_size)
        db = similar(x,weight_size)
        # TODO: support other modes
        bnmode = BN_MODE_SPATIAL
        if cache !== nothing # (Assume cache still exists)
            mean, ivar = cache.mean, cache.ivar
            cache_verbose && info("mean and ivar are fetched from the cache")
        else
            mean, ivar = CU_NULL, CU_NULL
        end
        bnmode = CUDNN.cudnnBatchNormMode_t(bnmode)
        CUDNN.cudnnBatchNormalizationBackward(handle, bnmode, Ref(T(alpha)), Ref(T(beta)), Ref(T(dalpha)), Ref(T(dbeta)), TD(x), x, TD(dy), dy, TD(dx), dx, TD(g), g, dg, db, eps, mean, ivar)

    else
        # At test mode, g .*( x ./ sqrt(var) - mean ./ sqrt(var)) .+ beta
        # is performed;
        # so the derivative dx = dy .* g. / sqrt(var + eps) since mean and var
        # are constants
        # Note: moments must exist since otherwise forward pass fails
        ivar = 1 ./ sqrt.(moments.var .+ eps)
        dx = (g !== nothing) ? (dy .* g .* ivar) : (dy .* ivar)
        if g !== nothing
            dg = sum(dy .* (x .- moments.mean) .* ivar, dims=_reddims(dy))
            db = sum(dy, dims=_reddims(dy))
        else
            dg, db = nothing, nothing
        end
    end
    return dg, db, dx
end
