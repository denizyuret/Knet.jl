using Knet, AutoGrad
using Knet: @cuda, Cptr, DT, TD, cudnnhandle
using AutoGrad: @primitive, @zerograd


const BN_MODE_SPATIAL = 1
const BN_MODE_ACTIVATION = 0
# Const min epsilon
const CUDNN_BN_MIN_EPS = 1e-5


# A black box ADT for storing the bn state
type BNCache
    mean
    ivar
    dx
    dg
    db
end

BNCache() = BNCache(nothing, nothing, nothing, nothing, nothing)


type BNMoments
    momentum::AbstractFloat
    mean
    var
    meaninit
    varinit
end

BNMoments(;momentum=0.1, meaninit=zeros, varinit=ones, o...) =
    BNMoments(momentum, nothing, nothing, meaninit, varinit)

BNMoments(momentum, mean, var) = BNMoments(momentum, mean, var, nothing, nothing)

# TODO: consider automatic type conversion
# TODO: other dimensionalities
function _lazy_init!(m::BNMoments, x)
    buf_size = (ndims(x) == 4) ? (1, 1, size(x, 3), 1) : (1, size(x,2))
    tx = typeof(x)
    ex = eltype(x)
    if m.mean == nothing
        m.mean = tx(m.meaninit(ex, buf_size...))
    end
    if m.var == nothing
        m.var = tx(m.varinit(ex, buf_size...))
    end
end


# Only spatial mode is supported
# TODO: support per-activation mode
# TODO: support 5d arrays
function batchnorm4{T}(g::KnetArray{T}, b::KnetArray{T}, x::KnetArray{T};
                       training=true,
                       cache=nothing,
                       moments=nothing,
                       eps=1e-5,
                       alpha=1, beta=0,
                       handle = cudnnhandle(),
                       cache_verbose=false, #reporting cache uses
                       o...)
    y = KnetArray{T}(size(x))
    weight_size = (1, 1, size(y, 3), 1)
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
        running_mean, running_var = C_NULL, C_NULL
        momentum = .1
    end
    # The training mode
    if training
        # Cache the mean and ivar for later
        if cache !== nothing
            mean = KnetArray{T}(weight_size)
            ivar = KnetArray{T}(weight_size)
        else
            mean = C_NULL
            ivar = C_NULL
        end
        @cuda(cudnn, cudnnBatchNormalizationForwardTraining,
              # Types
              (Cptr, UInt32,
               Ptr{T}, Ptr{T}, #alpha and beta
               Cptr, Ptr{T}, #xdesc and x
               Cptr, Ptr{T}, #ydesc and y
               Cptr, Ptr{T}, Ptr{T}, #desc, weight and bias
               Cdouble, Ptr{T}, Ptr{T}, #Decay factor, Running mean and Running var
               Cdouble, # eps
               Ptr{T}, Ptr{T}), #Cached mean and ivar
              # Actual Arguments
              handle, bnmode,
              Ref(T(alpha)), Ref(T(beta)),
              TD(x), x, #x
              TD(y), y, #y
              TD(g), g, b, #params
              momentum, running_mean, running_var,
              eps, mean, ivar) #end of @cuda
        # Cache the resulting mean and inverse variance
        if cache != nothing
            cache_verbose && info("mean and ivar data saved to cache")
            cache.mean = mean
            cache.ivar = ivar
        end
    else
        @assert (moments!==nothing) "You must provide moments for the test mode!"
        @cuda(cudnn, cudnnBatchNormalizationForwardInference,
              # Types
              (Cptr, UInt32,
               Ptr{T}, Ptr{T},
               Cptr, Ptr{T}, #x
               Cptr, Ptr{T}, #y
               Cptr, Ptr{T}, Ptr{T}, #params
               Ptr{T}, Ptr{T}, #rm and rf
               Cdouble),
              handle, bnmode,
              Ref(T(alpha)), Ref(T(beta)), #alpha and bete
              TD(x), x, # xdesc and x
              TD(y), y, #ydesc and y
              TD(g), g, b, #desc, scale and bias
              running_mean, running_var, #estimated stuff
              eps) #epsilon
    end
    return y
end

function batchnorm4{T}(x::KnetArray{T};o...)
    # Dummy buffers
    #  (cudnn doesn't support bn w/o affine
    #  although it is used in many applications)
    g = KnetArray{T}(ones(1,1,size(x,ndims(x)-1), 1))
    b = KnetArray{T}(zeros(1,1,size(x,ndims(x)-1), 1))
    return batchnorm4(g, b, x; o...)
end

function batchnorm4_back{T}(g::Union{KnetArray{T}, Void},
                            x::KnetArray{T}, dy::KnetArray{T};
                            training=true,
                            cache=nothing,
                            moments=nothing,
                            grad_cache_disabled=false,
                            eps=1e-5, alpha=1, beta=0,
                            dalpha=1, dbeta=0,
                            handle = cudnnhandle(),
                            cache_verbose=false,
                            o...)
    if training
        dx = KnetArray{T}(size(x))
        weight_size = (1, 1, size(dy, 3), 1)
        if g==nothing; g=KnetArray{T}(ones(weight_size)); end
        dg = KnetArray{T}(weight_size)
        db = KnetArray{T}(weight_size)
        # TODO: support other modes
        bnmode = BN_MODE_SPATIAL
        if cache !== nothing # (Assume cache still exists)
            mean, ivar = cache.mean, cache.ivar
            cache_verbose && info("mean and ivar are fetched from the cache")
        else
            mean, ivar = C_NULL, C_NULL
        end
        @cuda(cudnn, cudnnBatchNormalizationBackward,
              # C Types
              (Cptr, UInt32,
               Ptr{T}, Ptr{T}, #data difs
               Ptr{T}, Ptr{T}, #param difs
               Cptr, Ptr{T}, #x
               Cptr, Ptr{T}, #dy
               Cptr, Ptr{T}, #dx
               Cptr, Ptr{T}, Ptr{T}, Ptr{T}, #desc,g,dg,db
               Cdouble, Ptr{T}, Ptr{T}),
              # Actual arguments
              handle, bnmode,
              Ref(T(alpha)), Ref(T(beta)),
              Ref(T(dalpha)), Ref(T(dbeta)),
              TD(x), x, TD(dy), dy, TD(dx), dx,
              TD(g), g, dg, db,
              eps, mean, ivar)
       
    else
        # At test mode, g .*( x ./ sqrt(var) - mean ./ sqrt(var)) .+ beta
        # is performed;
        # so the derivative is dy .* g./sqrt(var + x) since mean and var
        # are constants
        # TODO: Test this operation
        # Note: moments must exist since otherwise forward pass fails
        ivar = 1 ./ sqrt.(moments.var .+ eps)
        dx = (g!==nothing) ? (dy .* g .* ivar) : (dy .* ivar)
        if g!==nothing
            dg = dy .* (x .- moments.mean) .* ivar
            db = sum(dy, (1,2,4))
        else
            dg, db = nothing, nothing
        end
    end
    return dg, db, dx
end


function batchnorm4g{T}(g::Union{KnetArray{T}, Array{T}},
                        x::Union{KnetArray{T}, Array{T}},
                        dy::Union{KnetArray{T}, Array{T}};
                        cache=nothing, o...)
    dg, db, dx = batchnorm4_back(g, x, dy; cache=cache, o...)
    if cache !== nothing
        cache.dx, cache.db = dx, db
    else
        warn("Calling batchnorm with affine without",
             " cache is not recommended for performance")
    end
    return dg
end

function batchnorm4b{T}(dy::Union{KnetArray{T}, Array{T}};
                        cache=nothing, o...)
    (cache == nothing || cache.db == nothing) && return sum(dy, (1,2,4))
    return cache.db
end

function batchnorm4x{T}(g::Union{KnetArray{T}, Array{T}},
                        x::Union{KnetArray{T}, Array{T}},
                        dy::Union{KnetArray{T}, Array{T}};
                        cache=nothing, o...)
    if cache !== nothing && cache.dx !== nothing
        return cache.dx
    end
    return batchnorm4_back(g, x, dy; cache=cache, o...)[3]
end

function batchnorm4x{T}(x::Union{KnetArray{T}, Array{T}},
                        dy::Union{KnetArray{T}, Array{T}}
                        ;o...)
    return batchnorm4_back(nothing, x, dy; o...)[3]
end


# CPU implementation
function _update_moments!(moments, mu, sigma2)
    moments === nothing && return
    moments.mean = moments.momentum .* mu .+
        (1 .- moments.momentum) .* moments.mean
    moments.var = moments.momentum .* sigma2 .+
        (1 .- moments.momentum) .* moments.var
end

function batchnorm4{T}(g::Array{T}, b::Array{T}, x::Array{T};
                       o...)
    xhat = batchnorm4(x; o...)
    return g .* xhat .+ b
end

function batchnorm4{T}(x::Array{T};
                       eps=1e-5, training=true,
                       cache=nothing, moments=nothing,
                       o...)
    eps = T(eps)
    if moments!==nothing
        _lazy_init!(moments, x)
    end
    if training
        mu = mean(x, (1,2,4))
        x_mu = x .- mean(x, (1,2,4))
        sigma2 = mean(x_mu .* x_mu, (1,2,4))
        ivar = 1 ./ sqrt.(sigma2 .+ eps)
        _update_moments!(moments, mu, sigma2)
        # Cache the resulting values
        if cache !== nothing
            cache.ivar = ivar
            cache.mean = mu
        end
    else #test mode
        @assert (moments!==nothing)  "Moments must be provided in test mode"
        x_mu = x .- moments.mean
        ivar = 1 ./ sqrt.(moments.var .+ eps)
    end
    return x_mu .* ivar
end

function _get_cache_data(cache, x, eps)
    if cache !== nothing
        mu = cache.mean
        x_mu = x .- mu
        ivar = cache.ivar
    else
        mu = mean(x, (1,2,4))
        x_mu = x .- mu
        ivar = 1 ./ sqrt.(mean(x_mu.*x_mu, (1,2,4)) .+ eps)
    end
    return x_mu, ivar
end

function batchnorm4_back{T}(g::Union{Array{T}, Void}, x::Array{T}, dy::Array{T};
                            eps=1e-5, training=true,
                            cache=nothing, moments=nothing,  o...)
    eps = T(eps)
    if training
        dims = (1, 2, 4)
        x_mu, ivar = _get_cache_data(cache, x, eps)
        # equations from the original paper
        if g !== nothing
            dg = sum(x_mu .* ivar .* dy, (1,2,4))
            db = sum(dy, (1,2,4))
            dy = g .* dy
        else
            dg, db = nothing, nothing
        end
        m = *(size(x, dims...)...)
        dsigma2 = -T(0.5) .* sum(dy .* x_mu .* ivar.^3, dims)
        dmu = sum(dy .* -ivar, dims) .-
            2dsigma2 .* sum(x_mu, dims) ./ m
        
        dx = dy .* ivar .+
            (dsigma2 .* 2x_mu .+ dmu) ./ m
    else #same reasoning with the gpu version
        ivar = 1 ./ sqrt.(moments.var .+ eps)
        dx = (g!==nothing) ? (dy .* g .* ivar) : (dy .* ivar)
        if g!==nothing
            dg = dy .* (x .- moments.mean) .* ivar
            db = sum(dy, (1,2,4))
        else
            dg, db = nothing, nothing
        end
    end
    return dg, db, dx
end


@primitive batchnorm4(x;o...),dy batchnorm4x(x, dy; o...)
@zerograd batchnorm4x(x, dy; o...)
# 4d with affine
@primitive batchnorm4(g, b, x; o...),dy batchnorm4g(g, x, dy; o...) batchnorm4b(dy; o...) batchnorm4x(g, x, dy; o...)
@zerograd batchnorm4x(g, x, dy; o...)
@zerograd batchnorm4g(g, x, dy; o...)
@zerograd batchnorm4b(dy; o...)


# Implement batchnorm2 using batchnorm4, with autograd

function batchnorm2(g, b, x; moments=nothing, o...)
    @inline _pad4(x) = reshape(x, (1,1,size(x,1,2)...))
    x = _pad4(x)
    args = (x,)
    if g !== nothing
        g = _pad4(g)
        b = _pad4(b)
        args = (g, b, x)
    end
    # process moments
    if moments !== nothing
        _lazy_init!(moments, x)    
        moments.mean = _pad4(moments.mean)
        moments.var = _pad4(moments.var)
    end
    y = mat(batchnorm4(args...;
                       moments=moments,
                       o...))
    if moments !== nothing
        moments.mean = mat(moments.mean)
        moments.var = mat(moments.var)
    end
    return y
end

batchnorm2(x;o...) = batchnorm2(nothing, nothing, x; o...)
