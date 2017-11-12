using Knet, AutoGrad
using Knet: @cuda, Cptr, DT, TD, cudnnhandle
using AutoGrad: @primitive, @zerograd


# Batch Normalization Mode
const BN_MODE_SPATIAL = 1
const BN_MODE_ACTIVATION = 0

# Const min epsilon
const CUDNN_BN_MIN_EPS = 1e-5

#global batchnorm2, batchnorm2_back, batchnorm4, batchnorm4_back, bncache
#global BN_MODE_SPATIAL, BN_MODE_ACTIVATION, CUDNN_BN_MIN_EPS

#=The black box data structure for caching bn data=#
bncache() = Any[nothing, nothing, nothing, nothing]

# indices (usage cache[_mean()] etc.)
@inline _mean() = 1
@inline _ivar() = 2
@inline _dgamma() = 3
@inline _dbeta() = 4

# User may want to reset and/or reuse the bncache
# may also help garbage collection
# TODO: consider supporting overwrites
function reset_bncache!(cache)
    for i = 1:length(cache)
        cache[i] = nothing
    end
end

# Only spatial mode is supported
# TODO: support per-activation mode
# TODO: support 5d arrays
function batchnorm4{T}(x::KnetArray{T},
                      g::KnetArray{T}, b::KnetArray{T};
                      training=true, cache=nothing,
                      eps=1e-5, momentum=.1,
                      alpha=1, beta=0,
                      running_mean=nothing,
                      running_var=nothing,
                      handle = cudnnhandle(),
                      cache_verbose=false, #reporting cache uses etc
                      o...)
    y = KnetArray{T}(size(x))
    weight_size = (1, 1, size(y, 3), 1)
    bnmode = BN_MODE_SPATIAL
    # Dummy buffers
    #bnscale = KnetArray{T}(ones(weight_size))
    #bnbias = KnetArray{T}(zeros(weight_size))
    if eps < CUDNN_BN_MIN_EPS
        eps = CUDNN_BN_MIN_EPS
        warn("eps ", eps,
             " is too small for cudnn, so it is set to ",
             CUDNN_BN_MIN_EPS)
    end
    if training
        # Cache the mean and ivar for later
        if cache !== nothing
            mean = KnetArray{T}(weight_size)
            ivar = KnetArray{T}(weight_size)
        else
            mean = C_NULL
            ivar = C_NULL
        end
        if running_mean == nothing
            running_mean = C_NULL
        end
        if running_var == nothing
            running_var = C_NULL
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
              eps,
              mean, ivar) #end of @cuda
        # Cache the resulting mean and inverse variance
        if cache != nothing
            cache_verbose && info("mean and ivar data saved to cache")
            cache[_mean()] = mean
            cache[_ivar()] = ivar
        end
    else
        if (running_mean == nothing || running_var == nothing)
            error("You must provide running mean and running var for the test mode!")
        end
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
    return batchnorm4(x, g, b; o...)
end


function batchnorm4x{T}(x::KnetArray{T}, g::KnetArray{T},
                       dy::KnetArray{T};
                       training=true, cache=nothing,
                       grad_cache_disabled=false,
                       eps=1e-5, alpha=1, beta=0,
                       dalpha=1, dbeta=0,
                       handle = cudnnhandle(),
                       cache_verbose=false,
                       o...)
    
    dx = KnetArray{T}(size(x))
    weight_size = (1, 1, size(dy, 3), 1)
    dg = KnetArray{T}(weight_size)
    db = KnetArray{T}(weight_size)
    # bnmode
    bnmode = BN_MODE_SPATIAL
    if cache !== nothing # (Assume cache still exists)
        mean, ivar = cache[_mean()], cache[_ivar()]
        cache_verbose && info("mean and ivar are fetched from the cache")
    else
        mean, ivar = C_NULL, C_NULL
    end
    
    if training
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
        # Cache the derivatives for feature use
        if cache !== nothing && !grad_cache_disabled
            # Clear the mean and ivar for gc
            cache[_dgamma()] = dg
            cache[_dbeta()] = db
            cache_verbose && info("dg and db ara fetched from the cache")
        end
    else
        # At test mode, g .*( x ./ sqrt(var) - mean ./ sqrt(var)) .+ beta
        # is performed;
        # so the derivative is dy .* g./sqrt(var + x) since mean and var
        # are constants
        # TODO: debug this operation
        dx = dy .* (g ./ sqrt.(running_var .+ eps))
    end
    return dx
end

function batchnorm4x{T}(x::KnetArray{T}, dy::KnetArray{T};
                       o...)
    g = KnetArray{T}(ones(1,1,size(x,ndims(x)-1), 1))
    # disable the gradient caching since g is a dummy buffer
    # (I'm not sure it is really necessary)
    return batchnorm4x(x, g, dy; grad_cache_disabled=true, o...)
end

function batchnorm4g{T}(x::KnetArray{T}, dy::KnetArray{T};
                        eps=1e-5, cache=nothing, training=true,
                        running_mean=nothing, running_var=nothing,
                        o...)
    if training
        x_mu, ivar = nothing, nothing
        if cache !== nothing
            if cache[_dgamma()] !== nothing #shortcut for efficiency
                return cache[_dgamma()]
            end
            # assume mean and ivar are cached (since it occurs in forward pass)
            mu, ivar = cache[_mean()], cache[_ivar()]
            x_mu = x .- mu
        else
            mu = mean(x, (1,2,4))
            x_mu = x .- mu
            ivar =1 ./ sqrt.(mean(x_mu.^2, (1,2,4)) .+ eps)
        end
        xhat = x_mu .* ivar
    else
        xhat = (x .- running_mean) ./ sqrt.(running_var .+ eps)
    end
    return sum(dy .* xhat, (1,2,4))
end

function batchnorm4b{T}(dy::KnetArray{T};
                       cache=nothing,o...)
    if cache!==nothing && cache[_dbeta()] !== nothing
        return cache[_dbeta()]
    end
    return sum(dy, (1,2,4))
end

# CPU IMPLEMENTATIONS
function batchnorm4{T}(x::Array{T}, g::Array{T}, b::Array{T};
                      o...)
    xhat = batchnorm4(x)
    return g .* xhat .+ b
end

function batchnorm4{T}(x::Array{T};
                       eps=1e-5, training=true,
                       running_mean=nothing,
                       running_var=nothing,
                       cache=nothing, momentum=.1,
                       o...)
    if training
        mu = mean(x, (1,2,4))
        x_mu = x .- mean(x, (1,2,4))
        sigma2 = mean(x_mu.^2, (1,2,4))
        ivar = 1 ./ sqrt.(sigma2 .+ eps)
        if running_mean !== nothing
            copy!(running_mean, momentum .* mu .+ (1-momentum) .* running_mean)
        end
        if running_var !== nothing
            copy!(running_var, momentum .* sigma2 .+ (1-momentum) .* running_var)
        end
        # Cache the resulting values
        if cache !== nothing
            cache[_ivar()] = ivar
            cache[_mean()] = mu
        end
    else #test mode
        @assert (running_mean !== nothing && running_var !== nothing)
        x_mu = x .- running_mean
        ivar = 1 ./ sqrt(running_var .+ eps)
    end
    return x_mu .* ivar
end

function batchnorm4x{T}(x::Array{T}, g::Array{T}, dy::Array{T};o...)
    dxhat = g .* dy
    return batchnorm4x(x, dy;o...)
end

# backward recompute utility for cpu
function _get_cache_data(cache, x, eps)
    if cache !== nothing
        mu = cache[_mean()]
        x_mu = x .- mu
        ivar = cache[_ivar()]
    else
        mu = mean(x, (1,2,4))
        x_mu = x .- mu
        ivar = 1 ./ sqrt.(mean(x_mu.^2, (1,2,4)) .+ eps)
    end
    return x_mu, ivar
end

function batchnorm4x{T}(x::Array{T}, dy::Array{T};
                     eps=1e-5, training=true,
                     cache=nothing, running_var=nothing, 
                     o...)
    if training
        x_mu, ivar = _get_cache_data(cache, x, eps)
        # equations from the original paper
        m = *(size(x, 1,2,4)...)
        dsigma2 = -0.5sum(dy .* x_mu .* ivar.^3, (1,2,4))
        
        dmu = sum(dy .* -ivar, (1,2,4)) .+
            dsigma2 .* -2sum(x_mu, (1,2,4)) ./ m
        
        dx = dy .* ivar .+
            dsigma2 .* 2x_mu ./ m .+
            dmu ./ m
    else #same reasoning with the gpu version
        dx = dy .* (1 ./ sqrt.(running_var .+ eps))
    end
    return dx
end

function batchnorm4g{T}(x::Array{T}, dy::Array{T};
                        eps=1e-5, training=true,
                        running_mean=nothing,
                        running_var=nothing,
                        o...)
    if training
        x_mu, ivar = _get_cache_data(cache, x, eps)
        xhat = x_mu .* ivar
    else
        xhat = (x .- running_mean) ./ (running_var .+ eps)
    end
    return dy .* xhat
end

function batchnorm4b{T}(dy::Array{T};
                        o...)
    return sum(dy, (1,2,4))
end


# 2D batch norm implementation
@inline _pad4(x) = reshape(x, (1,1,size(x)...))

function batchnorm2{T}(x::Union{Array{T}, KnetArray{T}},
                       g::Union{Array{T}, KnetArray{T}},
                       b::Union{Array{T}, KnetArray{T}}
                       ;o...)
    g = _pad4(g)
    b = _pad4(b)
    x = _pad4(x)
    return mat(batchnorm4(x, g, b; o...))
end

function batchnorm2{T}(x::Union{Array{T}, KnetArray{T}};
                       o...)
    x = _pad4(x)
    return mat(batchnorm4(x;o...))
end

function batchnorm2x{T}(x::Union{Array{T}, KnetArray{T}},
                        g::Union{Array{T}, KnetArray{T}},
                        dy::Union{Array{T}, KnetArray{T}};
                        o...)
    dy = _pad4(dy)
    g = _pad4(g)
    x = _pad4(x)
    dx = batchnorm4x(x, g, dy;o...)
    return mat(dx)
end

function batchnorm2x{T}(x::Union{Array{T}, KnetArray{T}},
                        dy::Union{Array{T}, KnetArray{T}};
                        o...)
    dy = _pad4(dy)
    x = _pad4(x)
    dx = batchnorm4x(x, dy;o...)
    return mat(dx)
end

function batchnorm2g{T}(x::Union{Array{T}, KnetArray{T}},
                        dy::Union{Array{T}, KnetArray{T}};
                        o...)
    dy = _pad4(dy)
    x = _pad4(x)
    dg = batchnorm4g(x, dy;o...)
    return mat(dg)
end

function batchnorm2b{T}(dy::Union{Array{T}, KnetArray{T}};
                        o...)
    dy = _pad4(dy)
    return mat(batchnorm4b(dy; o...))
end


# 4d w/o affine
@primitive batchnorm4(x;o...),dy batchnorm4x(x, dy; o...)
@zerograd batchnorm4x(x, dy; o...)
# 4d with affine
@primitive batchnorm4(x, g, b; o...),dy batchnorm4x(x, g,dy; o...) batchnorm4g(x, dy; o...) batchnorm4b(dy; o...)
@zerograd batchnorm4x(x, g, dy; o...)
@zerograd batchnorm4g(x, dy; o...)
@zerograd batchnorm4b(dy; o...)

# 2d w/o affine
@primitive batchnorm2(x;o...),dy batchnorm2x(x, dy; o...)
@zerograd batchnorm2x(x, dy; o...)
# 2d with affine
@primitive batchnorm2(x, g, b; o...),dy batchnorm2x(x, g,dy; o...) batchnorm2g(x, dy; o...) batchnorm2b(dy; o...)
@zerograd batchnorm2x(x, g, dy; o...)
@zerograd batchnorm2g(x, dy; o...)
@zerograd batchnorm2b(dy; o...)
