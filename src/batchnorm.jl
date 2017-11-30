"""
`bnmoments(;momentum=0.1, mean=nothing, var=nothing, meaninit=zeros, varinit=ones)` can be used
 directly load moments from data. `meaninit` and `varinit` are called if `mean` and `var` 
are nothing. Type and size of the `mean` and `var` are determined automatically from the inputs
in the `batchnorm` calls. A `BNMoments` object is returned.


# BNMoments
A high-level data structure used to store running mean and running variance
of batch normalization with the following fields:

 `momentum::AbstractFloat`: A real number between 0 and 1 to be used as the scale of
  last mean and variance. The existing running mean or variance is multiplied by 
  (1-momentum).
 
 `mean`: The running mean.

 `var`: The running variance.
 
 `meaninit`: The function used for initialize the running mean. Should either be `nothing` or
of the form `(eltype, dims...)->data`. `zeros` is a good option.

 `varinit`: The function used for initialize the running variance. Should either be `nothing` or
`(eltype, dims...)->data`. `ones` is a good option.
"""
bnmoments(;momentum=0.1, meaninit=zeros, varinit=ones, mean=nothing, var=nothing) =
    BNMoments(momentum, mean, var, meaninit, varinit)

type BNMoments
    momentum::AbstractFloat
    mean
    var
    meaninit
    varinit
end

#TODO: improve the documentation. Explain what batchnorm is.
"""

`batchnorm(x[, moments, params]; kwargs...)` performs batch normalization to `x`
with optional scaling factor and bias stored in `params`.

2d, 4d and 5d inputs are supported. Mean and variance are computed over 
dimensions (2,), (1,2,4) and (1,2,3,5) for 2d, 4d and 5d arrays, respectively.


`moments` stores running mean and variance to be used in testing. 
It is optional in the training mode, but mendatory in the test mode.
Training and test modes are controlled by the `training` keyword argument.

`params` stores the optional affine parameters gamma and beta.
`bnparams` function can be used to initialize `params`.

# Example
    
    # Inilization, C is an integer
    moments = bnmoments()
    params = bnparams(C)
    ...
    # size(x) -> (H, W, C, N)
    y = batchnorm(x, moments, params)
    # size(y) -> (H, W, C, N)
    

# Keywords

 `eps=1e-5`: The epsilon parameter added to the variance to avoid division by 0.
 
 `training`: When `training` is true, the mean and variance of `x` are used and `moments`
 argument is modified if it is provided. When `training` is false, mean and variance stored in 
 the `moments` argument are used. Default value is `true` when at least one of `x` and `params`
 is `AutoGrad.Rec`, `false` otherwise.
"""

function batchnorm(x, moments::Union{BNMoments, Void}=nothing, params=nothing;
                   training=nothing, o...)
    xnd = ndims(x)
    a = (x,)
    if params !== nothing
        g = reshape(_bnscale(params), _wsize(x))
        b = reshape(_bnbias(params), _wsize(x))
        a = (g, b, x)
    end
    if ~isa(training, Bool)
        training = isa(x, Rec) || isa(params, Rec)
    end
    if xnd == 2
        return batchnorm2(a...; o...,
                          moments=moments,
                          training=training,
                          cache=BNCache())
    elseif xnd in [4, 5]
        return batchnorm4(a...; o...,
                          moments=moments,
                          training=training,
                          cache=BNCache())
    else
        error("Unsupported input dimension ", xnd)
    end
end


"""
`bnparams(etype, channels)` creates a single 1d array that contains both 
scale and bias of batchnorm, where the first half is scale and the
second half is bias.

`bnparams(channels)` calls `bnparams` with `etype=Float64`, following Julia convention

"""
function bnparams(etype, channels::Integer)
    buf = Array{etype}(2channels)
    buf[1:channels] = 1
    buf[channels+1:end] = 0
    return buf
end

bnparams(channels::Integer) = bnparams(Float64, channels)


#= 
LOW-LEVEL API that won't be exported by default
=#
@inline _bnscale(param) = param[1:div(length(param), 2)]
@inline _bnbias(param) = param[div(length(param), 2)+1:end]

const BN_MODE_SPATIAL = 1
const BN_MODE_ACTIVATION = 0
const CUDNN_BN_MIN_EPS = 1e-5

# A black box data type for storing the bn state
type BNCache
    mean
    ivar
    dx
    dg
    db
end

BNCache() = BNCache(nothing, nothing, nothing, nothing, nothing)


# Dimension helpers
@inline _wsize(y) = ((1 for _=1:ndims(y)-2)..., size(y)[end-1], 1)
@inline _reddims(y) = ((i for i=1:ndims(y)-2)..., ndims(y))

# TODO: consider automatic type conversion
function _lazy_init!(m::BNMoments, x)
    x = getval(x)
    buf_size = (ndims(x) > 2) ? _wsize(x) : (size(x,1), 1)
    tx = typeof(x)
    ex = eltype(x)
    m.momentum = ex(m.momentum)
    if m.mean == nothing
        m.mean = tx(m.meaninit(ex, buf_size...))
    end
    if m.var == nothing
        m.var = tx(m.varinit(ex, buf_size...))
    end
end


# Only spatial mode is supported
# TODO: support per-activation mode
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
    g = KnetArray{T}(ones(_wsize(x)...))
    b = KnetArray{T}(zeros(_wsize(x)...))
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
        weight_size = _wsize(dy)
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
        # so the derivative dx = dy .* g. / sqrt(var + eps) since mean and var
        # are constants
        # TODO: Test this operation
        # Note: moments must exist since otherwise forward pass fails
        ivar = 1 ./ sqrt.(moments.var .+ eps)
        dx = (g !== nothing) ? (dy .* g .* ivar) : (dy .* ivar)
        if g !== nothing
            dg = sum(dy .* (x .- moments.mean) .* ivar, _reddims(dy))
            db = sum(dy, _reddims(dy))
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
    (cache == nothing || cache.db == nothing) && return sum(dy, _reddims(dy))
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
    dims = _reddims(x)
    if training
        mu = mean(x, dims)
        x_mu = x .- mean(x, dims)
        sigma2 = mean(x_mu .* x_mu, dims)
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
        mu = mean(x, _reddims(x))
        x_mu = x .- mu
        ivar = 1 ./ sqrt.(mean(x_mu.*x_mu, _reddims(x)) .+ eps)
    end
    return x_mu, ivar
end

function batchnorm4_back{T}(g::Union{Array{T}, Void}, x::Array{T}, dy::Array{T};
                            eps=1e-5, training=true,
                            cache=nothing, moments=nothing,  o...)
    eps = T(eps)
    dims = _reddims(x)
    if training
        x_mu, ivar = _get_cache_data(cache, x, eps)
        # equations from the original paper
        dyivar = dy .* ivar
        if g !== nothing
            dg = sum(x_mu .* dyivar, dims)
            db = sum(dy, dims)
            dy = g .* dy
        else
            dg, db = nothing, nothing
        end
        m = prod(size(x, dims...))
        dsigma2 = -T(0.5) .* sum(dyivar .* x_mu .* ivar.^2, dims)
        dmu = -sum(dyivar, dims) .- 2dsigma2 .* sum(x_mu, dims) ./ m
        dx = dyivar .+ (dsigma2 .* 2x_mu .+ dmu) ./ m
    else #same reasoning with the gpu version
        ivar = 1 ./ sqrt.(moments.var .+ eps)
        dx = (g !== nothing) ? (dy .* g .* ivar) : (dy .* ivar)
        if g !== nothing
            dg = sum(dy .* (x .- moments.mean) .* ivar, dims)
            db = sum(dy, dims)
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

function batchnorm2(g, b, x; moments=nothing, training=false, o...)
    # TODO: This support should be added when needed
    if training == false && (isa(g, Rec) || isa(x, Rec) || isa(b, Rec))
        error("Test mode backward is not supported with 2d")
    end
    @inline _pad4(x) = reshape(x, (1,1,size(x,1,2)...))
     # process moments
    if moments !== nothing
        _lazy_init!(moments, x)
        moments.mean = _pad4(moments.mean)
        moments.var = _pad4(moments.var)
    end
    x = _pad4(x)
    args = (x,)
    if g !== nothing
        g = _pad4(g)
        b = _pad4(b)
        args = (g, b, x)
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
