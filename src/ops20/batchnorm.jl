export batchnorm, bnmoments, bnparams
using AutoGrad: AutoGrad, @primitive1, value, recording
using Statistics: mean, var

# dy20200804: why not have bnparams and bnmoments in a single struct?
# dy20200804: what array types / sizes are they supposed to use? which ones are Param?
# moments and params need to have the same atype as x, params needs to be wrapped in Param.
#TODO: improve the documentation. Explain what batchnorm is.

mutable struct BNMoments
    momentum::AbstractFloat
    mean
    var
    meaninit
    varinit
end


"""
    batchnorm(x[, moments, params]; kwargs...)

perform batch normalization on `x` with optional mean and variance in `moments` and scaling
factor and bias in `params`. See https://arxiv.org/abs/1502.03167 for reference.

2d, 4d and 5d inputs are supported. Mean and variance are computed over dimensions (2,),
(1,2,4) and (1,2,3,5) for 2d, 4d and 5d arrays, respectively.

`moments` stores running mean and variance to be used at inference time.  It is optional in
training mode, but mandatory in test mode.  Training and test modes can be controlled by the
`training` keyword argument which defaults to `Knet.training()`.

`params` stores the optional affine parameters gamma and beta.  `bnparams` function can be
used to initialize `params`.

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

 `training=Knet.training()`: When `training` is true, the mean and variance of `x` are used
 and `moments` argument is modified if it is provided. When `training` is false, mean and
 variance stored in the `moments` argument are used.

"""
function batchnorm(x, moments::Union{BNMoments, Nothing}=nothing, params=nothing;
                   training=AutoGrad.recording(), o...)
    xnd = ndims(x)
    a = (x,)
    if params !== nothing
        g = reshape(_bnscale(params), _wsize(x))
        b = reshape(_bnbias(params), _wsize(x))
        a = (g, b, x)
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
    bnmoments(;momentum=0.1, mean=nothing, var=nothing, meaninit=zeros, varinit=ones)

Return a `BNMoments` object, a data structure used to store running mean and running
variance of batch normalization with the following fields:

* `momentum=0.1`: A real number between 0 and 1 to be used as the scale of last
mean and variance. The existing running mean or variance is multiplied by (1-momentum).

* `mean=nothing`: The running mean.

* `var=nothing`: The running variance.

* `meaninit=zeros`: The function used for initialize the running mean. Should either be
`nothing` or of the form `([eltype], dims...)->data`. `zeros` is a good option.

* `varinit=ones`: The function used for initialize the running variance. Should either be
`nothing` or `([eltype], dims...)->data`. `ones` is a good option.

This constructor can be used directly load moments from data. `meaninit` and `varinit` are
called if `mean` and `var` are nothing. Type and size of the `mean` and `var` are determined
automatically from the inputs in the `batchnorm` calls.

"""
bnmoments(;momentum=0.1, meaninit=zeros, varinit=ones, mean=nothing, var=nothing) =
    BNMoments(momentum, mean, var, meaninit, varinit)


"""
    bnparams(etype, channels::Integer)

Return a single 1d array that contains both scale and bias of batchnorm, where the first
half is scale and the second half is bias.

`bnparams(channels)` calls `bnparams(Float64, channels)`, following Julia convention.

"""
function bnparams(etype, channels::Integer)
    buf = Array{etype}(undef,2channels)
    buf[1:channels] .= 1
    buf[channels+1:end] .= 0
    return buf
end

bnparams(channels::Integer) = bnparams(Float64, channels)

#=
LOW-LEVEL API that won't be exported by default
=#
@inline _bnscale(param) = param[1:div(length(param), 2)]
@inline _bnbias(param) = param[div(length(param), 2)+1:end]

# Dimension helpers
@inline _wsize(y) = ((1 for _=1:ndims(y)-2)..., size(y)[end-1], 1)
@inline _reddims(y) = ((i for i=1:ndims(y)-2)..., ndims(y))

# A black box data type for storing the bn state
mutable struct BNCache
    mean
    ivar
    dx
    dg
    db
end

BNCache() = BNCache(nothing, nothing, nothing, nothing, nothing)

# CPU Implementation
function batchnorm4(g, b, x; o...)
    return _batchnorm4_fused(g, b, x; o...)
end

function batchnorm4(x; o...)
    return _batchnorm4_fused(nothing,nothing,x; o...)
end

function _batchnorm4_fused(g, b, x; eps=1e-5, training=AutoGrad.recording(), cache=nothing, moments=nothing, o...)
    T = eltype(x)
    y = copy(x)
    eps = T(eps)
    dims = _reddims(y)
    if moments!==nothing
        _lazy_init!(moments, x)
    end
    affine = (g !== nothing)
    if training
        mu = mean(y, dims=dims)
        sigma2 = var(y, dims=dims, corrected=false, mean=mu)
        _update_moments!(moments, mu, sigma2)
        sigma2 .= 1 ./ sqrt.(sigma2 .+ eps)
        ivar = sigma2
        if affine
            y .= g .* (y .- mu) .* ivar .+ b
        else
            y .= (y .- mu) .* ivar
        end
        if cache !== nothing
            cache.ivar = ivar
            cache.mean = mu
        end
    else
        @assert (moments!==nothing)  "Moments must be provided in test mode"
        ivar = 1 ./ sqrt.(moments.var .+ eps)
        if affine
            y .= g .* (y .- moments.mean) .* ivar .+ b
        else
            y .= (y .- moments.mean) .* ivar
        end
    end
    return y
end

function _update_moments!(moments, mu, sigma2)
    moments === nothing && return
    moments.mean .= moments.momentum .* mu .+
        (1 .- moments.momentum) .* moments.mean
    moments.var .= moments.momentum .* sigma2 .+
        (1 .- moments.momentum) .* moments.var
end

# TODO: consider automatic type conversion
function _lazy_init!(m::BNMoments, x)
    x = value(x)
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

# Implement batchnorm2 using batchnorm4, with autograd

function batchnorm2(g, b, x; moments=nothing, training=AutoGrad.recording(), o...)
    # TODO: This support should be added when needed
    if training == false && (isa(g, AutoGrad.Value) || isa(x, AutoGrad.Value) || isa(b, AutoGrad.Value))
        error("Test mode backward is not supported with 2d inputs")
    end
    @inline _pad4(x) = reshape(x, (1,1,size(x,1),size(x,2)))
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
                       training=training,
                       o...))
    if moments !== nothing
        moments.mean = mat(moments.mean)
        moments.var = mat(moments.var)
    end
    return y
end

batchnorm2(x;o...) = batchnorm2(nothing, nothing, x; o...)

# CPU backward
function batchnorm4_back(g, x, dy; eps=1e-5, training=AutoGrad.recording(), cache=nothing, moments=nothing,  o...)
    T = eltype(x)
    eps = T(eps)
    dims = _reddims(x)
    if training
        mu, ivar = _get_cache_data(cache, x, eps)
        x_mu = x .- mu
        # equations from the original paper
        dyivar = dy .* ivar
        if g !== nothing
            dg = sum(x_mu .* dyivar, dims=dims)
            db = sum(dy, dims=dims)
            dyivar .*= g
        else
            dg, db = nothing, nothing
        end
        m = prod(d->size(x,d), dims) # size(x, dims...))
        dsigma2 = -sum(dyivar .* x_mu .* ivar.^2, dims=dims) ./ 2
        dmu = -sum(dyivar, dims=dims) .- 2dsigma2 .* sum(x_mu, dims=dims) ./ m
        dx = dyivar .+ dsigma2 .* 2x_mu ./ m .+ dmu ./ m
    else #same reasoning with the gpu version
        ivar = 1 ./ sqrt.(moments.var .+ eps)
        dx = (g !== nothing) ? (dy .* g .* ivar) : (dy .* ivar)
        if g !== nothing
            dg = sum(dy .* (x .- moments.mean) .* ivar, dims=dims)
            db = sum(dy, dims=dims)
        else
            dg, db = nothing, nothing
        end
    end
    return dg, db, dx
end

function _get_cache_data(cache, x, eps)
    if cache !== nothing
        mu = cache.mean
        ivar = cache.ivar
    else
        mu = mean(x, dims=_reddims(x))
        ivar = 1 ./ sqrt.(var(x, dims=_reddims(x); mean=mu, correct=false) .+ eps)
    end
    return mu, ivar
end


function batchnorm4g(g,x,dy; cache=nothing, o...)
    dg, db, dx = batchnorm4_back(g, x, dy; cache=cache, o...)
    if cache !== nothing
        cache.dx, cache.db = dx, db
    else
        @warn("Calling batchnorm with affine without cache is not recommended for performance")
    end
    return dg
end

function batchnorm4b(dy; cache=nothing, o...)
    if cache == nothing || cache.db == nothing
        return sum(dy, dims=_reddims(dy))
    else
        return cache.db
    end
end

function batchnorm4x(g, x, dy; cache=nothing, o...)
    if cache !== nothing && cache.dx !== nothing
        return cache.dx
    else
        return batchnorm4_back(g, x, dy; cache=cache, o...)[3]
    end
end

function batchnorm4x(x, dy; o...)
    return batchnorm4_back(nothing, x, dy; o...)[3]
end


@primitive1 batchnorm4(x;o...),dy batchnorm4x(x, dy; o...)

# 4d with affine
@primitive1 batchnorm4(g, b, x; o...),dy batchnorm4g(g, x, dy; o...) batchnorm4b(dy; o...) batchnorm4x(g, x, dy; o...)
