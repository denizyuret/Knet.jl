using LinearAlgebra

"""
    SGD(;lr=0.001,gclip=0)
    update!(w,g,p::SGD)
    update!(w,g;lr=0.001)

Container for parameters of the Stochastic gradient descent (SGD)
optimization algorithm used by [`update!`](@ref).

SGD is an optimization technique to minimize an objective function by
updating its weights in the opposite direction of their gradient. The
learning rate (lr) determines the size of the step.  SGD updates the
weights with the following formula:

    w = w - lr * g

where `w` is a weight array, `g` is the gradient of the loss function
w.r.t `w` and `lr` is the learning rate.

If `norm(g) > gclip > 0`, `g` is scaled so that its norm is equal
to `gclip`.  If `gclip==0` no scaling takes place.

SGD is used by default if no algorithm is specified in the two
argument version of `update!`[@ref].
"""
mutable struct SGD
    lr::AbstractFloat
    gclip::AbstractFloat
end

const SGDLR = 0.001

SGD(; lr=SGDLR, gclip=0) = SGD(lr,gclip)

@deprecate Sgd SGD

"""
    Momentum(;lr=0.001, gclip=0, gamma=0.9)
    update!(w,g,p::Momentum)

Container for parameters of the Momentum optimization algorithm used
by [`update!`](@ref).

The Momentum method tries to accelerate SGD by adding a velocity term
to the update.  This also decreases the oscillation between successive
steps. It updates the weights with the following formulas:

    velocity = gamma * velocity + lr * g
    w = w - velocity

where `w` is a weight array, `g` is the gradient of the objective
function w.r.t `w`, `lr` is the learning rate, `gamma` is the momentum
parameter, `velocity` is an array with the same size and type of `w`
and holds the accelerated gradients.

If `norm(g) > gclip > 0`, `g` is scaled so that its norm is equal
to `gclip`.  If `gclip==0` no scaling takes place.

Reference: [Qian,
N. (1999)](http://doi.org/10.1016/S0893-6080(98)00116-6). On the
momentum term in gradient descent learning algorithms.  Neural
Networks : The Official Journal of the International Neural Network
Society, 12(1), 145–151.

"""
mutable struct Momentum
    lr::AbstractFloat
    gclip::AbstractFloat
    gamma::AbstractFloat
    velocity
end

Momentum(; lr=0.001, gclip=0, gamma=0.9)=Momentum(lr, gclip, gamma, nothing)


"""
    Nesterov(; lr=0.001, gclip=0, gamma=0.9)
    update!(w,g,p::Momentum)

Container for parameters of Nesterov's momentum optimization algorithm used
by [`update!`](@ref).

It is similar to standard [`Momentum`](@ref) but with a slightly different update
rule:

    velocity = gamma * velocity_old - lr * g
    w = w_old - velocity_old + (1+gamma) * velocity

where `w` is a weight array, `g` is the gradient of the objective
function w.r.t `w`, `lr` is the learning rate, `gamma` is the momentum
parameter, `velocity` is an array with the same size and type of `w`
and holds the accelerated gradients.

If `norm(g) > gclip > 0`, `g` is scaled so that its norm is equal
to `gclip`.  If `gclip == 0` no scaling takes place.

Reference Implementation : [Yoshua Bengio, Nicolas Boulanger-Lewandowski and Razvan P
ascanu](https://arxiv.org/pdf/1212.0901.pdf)
"""
mutable struct Nesterov
    lr::AbstractFloat
    gclip::AbstractFloat
    gamma::AbstractFloat
    velocity
end

Nesterov(; lr=0.001, gclip=0, gamma=0.9) = Nesterov(lr, gclip, gamma, nothing)


"""
    Adagrad(;lr=0.1, gclip=0, eps=1e-6)
    update!(w,g,p::Adagrad)

Container for parameters of the Adagrad optimization algorithm used by
[`update!`](@ref).

Adagrad is one of the methods that adapts the learning rate to each of
the weights.  It stores the sum of the squares of the gradients to
scale the learning rate.  The learning rate is adapted for each weight
by the value of current gradient divided by the accumulated
gradients. Hence, the learning rate is greater for the parameters
where the accumulated gradients are small and the learning rate is
small if the accumulated gradients are large. It updates the weights
with the following formulas:

    G = G + g .^ 2
    w = w - g .* lr ./ sqrt(G + eps)

where `w` is the weight, `g` is the gradient of the objective function
w.r.t `w`, `lr` is the learning rate, `G` is an array with the same
size and type of `w` and holds the sum of the squares of the
gradients. `eps` is a small constant to prevent a zero value in the
denominator.

If `norm(g) > gclip > 0`, `g` is scaled so that its norm is equal
to `gclip`.  If `gclip==0` no scaling takes place.

Reference: [Duchi, J., Hazan, E., & Singer,
Y. (2011)](http://jmlr.org/papers/v12/duchi11a.html). Adaptive
Subgradient Methods for Online Learning and Stochastic Optimization.
Journal of Machine Learning Research, 12, 2121–2159.

"""
mutable struct Adagrad
    lr::AbstractFloat
    gclip::AbstractFloat
    eps::AbstractFloat
    G
end

Adagrad(; lr=0.1, gclip=0, eps=1e-6)=Adagrad(lr, gclip, eps, nothing)


"""
    Adadelta(;lr=0.01, gclip=0, rho=0.9, eps=1e-6)
    update!(w,g,p::Adadelta)

Container for parameters of the Adadelta optimization algorithm used by
[`update!`](@ref).

Adadelta is an extension of Adagrad that tries to prevent the decrease
of the learning rates to zero as training progresses. It scales the
learning rate based on the accumulated gradients like Adagrad and
holds the acceleration term like Momentum. It updates the weights with
the following formulas:

    G = (1-rho) * g .^ 2 + rho * G
    update = g .* sqrt(delta + eps) ./ sqrt(G + eps)
    w = w - lr * update
    delta = rho * delta + (1-rho) * update .^ 2

where `w` is the weight, `g` is the gradient of the objective function
w.r.t `w`, `lr` is the learning rate, `G` is an array with the same
size and type of `w` and holds the sum of the squares of the
gradients. `eps` is a small constant to prevent a zero value in the
denominator.  `rho` is the momentum parameter and `delta` is an array
with the same size and type of `w` and holds the sum of the squared
updates.

If `norm(g) > gclip > 0`, `g` is scaled so that its norm is equal
to `gclip`.  If `gclip==0` no scaling takes place.

Reference: [Zeiler,
M. D. (2012)](http://arxiv.org/abs/1212.5701). ADADELTA: An Adaptive
Learning Rate Method.

"""
mutable struct Adadelta
    lr::AbstractFloat
    gclip::AbstractFloat
    rho::AbstractFloat
    eps::AbstractFloat
    G
    delta
end

Adadelta(; lr=0.01, gclip=0, rho=0.9, eps=1e-6)=Adadelta(lr, gclip, rho, eps, nothing, nothing)


"""
    Rmsprop(;lr=0.001, gclip=0, rho=0.9, eps=1e-6)
    update!(w,g,p::Rmsprop)

Container for parameters of the Rmsprop optimization algorithm used by
[`update!`](@ref).

Rmsprop scales the learning rates by dividing the root mean squared of
the gradients. It updates the weights with the following formula:

    G = (1-rho) * g .^ 2 + rho * G
    w = w - lr * g ./ sqrt(G + eps)

where `w` is the weight, `g` is the gradient of the objective function
w.r.t `w`, `lr` is the learning rate, `G` is an array with the same
size and type of `w` and holds the sum of the squares of the
gradients. `eps` is a small constant to prevent a zero value in the
denominator.  `rho` is the momentum parameter and `delta` is an array
with the same size and type of `w` and holds the sum of the squared
updates.

If `norm(g) > gclip > 0`, `g` is scaled so that its norm is equal
to `gclip`.  If `gclip==0` no scaling takes place.

Reference: [Tijmen Tieleman and Geoffrey Hinton
(2012)](https://dirtysalt.github.io/images/nn-class-lec6.pdf). "Lecture
6.5-rmsprop: Divide the gradient by a running average of its recent
magnitude."  COURSERA: Neural Networks for Machine Learning 4.2.

"""
mutable struct Rmsprop
    lr::AbstractFloat
    gclip::AbstractFloat
    rho::AbstractFloat
    eps::AbstractFloat
    G
end

Rmsprop(; lr=0.001, gclip=0, rho=0.9, eps=1e-6)=Rmsprop(lr, gclip, rho, eps, nothing)


"""
    Adam(;lr=0.001, gclip=0, beta1=0.9, beta2=0.999, eps=1e-8, l2decay=0)
    update!(w,g,p::Adam)

Container for parameters of the Adam optimization algorithm used by
[`update!`](@ref).

Adam is one of the methods that compute the adaptive learning rate. It
stores accumulated gradients (first moment) and the sum of the squared
of gradients (second).  It scales the first and second moment as a
function of time. Here is the update formulas:

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g .* g
    mhat = m ./ (1 - beta1 ^ t)
    vhat = v ./ (1 - beta2 ^ t)
    w = w - (lr / (sqrt(vhat) + eps)) * mhat

where `w` is the weight, `g` is the gradient of the objective function
w.r.t `w`, `lr` is the learning rate, `m` is an array with the same
size and type of `w` and holds the accumulated gradients. `v` is an
array with the same size and type of `w` and holds the sum of the
squares of the gradients. `eps` is a small constant to prevent a zero
denominator. `beta1` and `beta2` are the parameters to calculate bias
corrected first and second moments. `t` is the update count.

If `norm(g) > gclip > 0`, `g` is scaled so that its norm is equal
to `gclip`.  If `gclip==0` no scaling takes place.

Reference: [Kingma, D. P., & Ba,
J. L. (2015)](https://arxiv.org/abs/1412.6980). Adam: a Method for
Stochastic Optimization. International Conference on Learning
Representations, 1–13.

"""
mutable struct Adam
    lr::AbstractFloat
    gclip::AbstractFloat
    beta1::AbstractFloat
    beta2::AbstractFloat
    eps::AbstractFloat
    t::Int
    fstm
    scndm
    l2decay::AbstractFloat
end

Adam(; lr=0.001, gclip=0, beta1=0.9, beta2=0.999, eps=1e-8, l2decay=0)=Adam(lr, gclip, beta1, beta2, eps, 0, nothing, nothing, l2decay)


"""
    update!(weights, gradients, params)
    update!(weights, gradients; lr=0.001, gclip=0)

Update the `weights` using their `gradients` and the optimization
algorithm parameters specified by `params`.  The 2-arg version
defaults to the [`SGD`](@ref) algorithm with learning rate `lr` and
gradient clip `gclip`.  `gclip==0` indicates no clipping. The
`weights` and possibly `gradients` and `params` are modified in-place.

`weights` can be an individual numeric array or a collection of arrays
represented by an iterator or dictionary.  In the individual case,
`gradients` should be a similar numeric array of `size(weights)` and
`params` should be a single object.  In the collection case, each
individual weight array should have a corresponding params object.
This way different weight arrays can have their own optimization
state, different learning rates, or even different optimization
algorithms running in parallel.  In the iterator case, `gradients` and
`params` should be iterators of the same length as `weights` with
corresponding elements.  In the dictionary case, `gradients` and
`params` should be dictionaries with the same keys as `weights`.

Individual optimization parameters can be one of the following
types. The keyword arguments for each type's constructor and their
default values are listed as well.

* [`SGD`](@ref)`(;lr=0.001, gclip=0)`
* [`Momentum`](@ref)`(;lr=0.001, gclip=0, gamma=0.9)`
* [`Nesterov`](@ref)`(;lr=0.001, gclip=0, gamma=0.9)`
* [`Rmsprop`](@ref)`(;lr=0.001, gclip=0, rho=0.9, eps=1e-6)`
* [`Adagrad`](@ref)`(;lr=0.1, gclip=0, eps=1e-6)`
* [`Adadelta`](@ref)`(;lr=0.01, gclip=0, rho=0.9, eps=1e-6)`
* [`Adam`](@ref)`(;lr=0.001, gclip=0, beta1=0.9, beta2=0.999, eps=1e-8)`

# Example:

    w = rand(d)                 # an individual weight array
    g = lossgradient(w)         # gradient g has the same shape as w
    update!(w, g)               # update w in-place with SGD()
    update!(w, g; lr=0.1)       # update w in-place with SGD(lr=0.1)
    update!(w, g, SGD(lr=0.1))  # update w in-place with SGD(lr=0.1)

    w = (rand(d1), rand(d2))    # a tuple of weight arrays
    g = lossgradient2(w)        # g will also be a tuple
    p = (Adam(), SGD())         # p has params for each w[i]
    update!(w, g, p)            # update each w[i] in-place with g[i],p[i]

    w = Any[rand(d1), rand(d2)] # any iterator can be used
    g = lossgradient3(w)        # g will be similar to w
    p = Any[Adam(), SGD()]      # p should be an iterator of same length
    update!(w, g, p)            # update each w[i] in-place with g[i],p[i]

    w = Dict(:a => rand(d1), :b => rand(d2)) # dictionaries can be used
    g = lossgradient4(w)
    p = Dict(:a => Adam(), :b => SGD())
    update!(w, g, p)

"""
function update! end

for T in (Array{Float32},Array{Float64},KnetArray{Float32},KnetArray{Float64}); @eval begin

    function update!(w::$T, g::$T, p::SGD)
        gclip!(g, p.gclip)
        axpy!(-p.lr, g, w)
    end

    # Two arg defaults to SGD
    function update!(w::$T, g::$T; lr=SGDLR, gclip=0)
        gclip!(g, gclip)
        axpy!(-lr, g, w)
    end

    function update!(w::$T, g::$T, p::Momentum)
        gclip!(g, p.gclip)
        if p.velocity===nothing; p.velocity=zero(w); end
        lmul!(p.gamma, p.velocity)
        axpy!(p.lr, g, p.velocity)
        axpy!(-1, p.velocity, w)
    end

    # https://arxiv.org/pdf/1212.0901.pdf Eq. (7)
    function update!(w::$T, g::$T, p::Nesterov)
        gclip!(g, p.gclip)
        p.velocity ===nothing && (p.velocity = zero(w))
        lmul!(p.gamma, p.velocity)
        axpy!(-1, p.velocity, w)
        axpy!(-p.lr, g, p.velocity)
        axpy!(1+p.gamma, p.velocity, w)
    end

    function update!(w::$T, g::$T, p::Adam)
        l2decay!(w, g, p)
        gclip!(g, p.gclip)
        if p.fstm===nothing; p.fstm=zero(w); p.scndm=zero(w); end
        p.t += 1
        lmul!(p.beta1, p.fstm)
        axpy!(1-p.beta1, g, p.fstm)
        lmul!(p.beta2, p.scndm)
        axpy!(1-p.beta2, g .* g, p.scndm)
        fstm_corrected = p.fstm / (1 - p.beta1 ^ p.t)
        scndm_corrected = p.scndm / (1 - p.beta2 ^ p.t)
        axpy!(-p.lr, (fstm_corrected ./ (sqrt.(scndm_corrected) .+ p.eps)), w)
    end

    function update!(w::$T, g::$T, p::Adagrad)
        gclip!(g, p.gclip)
        if p.G===nothing; p.G=zero(w); end
        axpy!(1, g .* g, p.G)
        axpy!(-p.lr, g ./ sqrt.(p.G .+ p.eps), w)
    end

    function update!(w::$T, g::$T, p::Adadelta)
        gclip!(g, p.gclip)
        if p.G===nothing; p.G=zero(w); p.delta=zero(w); end
        lmul!(p.rho, p.G)
        axpy!(1-p.rho, g .* g, p.G)
        dw = g .* sqrt.(p.delta .+ p.eps) ./ sqrt.(p.G .+ p.eps)
        lmul!(p.rho, p.delta)
        axpy!(1-p.rho, dw .* dw , p.delta)
        axpy!(-p.lr, dw, w)
    end

    function update!(w::$T, g::$T, p::Rmsprop)
        gclip!(g, p.gclip)
        if p.G===nothing; p.G=zero(w); end
        lmul!(p.rho, p.G)
        axpy!(1-p.rho, g .* g, p.G)
        axpy!(-p.lr, g ./ sqrt.(p.G .+ p.eps), w)
    end

    # If type of g does not match, something may be wrong
    update!(w::$T, g, p)=error("Gradient type mismatch: w::$(typeof(w)) g::$(typeof(g))")
    update!(w::$T, g; o...)=error("Gradient type mismatch: w::$(typeof(w)) g::$(typeof(g))")

    # AutoGrad may return Nothing for a zero gradient
    update!(w::$T, g::Nothing, p)=w
    update!(w::$T, g::Nothing; o...)=w

end; end

# AutoGrad may return Nothing for a zero gradient
update!(w, g::Nothing, p)=w
update!(w, g::Nothing; o...)=w

# This takes care of arrays, tuples, iterators in general.
function update!(w,g,p)
    if !(length(w)==length(g)==length(p))
        error("weight, gradient, and optimization parameters not the same length.")
    end
    if isbitstype(eltype(w))
        error("Bad args: $((typeof(w),typeof(g),typeof(p)))")
    end
    for (wi,gi,pi) in zip(w,g,p)
        update!(wi,gi,pi)
    end
end

# We still need an extra method for Dict.
function update!(w::AbstractDict,g::AbstractDict,p::AbstractDict)
    # g may have some keys missing!
    # if !(length(w)==length(g)==length(p))
    #     error("weight, gradient, and optimization parameters not the same length.")
    # end
    for k in keys(g)
        update!(w[k],g[k],p[k])
    end
end

# Two arg version defaults to SGD.
function update!(w,g;lr=SGDLR,gclip=0)
    if !(length(w)==length(g))
        error("weight, gradient not the same length.")
    end
    for (wi,gi) in zip(w,g)
        update!(wi,gi;lr=lr,gclip=gclip)
    end
end

# Two arg version defaults to SGD.
function update!(w::AbstractDict,g::AbstractDict;lr=SGDLR,gclip=0)
    # g may have some keys missing!
    # if !(length(w)==length(g))
    #     error("weight, gradient not the same length.")
    # end
    for k in keys(g)
        update!(w[k],g[k];lr=lr,gclip=gclip)
    end
end

function gclip!(g, gclip)
    if gclip == 0
        g
    else
        gnorm = norm(g)
        if gnorm <= gclip
            g
        else
            lmul!(gclip/gnorm, g)
        end
    end
end

function l2decay!(w, g, o)
    o.l2decay == 0 && return g
    axpy!(o.l2decay, w, g)
end


"""
    optimizers(model, otype; options...)

Given parameters of a `model`, initialize and return corresponding
optimization parameters for a given optimization type `otype` and
optimization options `options`. This is useful because each numeric
array in model needs its own distinct optimization
parameter. `optimizers` makes the creation of optimization parameters
that parallel model parameters easy when all of them use the same type
and options.

"""
optimizers(::KnetArray{<:Number},otype; o...) = otype(;o...)
optimizers(::AbstractArray{<:Number},otype; o...) = otype(;o...)
optimizers(a::AbstractDict,otype; o...)=Dict([ k=>optimizers(v,otype;o...) for (k,v) in a ])
optimizers(a::Tuple,otype; o...)=map(x->optimizers(x,otype;o...), a)
optimizers(a::AbstractArray,otype; o...)=map(x->optimizers(x,otype;o...), a)
optimizers(a,otype;o...)=nothing
