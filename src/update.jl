# These types define per-parameter config and state for various optimization algorithms. The
# lowercase functions create an optimization iterator and the bang versions run the iterator
# both calling minimize. Minimize takes the given type as a global default and attaches a copy
# of it to any parameter's Param.opt if not already set. One can override this default by
# pre-setting Param.opt of a parameter, in which case it will not be overriden by minimize.

# TODO: handle common tasks like gclip and decay in minimize?
# TODO: use common/consistent keyword arg names.
# https://jlmelville.github.io/mize/nesterov.html

"""
    minimize(func, data, optimizer=Adam(); params)
    sgd     (func, data; lr=0.1,  gclip, params)
    momentum(func, data; lr=0.05, gamma=0.95, gclip, params)
    nesterov(func, data; lr=0.05, gamma=0.95, gclip, params)
    adagrad (func, data; lr=0.05, eps=1e-6, gclip, params)
    rmsprop (func, data; lr=0.01, rho=0.9, eps=1e-6, gclip, params)
    adadelta(func, data; lr=1.0,  rho=0.9, eps=1e-6, gclip, params)
    adam    (func, data; lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, gclip, params)

Return an iterator which applies `func` to arguments in `data`, i.e.  `(func(args...) for
args in data)`, and updates the parameters every iteration to minimize `func`.  `func`
should return a scalar value.

The common keyword argument `params` can be used to list the `Param`s to be optimized.  If
not specified, any `Param` that takes part in the computation of `func(args...)` will be
updated.

The common keyword argument `gclip` can be used to implement per-parameter gradient
clipping. For a parameter gradient `g`, if `norm(g) > gclip > 0`, `g` is scaled so that its
norm is equal to `gclip`. If not specified no gradient clipping is performed.

These functions do not perform optimization, but return an iterator that can. Any function
that produces values from an iterator can be used with such an object, e.g.
`progress!(sgd(f,d))` iterates the sgd optimizer and displays a progress bar. For convenience,
appending `!` to the name of the function iterates and returns `nothing`, i.e. `sgd!(...)` is
equivalent to `(for x in sgd(...) end)`.

We define optimizers as lazy iterators to have explicit control over them:
* To report progress use `progress(sgd(f,d))`.
* To run until convergence use `converge(sgd(f,cycle(d)))`.
* To run multiple epochs use `sgd(f,repeat(d,n))`.
* To run a given number of iterations use `sgd(f,take(cycle(d),n))`.
* To do a task every n iterations use `(task() for (i,j) in enumerate(sgd(f,d)) if i%n == 1)`.

These functions apply the same algorithm with the same configuration to every parameter by
default. `minimize` takes an explicit optimizer argument, all others call `minimize` with an
appropriate optimizer argument (see `@doc update!` for a list of possible optimizers). Before
calling [`update!`](@ref) on a `Param`, `minimize` sets its `opt` field to a copy of this
default optimizer if it is not already set. The `opt` field is used by the `update!` function
to determine the type of update performed on that parameter.  If you need finer grained
control, you can set the optimizer of an individual `Param` by setting its `opt` field before
calling one of these functions. They will not override the `opt` field if it is already set,
e.g. `sgd(model,data)` will perform an `Adam` update for a parameter whose `opt` field is an
`Adam` object. This also means you can stop and start the training without losing optimization
state, the first call will set the `opt` fields and the subsequent calls will not override
them.

Given a parameter `w` and its gradient `g` here are the updates applied by each optimizer:

    # sgd (http://en.wikipedia.org/wiki/Stochastic_gradient_descent)
    w .= w - lr * g

    # momentum (http://jlmelville.github.io/mize/nesterov.html)
    v .= gamma * v - lr * g
    w .= w + v
   
    # nesterov (http://jlmelville.github.io/mize/nesterov.html)
    w .= w - gamma * v
    v .= gamma * v - lr * g
    w .= w + (1 + gamma) * v
    
    # adagrad (http://www.jmlr.org/papers/v12/duchi11a.html)
    G .= G + g .^ 2
    w .= w - lr * g ./ sqrt(G + eps)

    # rmsprop (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    G .= rho * G + (1-rho) * g .^ 2 
    w .= w - lr * g ./ sqrt(G + eps)

    # adadelta (http://arxiv.org/abs/1212.5701)
    G .= rho * G + (1-rho) * g .^ 2
    update = sqrt(delta + eps) .* g ./ sqrt(G + eps)
    w = w - lr * update
    delta = rho * delta + (1-rho) * update .^ 2
    
    # adam (http://arxiv.org/abs/1412.6980)
    v = beta1 * v + (1 - beta1) * g
    G = beta2 * G + (1 - beta2) * g .^ 2
    vhat = v ./ (1 - beta1 ^ t)
    Ghat = G ./ (1 - beta2 ^ t)
    w = w - (lr / (sqrt(Ghat) + eps)) * vhat

"""
minimize, minimize!, sgd, sgd!, momentum, momentum!, nesterov, nesterov!, adagrad, adagrad!, rmsprop, rmsprop!, adadelta, adadelta!, adam, adam!

using LinearAlgebra

"""
    SGD(;lr=0.1,gclip=0)
    update!(w,g,p::SGD)
    update!(w,g;lr=0.1)

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
    gclip::AbstractFloat        # TODO: should gclip, decay etc be global?
end

const SGDLR = 0.1

SGD(; lr=SGDLR, gclip=0) = SGD(lr,gclip)
sgd(f,d;lr=SGDLR, gclip=0, o...)=minimize(f,d,SGD(lr,gclip);o...)
sgd!(x...;o...)=for y in sgd(x...;o...); end

clone(s::SGD)=SGD(s.lr,s.gclip)

function Sgd(x...;o...)
    @warn "Sgd is deprecated, use SGD instead." maxlog=1
    SGD(x...; o...)
end

"""
    Momentum(;lr=0.05, gclip=0, gamma=0.95)
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
    gamma::AbstractFloat
    gclip::AbstractFloat
    velocity
end

Momentum(; lr=0.05, gclip=0, gamma=0.95)=Momentum(lr, gamma, gclip, nothing)
momentum(f,d;lr=0.05,gclip=0,gamma=0.95,o...)=minimize(f,d,Momentum(lr,gamma,gclip,nothing);o...)
momentum!(x...;o...)=for y in momentum(x...;o...); end

clone(m::Momentum)=Momentum(m.lr,m.gamma,m.gclip,nothing)


"""
    Nesterov(; lr=0.05, gclip=0, gamma=0.95)
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
    gamma::AbstractFloat
    gclip::AbstractFloat
    velocity
end

Nesterov(; lr=0.05, gclip=0, gamma=0.95) = Nesterov(lr, gamma, gclip, nothing)
nesterov(f,d;lr=0.05,gclip=0,gamma=0.95,o...)=minimize(f,d,Nesterov(lr,gamma,gclip,nothing);o...)
nesterov!(x...;o...)=for y in nesterov(x...;o...); end

clone(m::Nesterov)=Nesterov(m.lr,m.gamma,m.gclip,nothing)

"""
    Adagrad(;lr=0.05, gclip=0, eps=1e-6)
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
    eps::AbstractFloat
    gclip::AbstractFloat
    G
end

Adagrad(; lr=0.05, gclip=0, eps=1e-6)=Adagrad(lr, eps, gclip, nothing)
adagrad(f,d;lr=0.05,gclip=0,eps=1e-6,o...)=minimize(f,d,Adagrad(lr,eps,gclip,nothing);o...)
adagrad!(x...;o...)=for y in adagrad(x...;o...); end

clone(a::Adagrad)=Adagrad(a.lr,a.eps,a.gclip,nothing)

"""
    Rmsprop(;lr=0.01, gclip=0, rho=0.9, eps=1e-6)
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
    rho::AbstractFloat
    eps::AbstractFloat
    gclip::AbstractFloat
    G
end

Rmsprop(; lr=0.01, gclip=0, rho=0.9, eps=1e-6)=Rmsprop(lr, rho, eps, gclip, nothing)
rmsprop(f,d;lr=0.01,gclip=0,rho=0.9,eps=1e-6,o...)=minimize(f,d,Rmsprop(lr,rho,eps,gclip,nothing);o...)
rmsprop!(x...;o...)=for y in rmsprop(x...;o...); end

clone(r::Rmsprop)=Rmsprop(r.lr,r.rho,r.eps,r.gclip,nothing)

"""
    Adadelta(;lr=1.0, gclip=0, rho=0.9, eps=1e-6)
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
    rho::AbstractFloat
    eps::AbstractFloat
    gclip::AbstractFloat
    G
    delta
end

Adadelta(; lr=1.0, gclip=0, rho=0.9, eps=1e-6)=Adadelta(lr, rho, eps, gclip, nothing, nothing)
adadelta(f,d;lr=1.0,gclip=0,rho=0.9,eps=1e-6,o...)=minimize(f,d,Adadelta(lr,rho,eps,gclip,nothing,nothing);o...)
adadelta!(x...;o...)=for y in adadelta(x...;o...); end

clone(a::Adadelta)=Adadelta(a.lr,a.rho,a.eps,a.gclip,nothing,nothing)

"""
    Adam(;lr=0.001, gclip=0, beta1=0.9, beta2=0.999, eps=1e-8)
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
    beta1::AbstractFloat
    beta2::AbstractFloat
    eps::AbstractFloat
    t::Int
    gclip::AbstractFloat
    fstm
    scndm
end

Adam(; lr=0.001, gclip=0, beta1=0.9, beta2=0.999, eps=1e-8)=Adam(lr, beta1, beta2, eps, 0, gclip, nothing, nothing)
adam(f,d;lr=0.001,gclip=0,beta1=0.9,beta2=0.999,eps=1e-8,o...)=minimize(f,d,Adam(lr,beta1,beta2,eps,0,gclip,nothing,nothing);o...)
adam!(x...;o...)=for y in adam(x...;o...); end

clone(a::Adam)=Adam(a.lr,a.beta1,a.beta2,a.eps,0,a.gclip,nothing,nothing)

"""
    update!(weights::Param, gradients)
    update!(weights, gradients; lr=0.1, gclip=0)
    update!(weights, gradients, optimizers)

Update the `weights` using their `gradients` and the optimization algorithms specified using
(1) the `opt` field of a `Param`, (2) keyword arguments, (3) the third argument.

`weights` can be an individual `Param`, numeric array, or a collection of arrays/Params
represented by an iterator or dictionary. `gradients` should be a matching individual array or
collection. In the first form, the optimizer should be specified in `weights.opt`. In the
second form the optimizer defaults to [`SGD`](@ref) with learning rate `lr` and gradient clip
`gclip`. In the third form `optimizers` should be a matching individual optimizer or
collection of optimizers.  The `weights` and possibly `gradients` and `optimizers` are
modified in-place.

Individual optimization parameters can be one of the following types. The keyword arguments
for each constructor and their default values are listed as well.

* [`SGD`](@ref)`(;lr=0.1, gclip=0)`
* [`Momentum`](@ref)`(;lr=0.05, gamma=0.95, gclip=0)`
* [`Nesterov`](@ref)`(;lr=0.05, gamma=0.95, gclip=0)`
* [`Adagrad`](@ref)`(;lr=0.05, eps=1e-6, gclip=0)`
* [`Rmsprop`](@ref)`(;lr=0.01, rho=0.9, eps=1e-6, gclip=0)`
* [`Adadelta`](@ref)`(;lr=1.0, rho=0.9, eps=1e-6, gclip=0)`
* [`Adam`](@ref)`(;lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, gclip=0)`

# Example:

    w = Param(rand(d), Adam())  # a Param with a specified optimizer
    g = lossgradient0(w)        # gradient g has the same shape as w
    update!(w, g)               # update w in-place with Adam()

    w = rand(d)                 # an individual weight array
    g = lossgradient1(w)        # gradient g has the same shape as w
    update!(w, g)               # update w in-place with SGD()
    update!(w, g; lr=0.1)       # update w in-place with SGD(lr=0.1)
    update!(w, g, SGD(lr=0.1))  # update w in-place with SGD(lr=0.1)

    w = (rand(d1), rand(d2))    # a tuple of weight arrays
    g = lossgradient2(w)        # g will also be a tuple
    p = (Adam(), SGD())         # p has optimizers for each w[i]
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
update!(x::Param, g) = (x.opt === nothing ? update!(x.value, g) : update!(x.value, g, x.opt))
update!(w::Param, g::Nothing)=w   # AutoGrad may return Nothing for a zero gradient


# Two arg version defaults to SGD
update!(w, g; lr=SGDLR, gclip=0) = update!(w, g, SGD(lr, gclip))
update!(w, g::Nothing; o...)=w

# This fallback takes care of arrays, tuples, iterators in general.
function update!(w,g,p)
    if isbitstype(eltype(w))
        error("Bad args: $((typeof(w),typeof(g),typeof(p)))")
    end
    if p isa SGD  # This comes from the 2-arg version
        p1 = p; p = (p1 for wi in w)
    end
    if !(length(w)==length(g)==length(p))
        error("weight, gradient, and optimization parameters not the same length.")
    end
    for (wi,gi,pi) in zip(w,g,p)
        update!(wi,gi,pi)
    end
end

update!(w, g::Nothing, p)=w

# We still need an extra method for Dict.
function update!(w::AbstractDict,g::AbstractDict,p::AbstractDict)
    for k in keys(g)            # g may have fewer keys than w
        update!(w[k],g[k],p[k])
    end
end

# Generic three arg version for float arrays
# Fix #579: leave g untyped, it can be Sparse.
update!(w::Array{T,N}, g, p) where {T<:Number,N} = gclip_update!(w, g, p)
update!(w::CuArray{T,N}, g, p) where {T,N} = gclip_update!(w, g, p)
update!(w::KnetArray{T,N}, g, p) where {T,N} = gclip_update!(w, g, p)

function gclip_update!(w, g, p)
    gclip!(g, p.gclip)          # gclip! supports AutoGrad.Sparse
    g = AutoGrad.full(g)
    _update!(w, g, p)
end

function _update!(w, g, p::SGD)
    axpy!(-p.lr, g, w)
end

function _update!(w, g, p::Momentum)
    if p.velocity===nothing; p.velocity=zero(w); end
    lmul!(p.gamma, p.velocity)
    axpy!(-p.lr, g, p.velocity)
    axpy!(1, p.velocity, w)
end

# https://arxiv.org/pdf/1212.0901.pdf Eq. (7)
function _update!(w, g, p::Nesterov)
    p.velocity ===nothing && (p.velocity = zero(w))
    lmul!(p.gamma, p.velocity)
    axpy!(-1, p.velocity, w)
    axpy!(-p.lr, g, p.velocity)
    axpy!(1+p.gamma, p.velocity, w)
end

function _update!(w, g, p::Adagrad)
    T = eltype(w)
    if p.G===nothing; p.G=zero(w); end
    axpy!(1, g .* g, p.G)
    axpy!(-p.lr, g ./ sqrt.(p.G .+ T(p.eps)), w)
end

function _update!(w, g, p::Rmsprop)
    T = eltype(w)
    if p.G===nothing; p.G=zero(w); end
    lmul!(p.rho, p.G)
    axpy!(1-p.rho, g .* g, p.G)
    axpy!(-p.lr, g ./ sqrt.(p.G .+ T(p.eps)), w)
end

function _update!(w, g, p::Adadelta)
    T = eltype(w)
    if p.G===nothing; p.G=zero(w); p.delta=zero(w); end
    lmul!(p.rho, p.G)
    axpy!(1-p.rho, g .* g, p.G)
    dw = g .* sqrt.(p.delta .+ T(p.eps)) ./ sqrt.(p.G .+ T(p.eps))
    lmul!(p.rho, p.delta)
    axpy!(1-p.rho, dw .* dw , p.delta)
    axpy!(-p.lr, dw, w)
end

function _update!(w, g, p::Adam)
    T = eltype(w)
    if p.fstm===nothing; p.fstm=zero(w); p.scndm=zero(w); end
    p.t += 1
    lmul!(p.beta1, p.fstm)
    axpy!(1-p.beta1, g, p.fstm)
    lmul!(p.beta2, p.scndm)
    axpy!(1-p.beta2, g .* g, p.scndm)
    fstm_corrected = p.fstm / T(1 - p.beta1 ^ p.t)
    scndm_corrected = p.scndm / T(1 - p.beta2 ^ p.t)
    axpy!(-p.lr, (fstm_corrected ./ (sqrt.(scndm_corrected) .+ T(p.eps))), w)
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
function optimizers(x...; o...)
    @warn "optimizers is deprecated, use sgd, adam etc. instead." maxlog=1
    _optimizers(x...; o...)
end

_optimizers(::KnetArray{<:Number},otype; o...) = otype(;o...)
_optimizers(::CuArray{<:Number},otype; o...) = otype(;o...)
_optimizers(::AbstractArray{<:Number},otype; o...) = otype(;o...)
_optimizers(a::AbstractDict,otype; o...)=Dict([ k=>_optimizers(v,otype;o...) for (k,v) in a ])
_optimizers(a::Tuple,otype; o...)=map(x->_optimizers(x,otype;o...), a)
_optimizers(a::AbstractArray,otype; o...)=map(x->_optimizers(x,otype;o...), a)
_optimizers(a,otype;o...)=nothing


