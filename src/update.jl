"""

    Sgd(;lr=0.001)
    update!(w,g,p::Sgd)
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

SGD is used by default if no algorithm is specified in the two
argument version of `update!`[@ref].

"""
type Sgd
    lr::AbstractFloat
end

const SGDLR=0.001

Sgd(;lr=SGDLR)=Sgd(lr)


"""

    Momentum(;lr=0.001, gamma=0.9)
    update(w,g,p::Momentum)

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

Reference: [Qian,
N. (1999)](http://doi.org/10.1016/S0893-6080(98)00116-6). On the
momentum term in gradient descent learning algorithms.  Neural
Networks : The Official Journal of the International Neural Network
Society, 12(1), 145–151.

"""
type Momentum
    lr::AbstractFloat
    gamma::AbstractFloat
    velocity
end

Momentum(; lr=0.001, gamma=0.9)=Momentum(lr, gamma, nothing)


"""

    Adagrad(;lr=0.1, eps=1e-6)
    update(w,g,p::Adagrad)

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

Reference: [Duchi, J., Hazan, E., & Singer,
Y. (2011)](http://jmlr.org/papers/v12/duchi11a.html). Adaptive
Subgradient Methods for Online Learning and Stochastic Optimization.
Journal of Machine Learning Research, 12, 2121–2159.

"""
type Adagrad
    lr::AbstractFloat
    eps::AbstractFloat
    G
end

Adagrad(; lr=0.1, eps=1e-6)=Adagrad(lr, eps, nothing)


"""

    Adadelta(;lr=0.01, rho=0.9, eps=1e-6)
    update(w,g,p::Adadelta)

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

Reference: [Zeiler,
M. D. (2012)](http://arxiv.org/abs/1212.5701). ADADELTA: An Adaptive
Learning Rate Method.

"""
type Adadelta
	lr::AbstractFloat
	rho::AbstractFloat
	eps::AbstractFloat
	G
	delta
end

Adadelta(; lr=0.01, rho=0.9, eps=1e-6)=Adadelta(lr, rho, eps, nothing, nothing)


"""

    Rmsprop(;lr=0.001, rho=0.9, eps=1e-6)
    update(w,g,p::Rmsprop)

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

Reference: [Tijmen Tieleman and Geoffrey Hinton
(2012)](https://dirtysalt.github.io/images/nn-class-lec6.pdf). "Lecture
6.5-rmsprop: Divide the gradient by a running average of its recent
magnitude."  COURSERA: Neural Networks for Machine Learning 4.2.

"""
type Rmsprop
	lr::AbstractFloat
	rho::AbstractFloat
	eps::AbstractFloat
	G
end

Rmsprop(; lr=0.001, rho=0.9, eps=1e-6)=Rmsprop(lr, rho, eps, nothing)


"""

    Adam(;lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    update(w,g,p::Adam)

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

Reference: [Kingma, D. P., & Ba,
J. L. (2015)](https://arxiv.org/abs/1412.6980). Adam: a Method for
Stochastic Optimization. International Conference on Learning
Representations, 1–13.

"""
type Adam
    lr::AbstractFloat
    beta1::AbstractFloat
    beta2::AbstractFloat
    eps::AbstractFloat
    t::Int
    fstm
    scndm
end

Adam(; lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)=Adam(lr, beta1, beta2, eps, 0, nothing, nothing)


"""

    update!(weights, gradients, params)
    update!(weights, gradients; lr=0.001)

Update the `weights` using their `gradients` and the optimization
algorithm parameters specified by `params`.  The 2-arg version
defaults to the [`Sgd`](@ref) algorithm with learning rate `lr`.  The
`weights` and possibly `params` are modified in-place.

`weights` can be an individual numeric array or a collection of arrays
represented by an iterator or dictionary.  In the individual case,
`gradients` should be a similar numeric array of `size(weights)` and
`params` should be a single object.  In the collection case, each
individual weight array should have a corresponding params object.  In
the iterator case, `gradients` and `params` should be iterators of the
same length as `weights` with corresponding elements.  In the
dictionary case, `gradients` and `params` should be dictionaries with
the same keys as `weights`.  See [Optimizers](@ref) for a usage
example.

Individual optimization parameters can be one of the following types:
* [`Sgd`](@ref)`(;lr=0.001)`
* [`Momentum`](@ref)`(;lr=0.001, gamma=0.9)`
* [`Rmsprop`](@ref)`(;lr=0.001, rho=0.9, eps=1e-6)`
* [`Adagrad`](@ref)`(;lr=0.1, eps=1e-6)`
* [`Adadelta`](@ref)`(;lr=0.01, rho=0.9, eps=1e-6)`
* [`Adam`](@ref)`(;lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)`

"""
function update!{T<:AbstractFloat}(w::KorA{T}, g::KorA{T}, p::Sgd)
    axpy!(-p.lr, g, w)
end

function update!{T<:AbstractFloat}(w::KorA{T}, g::KorA{T}, p::Momentum)
    if p.velocity===nothing; p.velocity=zeros(w); end
    scale!(p.gamma, p.velocity)
    axpy!(p.lr, g, p.velocity)
    axpy!(-1, p.velocity, w)
end

function update!{T<:AbstractFloat}(w::KorA{T}, g::KorA{T}, p::Adam)
    if p.fstm===nothing; p.fstm=zeros(w); p.scndm=zeros(w); end
    p.t += 1
    scale!(p.beta1, p.fstm)
    axpy!(1-p.beta1, g, p.fstm)
    scale!(p.beta2, p.scndm)
    axpy!(1-p.beta2, g .* g, p.scndm)
    fstm_corrected = p.fstm / (1 - p.beta1 ^ p.t) 
    scndm_corrected = p.scndm / (1 - p.beta2 ^ p.t)
    axpy!(-p.lr, (fstm_corrected ./ (sqrt(scndm_corrected) + p.eps)), w)
end

function update!{T<:AbstractFloat}(w::KorA{T}, g::KorA{T}, p::Adagrad)
    if p.G===nothing; p.G=zeros(w); end
    axpy!(1, g .* g, p.G)
    axpy!(-p.lr, g ./ sqrt(p.G + p.eps), w)
end

function update!{T<:AbstractFloat}(w::KorA{T}, g::KorA{T}, p::Adadelta)
    if p.G===nothing; p.G=zeros(w); p.delta=zeros(w); end
    scale!(p.rho, p.G)
    axpy!(1-p.rho, g .* g, p.G)
    dw = g .* sqrt(p.delta + p.eps) ./ sqrt(p.G + p.eps)
    scale!(p.rho, p.delta)
    axpy!(1-p.rho, dw .* dw , p.delta)
    axpy!(-p.lr, dw, w)
end

function update!{T<:AbstractFloat}(w::KorA{T}, g::KorA{T}, p::Rmsprop)
    if p.G===nothing; p.G=zeros(w); end
    scale!(p.rho, p.G)
    axpy!(1-p.rho, g .* g, p.G)
    axpy!(-p.lr, g ./ sqrt(p.G + p.eps), w)
end

# This takes care of arrays, tuples, iterators in general.
function update!(w,g,p)
    if !(length(w)==length(g)==length(p))
        error("weight, gradient, and optimization parameters not the same length.")
    end
    if isbits(eltype(w))
        error("Bad args: $((typeof(w),typeof(g),typeof(p)))")
    end
    for (wi,gi,pi) in zip(w,g,p)
        update!(wi,gi,pi)
    end
end

# We still need an extra method for Dict.
function update!(w::Associative,g::Associative,p::Associative)
    for k in keys(w)
        update!(w[k],g[k],p[k])
    end
end

# Two arg version for the simple default Sgd update.
function update!(w,g;lr=SGDLR)
    if !(length(w)==length(g))
        error("weight, gradient not the same length.")
    end
    sgd = Sgd(lr)
    for (wi,gi) in zip(w,g)
        update!(wi,gi,sgd)
    end
end

function update!(w::Associative,g::Associative;lr=SGDLR)
    sgd = Sgd(lr)
    for k in keys(w)
        update!(w[k],g[k],sgd)
    end
end

# To distinuish the two-arg update! for the numeric weight arrays:
update!{T<:AbstractFloat}(w::KorA{T},g::KorA{T};lr=SGDLR)=update!(w,g,Sgd(lr))
