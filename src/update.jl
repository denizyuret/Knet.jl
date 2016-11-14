"""

Stochastic gradient descent is an optimization technique to minimize 
an objective function paremeterized by a model's parameters.
It updates the parameters in the opposite direction of the gradient 
obtained by taking the gradient of the objective function w.r.t
the paremeters. The learning rate (lr) determines the size of the step.
It updates the weight with the following formula:

```
w = w - lr * g
```

where w is the weight, g is the gradient of the objective function w.r.t w and
lr is the learning rate.

** Arguments **
* `lr::AbstractFloat=0.001`

** Usage **

You can create the Sgd instance with default learning rate value or 
you can sepecify the learning rate. Then you can use the created instance in
the update! function.

```julia
#prms = Sgd()# use the default value
prms = Sgd(;lr=0.1)
update!(w, g, prms)

#You can change the lr later
prms.lr = 0.01
```
"""
type Sgd
	lr::AbstractFloat
end

Sgd(;lr=0.001) = Sgd(lr)

"""

Momentum method tries to accelerate the Sgd by adding a velocity term to the update.
It also decreases the oscilation between the opposite directions. It updates the weight
with the following formulas:

```
velocity = gamma * velocity + lr * g
w = w - velocity
```

where w is the weight, g is the gradient of the objective function w.r.t w,
lr is the learning rate, gamma is the momentum parameter, 
velocity is an array with the same size and type of w and holds the accelerated gradients.

** Arguments **
* `lr::AbstractFloat=0.001`
* `gamma::AbstractFloat=0.9`
* `velocity=zeros(w)`

** Usage **
You can create the Momentum instance with default values or 
you can sepecify the parameters. Then you can use the created instance in
the update! function.

```julia
#prms = Momentum() # use default values
prms = Momentum(;lr=0.1, gamma=0.95) # generally you do not need to specify the velocity parameter
update!(w, g, prms)

#You can change the parameters later
prms.lr=0.01
prms.gamma=0.9
```

Reference:

Qian, N. (1999). On the momentum term in gradient descent learning algorithms. 
Neural Networks : The Official Journal of the International Neural Network Society, 12(1), 145–151. http://doi.org/10.1016/S0893-6080(98)00116-6
"""
type Momentum
	lr::AbstractFloat
	gamma::AbstractFloat
	velocity
end

function Momentum(w; lr=0.001, gamma=0.9, velocity=zeros(w))
	@assert size(w) == size(velocity)
	Momentum(lr, gamma, velocity)
end

"""

Adagrad is one of the methods that adapts the learning rate to the parameters.
It stores the sum of the squares of the gradients to scale the learning rate.
The learning rate is adapted for each gradient by the value of current gradient
divided by the accumulated gradients. Hence, the learning rate is greater for
the parameters where the accumulated gradients are small and the learning rate is
small if the accumulated gradients are big. It updates the weight with the following formulas:

```
G = G + g .^ 2
w = w - g .* lr ./ sqrt(G + eps)
```

where w is the weight, g is the gradient of the objective function w.r.t w,
lr is the learning rate, G is an array with the same size and type of w and holds the 
sum of the squares of the gradients. eps is a small value to prevent the zero value
on the denominator.

** Arguments **
* `lr::AbstractFloat=0.001`
* `eps::AbstractFloat=1e-6`
* `G=zeros(w)`

** Usage **
You can create the Adagrad instance with default values or 
you can sepecify the parameters. Then you can use the created instance in
the update! function.

```julia
#prms = Adagrad() # use default values
prms = Adagrad(;lr=0.1) # generally you do not need to specify the G and eps parameters
update!(w, g, prms)

#You can change the parameters later
prms.lr=0.01
```

Reference:

Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
Journal of Machine Learning Research, 12, 2121–2159. Retrieved from http://jmlr.org/papers/v12/duchi11a.html
"""
type Adagrad
	lr::AbstractFloat
	eps::AbstractFloat
	G
end

function Adagrad(w; lr=0.001, eps=1e-6, G=zeros(w))
	@assert size(w) == size(G)
	Adagrad(lr, eps, G)
end

"""

Adadelta is an extension to Adagrad to preventing the convergence of the 
learning rate to zero with increase of time. Adadelta uses two ideas from Momentum and
Adagrad. It scales the learning rate based on the accumulated gradients and
holds the acceleration term like Momentum. It updates the weight with the following formulas:

```
G = (1-rho) * g .^ 2 + rho * G
update = g .* sqrt(delta + eps) ./ sqrt(G + eps)
w = w - lr * update
delta = rho * delta + (1-rho) * update .^ 2
```

where w is the weight, g is the gradient of the objective function w.r.t w,
lr is the learning rate, G is an array with the same size and type of w and holds the 
sum of the squares of the gradients. eps is a small value to prevent the zero value
on the denominator. rho is the momentum parameter and delta is an array with the same
size and type of w and holds the sum of the squared updates.

** Arguments **
* `lr::AbstractFloat=0.001`
* `rho::AbstractFloat=0.9`
* `eps::AbstractFloat=1e-6`
* `G=zeros(w)`
* `delta=zeros(w)`

** Usage **
You can create the Adadelta instance with default values or 
you can sepecify the parameters. Then you can use the created instance in
the update! function.

```julia
#prms = Adadelta() # use default values
prms = Adadelta(;lr=0.1, rho=0.8) # generally you do not need to specify the G, delta and eps parameters
update!(w, g, prms)

#You can change the parameters later
prms.lr=0.01
prms.rho=.9
prms.eps=1-e8
```

Reference:

Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. Retrieved from http://arxiv.org/abs/1212.5701
"""
type Adadelta
	lr::AbstractFloat
	rho::AbstractFloat
	eps::AbstractFloat
	G
	delta
end

function Adadelta(w; lr=0.001, rho=0.9, eps=1e-6, G=zeros(w), delta=zeros(w))
	@assert size(w) == size(G)
	Adadelta(lr, rho, eps, G, delta)
end

"""

Rmsprop is a similar method to Adadelta that tries to improve Adagrad. It scales
the learning rates by dividing the root mean squared of the gradients. It updates
the weight with the following formula:

```
G = (1-rho) * g .^ 2 + rho * G
w = w - lr * g ./ sqrt(G + eps)
```

where w is the weight, g is the gradient of the objective function w.r.t w,
lr is the learning rate, G is an array with the same size and type of w and holds the 
sum of the squares of the gradients. eps is a small value to prevent the zero value
on the denominator. rho is the momentum parameter and delta is an array with the same
size and type of w and holds the sum of the squared updates.

** Arguments **
* `lr::AbstractFloat=0.001`
* `rho::AbstractFloat=0.9`
* `eps::AbstractFloat=1e-6`
* `G=zeros(w)`

** Usage **
You can create the Rmsprop instance with default values or 
you can sepecify the parameters. Then you can use the created instance in
the update! function.

```julia
#prms = Rmsprop() # use default values
prms = Rmsprop(;lr=0.1, rho=0.9) # generally you do not need to specify the G and eps parameters
update!(w, g, prms)

#You can change the parameters later
prms.lr=0.01
prms.rho=0.8
prms.eps=1-e8
```

Reference:

Tieleman, Tijmen, and Geoffrey Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude." 
COURSERA: Neural Networks for Machine Learning 4.2 (2012).
"""
type Rmsprop
	lr::AbstractFloat
	rho::AbstractFloat
	eps::AbstractFloat
	G
end

function Rmsprop(w; lr=0.001, rho=0.9, eps=1e-6, G=zeros(w))
	@assert size(w) == size(G)
	Rmsprop(lr, rho, eps, G)
end

"""

Adam is one of the methods that compute the adaptive learning rate. It stores
accumulated gradients (first moment) and the sum of the squared of gradients (second).
It scales the first and second moment with increase of time. Here is the update formulas:

```
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g .* g
mhat = m ./ (1 - beta1 ^ t)
vhat = v ./ (1 - beta2 ^ t)
w = w - (lr / (sqrt(vhat) + eps)) * mhat
```

where w is the weight, g is the gradient of the objective function w.r.t w,
lr is the learning rate, m is an array with the same size and type of w and holds the 
accumulated gradients. v is an array with the same size and type of w and holds the 
sum of the squares of the gradients. eps is a small value to prevent the zero value
on the denominator. beta1 and beta2 are the parameters to calculate bias corrected
first and second moments. t is the update count.

** Arguments **
* `lr::AbstractFloat=0.001`
* `beta1::AbstractFloat=0.9`
* `beta2::AbstractFloat=0.999`
* `t::AbstractFloat=1`
* `eps::AbstractFloat=1e-8`
* `fstm=zeros(w)`
* `scndm=zeros(w)`

** Usage **
You can create the Adam instance with default values or 
you can sepecify the parameters. Then you can use the created instance in
the update! function.

```julia
#prms = Adam() # use default values
prms = Adam(;lr=0.1, beta1=0.95, beta2=0.995) # # generally you do not need to specify the fstm, scndm and eps parameters
update!(w, g, prms)

#You can change the parameters later
prms.lr=0.01
prms.beta1=0.8
prms.beta2=0.9
prms.eps=1e-9
```

Reference:

Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations, 1–13.
"""
type Adam
	lr::AbstractFloat
	beta1::AbstractFloat
	beta2::AbstractFloat
	t::AbstractFloat
	eps::AbstractFloat
	fstm
	scndm
end

function Adam(w; lr=0.001, beta1=0.9, beta2=0.999, t=1, eps=1e-8, fstm=zeros(w), scndm=zeros(w))
	@assert size(w) == size(fstm)
	@assert size(w) == size(scndm)
	Adam(lr, beta1, beta2, t, eps, fstm, scndm)
end

"""

```update!``` takes a weight array w, the gradient (g) of the objective function w.r.t w
and an instance of the parameters of an optimization method. It updates w depending on g
and the specified optimization method with parameters in-place. If the any parameters of
the optimization method need to be updated, it is also updated in-place. The function 
returns updated w and the instance of the parameters.

** Arguments **
* `w`
* `g`
* `prms::T`

T might be one of the following:

```julia
Sgd
Momentum
Adagrad
Adadelta
Rmsprop
Adam
```

** Usage **

```julia
#prms = Sgd()
#prms = Momentum()
#prms = Adagrad()
#prms = Adadelta()
#prms = Rmsprop()
prms = Adam()

w, prms = update!(w, g, prms)
#update!(w, g, prm)
```
"""
function update!(w, g, prms::Sgd)
	axpy!(-1 * prms.lr, g, w)
	return w, prms
end

function update!(w, g, prms::Momentum)
	prms.velocity = prms.gamma * prms.velocity
	axpy!(prms.lr, g, prms.velocity)
	axpy!(-1, prms.velocity, w)
	return w, prms
end

function update!(w, g, prms::Adam)
	prms.fstm = prms.beta1*prms.fstm
	axpy!(1-prms.beta1, g, prms.fstm)
	
	prms.scndm = prms.beta2*prms.scndm
	axpy!(1-prms.beta2, g .* g, prms.scndm)

	fstm_corrected = prms.fstm / (1 - prms.beta1 ^ prms.t) 
	scndm_corrected = prms.scndm / (1 - prms.beta2 ^ prms.t)
	
	axpy!(-1 * prms.lr, (fstm_corrected ./ (sqrt(scndm_corrected) + prms.eps)), w)

	prms.t = prms.t + 1
	return w, prms
end

function update!(w, g, prms::Adagrad)
	axpy!(1, g .* g, prms.G)
	axpy!(-1 * prms.lr, g ./ sqrt(prms.G + prms.eps), w)
	return w, prms
end

function update!(w, g, prms::Adadelta)
	prms.G = prms.rho*prms.G
	axpy!(1-prms.rho, g .* g, prms.G)
	update = g .* sqrt(prms.delta + prms.eps) ./ sqrt(prms.G + prms.eps)
	axpy!(-1 * prms.lr, update, w)

	prms.delta = prms.rho * prms.delta
	axpy!(1-prms.rho, update .* update , prms.delta)
	return w, prms
end

function update!(w, g, prms::Rmsprop)
	prms.G = prms.rho*prms.G
	axpy!(1-prms.rho, g .* g, prms.G)
	axpy!(-1 * prms.lr, g ./ sqrt(prms.G + prms.eps), w)
	return w, prms
end
