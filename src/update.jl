type Sgd
	lr::AbstractFloat
end

Sgd(;lr=0.001) = Sgd(lr)

type Momentum
	lr::AbstractFloat
	gamma::AbstractFloat
	velocity
end

function Momentum(w; lr=0.001, gamma=0.9, velocity=zeros(w))
	@assert size(w) == size(velocity)
	Momentum(lr, gamma, velocity)
end

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

type Adagrad
	lr::AbstractFloat
	eps::AbstractFloat
	G
end

function Adagrad(w; lr=0.001, eps=1e-6, G=zeros(w))
	@assert size(w) == size(G)
	Adagrad(lr, eps, G)
end

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

#sgd
function update!(w, g, prms::Sgd)
	axpy!(-1 * prms.lr, g, w)
	return w, prms
end

#Qian, N. (1999). On the momentum term in gradient descent learning algorithms. 
#Neural Networks : The Official Journal of the International Neural Network Society, 12(1), 145–151. http://doi.org/10.1016/S0893-6080(98)00116-6
function update!(w, g, prms::Momentum)
	prms.velocity = prms.gamma * prms.velocity
	axpy!(prms.lr, g, prms.velocity)
	axpy!(-1, prms.velocity, w)
	return w, prms
end

#Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations, 1–13.
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

#Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
#Journal of Machine Learning Research, 12, 2121–2159. Retrieved from http://jmlr.org/papers/v12/duchi11a.html
function update!(w, g, prms::Adagrad)
	axpy!(1, g .* g, prms.G)
	axpy!(-1 * prms.lr, g ./ sqrt(prms.G+ prms.eps), w)
	return w, prms
end

#Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. Retrieved from http://arxiv.org/abs/1212.5701
function update!(w, g, prms::Adadelta)
	prms.G = prms.rho*prms.G
	axpy!(1-prms.rho, g .* g, prms.G)
	update = g .* sqrt(prms.delta + prms.eps) ./ sqrt(prms.G + prms.eps)
	axpy!(-1 * prms.lr, update, w)

	prms.delta = prms.rho * prms.delta
	axpy!(1-prms.rho, update .* update , prms.delta)
	return w, prms
end

#http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
function update!(w, g, prms::Rmsprop)
	prms.G = prms.rho*prms.G
	axpy!(1-prms.rho, g .* g, prms.G)
	axpy!(-1 * prms.lr, g ./ sqrt(prms.G + prms.eps), w)
	return w, prms
end
