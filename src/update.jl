abstract AbsParams

type SgdParams <: AbsParams
	lr::AbstractFloat
end

type MomentumParams <: AbsParams
	lr::AbstractFloat
	gamma::AbstractFloat
	velocity
end

type AdamParams <: AbsParams
	lr::AbstractFloat
	beta1::AbstractFloat
	beta2::AbstractFloat
	t::AbstractFloat
	eps::AbstractFloat
	fstm
	scndm
end

type AdagradParams <: AbsParams
	lr::AbstractFloat
	eps::AbstractFloat
	G
end

type AdadeltaParams <: AbsParams
	lr::AbstractFloat
	rho::AbstractFloat
	eps::AbstractFloat
	G
	delta
end

type RmspropParams <: AbsParams
	lr::AbstractFloat
	rho::AbstractFloat
	eps::AbstractFloat
	G
end

function sgd!(params::AbsParams, w, g)
	axpy!(-1 * params.lr, g, w)
end


#Qian, N. (1999). On the momentum term in gradient descent learning algorithms. 
#Neural Networks : The Official Journal of the International Neural Network Society, 12(1), 145–151. http://doi.org/10.1016/S0893-6080(98)00116-6
function momentum!(params::MomentumParams, w, g)
	params.velocity = params.gamma * params.velocity
	axpy!(params.lr, g, params.velocity)
	axpy!(-1, params.velocity, w)
end

#Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations, 1–13.
function adam!(params::AdamParams, w, g)
	params.fstm = params.beta1*params.fstm
	axpy!(1-params.beta1, g, params.fstm)
	
	params.scndm = params.beta2*params.scndm
	axpy!(1-params.beta2, g .* g, params.scndm)

	fstm_corrected = params.fstm / (1 - params.beta1 ^ params.t) 
	scndm_corrected = params.scndm / (1 - params.beta2 ^ params.t)
	
	axpy!(-1 * params.lr, (fstm_corrected ./ (sqrt(scndm_corrected) + params.eps)), w)

	params.t = params.t + 1
end

#Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
#Journal of Machine Learning Research, 12, 2121–2159. Retrieved from http://jmlr.org/papers/v12/duchi11a.html
function adagrad!(params::AdagradParams, w, g)
	axpy!(1, g .* g, params.G)
	axpy!(-1 * params.lr, g ./ sqrt(params.G+ params.eps), w)
end

#Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. Retrieved from http://arxiv.org/abs/1212.5701
function adadelta!(params::AdadeltaParams, w, g)
	params.G = params.rho*params.G
	axpy!(1-params.rho, g .* g, params.G)
	update = g .* sqrt(params.delta + params.eps) ./ sqrt(params.G + params.eps)
	axpy!(-1 * params.lr, update, w)

	params.delta = params.rho * params.delta
	axpy!(1-params.rho, update .* update , params.delta)
end

#http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
function rmsprop!(params::RmspropParams, w, g)
	params.G = params.rho*params.G
	axpy!(1-params.rho, g .* g, params.G)
	axpy!(-1 * params.lr, g ./ sqrt(params.G + params.eps), w)
end
