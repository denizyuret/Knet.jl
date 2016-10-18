abstract UpdateParams

type SGDParams <: UpdateParams
	lr::AbstractFloat
	w
end

type MomentumParams <: UpdateParams
	lr::AbstractFloat
	gamma::AbstractFloat
	velocity
	w
end

type AdamParams <: UpdateParams
	lr::AbstractFloat
	beta1::AbstractFloat
	beta2::AbstractFloat
	t::AbstractFloat
	eps::AbstractFloat
	fstm
	scndm
	w
end

function sgd!(params::SGDParams, grad)
	params.w = params.w - params.lr*grad
end

function momentum!(params::MomentumParams, grad)
	params.velocity = params.gamma * params.velocity + params.lr*grad
	params.w = params.w - params.velocity
end

function adam!(params::AdamParams, grad)
	params.fstm = params.beta1 * params.fstm + (1 - params.beta1)*grad
	params.scndm = params.beta2 * params.scndm + (1 - params.beta2)*(grad .^ 2)
	fstm_corrected = params.fstm / (1 - params.beta1 ^ params.t) 
	scndm_corrected = params.scndm / (1 - params.beta2 ^ params.t)
	params.w = params.w - params.lr * fstm_corrected ./ (sqrt(scndm_corrected) + params.eps)
	params.t = params.t + 1
end
