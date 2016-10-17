abstract UpdateParams

type SGDParams <: UpdateParams
	lr::AbstractFloat
end

type MomentumParams <: UpdateParams
	lr::AbstractFloat
	gamma::AbstractFloat
	velocity
end

function sgd!(weight, grad; lr=0.001)
	copy!(weight, weight - lr*grad)
end

function sgd!(weight, grad, params::SGDParams)
	sgd!(weight, grad; lr=params.lr)
end

function momentum!(weight, grad, velocity; lr=0.001, gamma=0.95)
	copy!(velocity, gamma * velocity + lr * grad)
	copy!(weight, weight - velocity)
end

function momentum!(weight, grad, params::MomentumParams)
	momentum!(weight, grad, params.velocity; lr=params.lr, gamma=params.gamma)
end
