using Knet, Base.Test

# x* = f(1, · · · , 1)
# f(x*) = 0

rosenbrock(x) = sum((1-x[1:end-1]).^2 + 100*(x[2:end]-x[1:end-1].^2).^2)

grads = grad(rosenbrock)
srand(123456789)

function test_base(w, params)
	prev = 0
	i = 1

	current = rosenbrock(w)
	while i <= 1000000 && abs(current - prev) > 1e-10 && current > 1e-3
		g = grads(w)
		w, params = update!(w, g, params)
		prev = current
		current = rosenbrock(w)

		i += 1
	end

	@test current <= 1e-3
	info("$(typeof(params)) Passed!\nConverged to $(current) at epoch $(i-1)")
end

function test_sgd(w)
	params = Sgd(;lr=0.0005)
	test_base(w, params)
end

function test_momentum(w)
	params = Momentum(w;lr=0.00005, gamma=0.95, velocity=zeros(w))
	test_base(w, params)
end

function test_adam(w)
	params = Adam(w; lr=0.005, beta1=0.9, beta2=0.95, t=1, eps=1e-8, fstm=zeros(w), scndm=zeros(w))
	test_base(w, params)
end

function test_adagrad(w)
	params = Adagrad(w; lr=0.35, eps=1e-6, G=zeros(w))
	test_base(w, params)
end

function test_adadelta(w)
	params = Adadelta(w; lr=0.001, rho=0.9, eps=1e-6, G=zeros(w), delta=zeros(w))
	test_base(w, params)
end

function test_rmsprop(w)
	params = Rmsprop(w; lr=0.0002, rho=0.9, eps=1e-6, G=zeros(w))
	test_base(w, params)
end

#GPU Tests
info("GPU Tests")
raw = randn(100, 1)
w = convert(KnetArray, raw)
@time test_sgd(copy(w))
@time test_momentum(copy(w))
@time test_adam(copy(w))
@time test_adagrad(copy(w))
@time test_adadelta(copy(w))
@time test_rmsprop(copy(w))

#CPU Tests
gpu(-1)
info("CPU Tests")
w = randn(100, 1)
@time test_sgd(copy(w))
@time test_momentum(copy(raw))
@time test_adam(copy(raw))
@time test_adagrad(copy(raw))
@time test_adadelta(copy(raw))
@time test_rmsprop(copy(raw))

