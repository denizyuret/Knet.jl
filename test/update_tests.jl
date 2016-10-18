using Knet, Base.Test

# x* = f(1, · · · , 1)
# f(x*) = 0

rosenbrock(x) = sum((1-x[1:end-1]).^2 + 100*(x[2:end]-x[1:end-1].^2).^2)

grads = grad(rosenbrock)
srand(123456789)

function test_base(w, params, f)
	prev = 0
	i = 1

	current = rosenbrock(w)
	while i <= 1000000 && abs(current - prev) > 1e-10 && current > 1e-3
		g = grads(w)
		f(params, w, g)
		prev = current
		current = rosenbrock(w)

		i += 1
	end

	@test current <= 1e-3
	info("$f Passed!\nConverged to $(current) at epoch $(i-1)")
end

function test_sgd(w)
	params = SgdParams(0.0005)
	test_base(w, params, sgd!)
end

function test_momentum(w)
	params = MomentumParams(0.00005, 0.95, convert(typeof(w), zeros(size(w))))
	test_base(w, params, momentum!)
end

function test_adam(w)
	params = AdamParams(0.005, 0.9, 0.95, 1, 1e-8, convert(typeof(w), zeros(size(w))), convert(typeof(w), zeros(size(w))))
	test_base(w, params, adam!)
end

function test_adagrad(w)
	params = AdagradParams(0.35, 1e-6, convert(typeof(w), zeros(size(w))))
	test_base(w, params, adagrad!)
end

function test_adadelta(w)
	params = AdadeltaParams(0.001, 0.9, 1e-6, convert(typeof(w), zeros(size(w))), convert(typeof(w), zeros(size(w))))
	test_base(w, params, adadelta!)
end

function test_rmsprop(w)
	params = RmspropParams(0.0002, 0.9, 1e-6, convert(typeof(w), zeros(size(w))))
	test_base(w, params, rmsprop!)
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

