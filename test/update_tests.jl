using Knet, Base.Test

# x* = f(1, · · · , 1)
# f(x*) = 0

rosenbrock(x) = sum((1-x[1:end-1]).^2 + 100*(x[2:end]-x[1:end-1].^2).^2)

grads = grad(rosenbrock)
srand(123456789)
w = convert(KnetArray, randn(100, 1))

function test_sgd(w)
	prev = 0
	i = 1

	params = SGDParams(0.00005, w)

	current = rosenbrock(params.w)
	while i <= 1000000 && abs(current - prev) > 1e-10 && current > 1e-4
		g = grads(params.w)
		sgd!(params, g)
		prev = current
		current = rosenbrock(params.w)

		i += 1
	end

	@test current <= 1e-4
	info("SGD Passed!\nConverged to $(current) at epoch $(i-1)")
end

function test_momentum(w)
	prev = 0
	i = 1

	params = MomentumParams(0.00005, 0.95, convert(KnetArray, zeros(size(w))), w)

	current = rosenbrock(params.w)

	while i <= 1000000 && abs(current - prev) > 1e-10 && current > 1e-4
		g = grads(params.w)
		momentum!(params, g)
		prev = current
		current = rosenbrock(params.w)

		i += 1
	end

	@test current <= 1e-4
	info("Momentum Passed!\nConverged to $(current) at epoch $(i-1)\n")
end

function test_adam(w)
	prev = 0
	i = 1

	params = AdamParams(0.001, 0.9, 0.95, 1, 1e-8, convert(KnetArray, zeros(size(w))), convert(KnetArray, zeros(size(w))), w)

	current = rosenbrock(params.w)

	while i <= 1000000 && abs(current - prev) > 1e-10 && current > 1e-4
		g = grads(params.w)
		adam!(params, g)
		prev = current
		current = rosenbrock(params.w)

		i += 1
	end

	@test current <= 1e-4
	info("Adam Passed!\nConverged to $(current) at epoch $(i-1)\n")
end

function test_adagrad(w)
	prev = 0
	i = 1

	params = AdagradParams(0.4, 1e-8, convert(KnetArray, zeros(size(w))), w)

	current = rosenbrock(params.w)

	while i <= 1000000 && abs(current - prev) > 1e-10 && current > 1e-4
		g = grads(params.w)
		adagrad!(params, g)
		prev = current
		current = rosenbrock(params.w)

		i += 1
	end

	@test current <= 1e-4
	info("Adagrad Passed!\nConverged to $(current) at epoch $(i-1)\n")
end


@time @test 1 == 1

#@time test_sgd(copy(w))
#@time test_momentum(copy(w))
@time test_adam(copy(w))
#@time test_adagrad(copy(w))
