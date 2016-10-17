using Knet, Base.Test

# x* = f(1, · · · , 1)
# f(x*) = 0

rosenbrock(x) = sum((1-x[1:end-1]).^2 + 100*(x[2:end]-x[1:end-1].^2).^2)

grads = grad(rosenbrock)
srand(123456789)

function test_sgd()
	w = randn(100, 1)
	prev = 0
	current = rosenbrock(w)
	i = 1

	params = SGDParams(0.00005)

	while i <= 1000000 && abs(current - prev) > 1e-10 && current > 1e-4
		g = grads(w)
		sgd!(w, g, params)
		prev = current
		current = rosenbrock(w)

		i += 1
	end

	@test rosenbrock(w) <= 1e-4
	info("SGD Passed!\nConverged to $(current) at epoch $(i-1)")
end

function test_momentum()
	w = randn(100, 1)
	prev = 0
	current = rosenbrock(w)
	i = 1

	params = MomentumParams(0.00005, 0.95, zeros(size(w)))

	while i <= 1000000 && abs(current - prev) > 1e-10 && current > 1e-4
		g = grads(w)
		momentum!(w, g, params)
		prev = current
		current = rosenbrock(w)

		i += 1
	end

	@test rosenbrock(w) <= 1e-4
	info("Momentum Passed!\nConverged to $(current) at epoch $(i-1)\n")
end


test_sgd()
test_momentum()
