using Knet, Base.Test
srand(123456789)

# x* = f(1, · · · , 1)
# f(x*) = 0

rosenbrock(x) = sum((1-x[1:end-1]).^2 + 100*(x[2:end]-x[1:end-1].^2).^2)

grads = gradloss(rosenbrock)
dims = 10
ftol = 1e-3
xtol = 1e-10
maxiter = 25000

function test_base(w, params)
    i = 1
    prev = 0
    current = 1
    while i <= maxiter && abs(current - prev) > xtol && current > ftol
	prev = current
	g, current = grads(w)
	w, params  = update!(w, g, params)
	i += 1
    end
    @test current <= ftol
    info("$(typeof(params)) for $(typeof(w)) $(current <= ftol ? "Passed" : "Failed")!\nConverged to $(current) at epoch $(i-1)")
end

function test_sgd(w)
    params = Sgd(;lr=0.0005)
    test_base(w, params)
end

function test_momentum(w)
    params = Momentum(w;lr=0.00025, gamma=0.95)
    test_base(w, params)
end

function test_adam(w)
    params = Adam(w; lr=0.005, beta1=0.9, beta2=0.95, t=1, eps=1e-8)
    test_base(w, params)
end

function test_adagrad(w)
    params = Adagrad(w; lr=0.35, eps=1e-6)
    test_base(w, params)
end

function test_adadelta(w)
    params = Adadelta(w; lr=0.01, rho=0.5, eps=1e-6)
    test_base(w, params)
end

function test_rmsprop(w)
    params = Rmsprop(w; lr=0.0005, rho=0.9, eps=1e-6)
    test_base(w, params)
end

@testset "update!" begin

#CPU Tests
    w = randn(dims)
    @time test_sgd(copy(w))
    @time test_momentum(copy(w))
    @time test_adam(copy(w))
    @time test_adagrad(copy(w))
    @time test_adadelta(copy(w))
    @time test_rmsprop(copy(w))

#GPU Tests
    if gpu() >= 0
        w = KnetArray(randn(dims))
        @time test_sgd(copy(w))
        @time test_momentum(copy(w))
        @time test_adam(copy(w))
        @time test_adagrad(copy(w))
        @time test_adadelta(copy(w))
        @time test_rmsprop(copy(w))
    end
end

nothing
