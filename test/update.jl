if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using Knet

# x* = f(1, · · · , 1)
# f(x*) = 0

rosenbrock(x) = sum((1-x[1:end-1]).^2 + 100*(x[2:end]-x[1:end-1].^2).^2)
rosengrad = gradloss(rosenbrock)
srand(123456789)
dims = 6

function rosenopt(w, params; verbose=false, ftol = 1e-3, xtol = 1e-10, maxiter = 12000)
    i = 1
    prev = 0
    current = 1
    t0 = time()
    while i <= maxiter && abs(current - prev) > xtol && current > ftol
	prev = current
	g, current = rosengrad(w)
	w, params  = update!(w, g, params)
	i += 1
    end
    t1 = time()
    if verbose
        @printf("%s: f=%f iter=%-5d time=%.2f type=%s opt=%s\n",
                (current <= ftol ? "PASS" : "FAIL"), 
                current, i-1, t1-t0, typeof(w), typeof(params))
    end
    return current <= ftol
end

@testset "update!" begin
    w = randn(dims) #CPU Tests
    @test rosenopt(copy(w),Sgd(lr=0.0005))
    @test rosenopt(copy(w),Momentum(w;lr=0.00025, gamma=0.95))
    @test rosenopt(copy(w),Adam(w; lr=0.005, beta1=0.9, beta2=0.95, t=1, eps=1e-8))
    @test rosenopt(copy(w),Adagrad(w; lr=0.35, eps=1e-6))
    @test rosenopt(copy(w),Adadelta(w; lr=0.01, rho=0.5, eps=1e-6))
    @test rosenopt(copy(w),Rmsprop(w; lr=0.0005, rho=0.9, eps=1e-6))
    if gpu() >= 0
        w = KnetArray(w) #GPU Tests
        @test rosenopt(copy(w),Sgd(lr=0.0005))
        @test rosenopt(copy(w),Momentum(w;lr=0.00025, gamma=0.95))
        @test rosenopt(copy(w),Adam(w; lr=0.005, beta1=0.9, beta2=0.95, t=1, eps=1e-8))
        @test rosenopt(copy(w),Adagrad(w; lr=0.35, eps=1e-6))
        @test rosenopt(copy(w),Adadelta(w; lr=0.01, rho=0.5, eps=1e-6))
        @test rosenopt(copy(w),Rmsprop(w; lr=0.0005, rho=0.9, eps=1e-6))
    end
end

nothing

