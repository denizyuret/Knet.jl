include("header.jl")
using Printf

# x* = f(1, · · · , 1)
# f(x*) = 0

rosenbrock(x) = sum(abs2.(1 .- x[1:end-1]) .+ 100 .* abs2.(x[2:end] .- abs2.(x[1:end-1])))

function rosenmulti(x)
    v = value(x)
    if isbitstype(eltype(v))
        rosenbrock(x)
    elseif isa(v, AbstractDict)
        rosenbrock(x[:a]) + rosenbrock(x[:b])
    else
        rosenbrock(x[1]) + rosenbrock(x[2])
    end
end

rosengrad = gradloss(rosenmulti)
Random.seed!(123456789) # TODO: tests sensitive to random seed ???

function rosenopt(w, params; verbose=false, ftol = 1e-3, xtol = 1e-10, maxiter = 12000)
    i = 1
    prev = 0
    current = 1
    t0 = time()
    while i <= maxiter && abs(current - prev) > xtol && current > ftol
	prev = current
	g, current = rosengrad(w)
	update!(w, g, params)
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

    dims = 6
    w = randn(dims)
    # CPU Tests
    @test rosenopt(copy(w),Sgd(lr=0.0005))
    @test rosenopt(copy(w),Momentum(lr=0.00025, gamma=0.95))
    @test rosenopt(copy(w),Nesterov(lr=0.00025, gamma=0.95))
    @test rosenopt(copy(w),Adam(lr=0.005, beta1=0.9, beta2=0.95, eps=1e-8))
    @test rosenopt(copy(w),Adagrad(lr=0.35, eps=1e-6))
    @test rosenopt(copy(w),Adadelta(lr=0.01, rho=0.5, eps=1e-6))
    @test rosenopt(copy(w),Rmsprop(lr=0.0005, rho=0.9, eps=1e-6))
    adam() = Adam(lr=0.005, beta1=0.9, beta2=0.95, eps=1e-8)
    v = 10*randn(dims)
    @test rosenopt((copy(w),copy(v)), (adam(),adam()))
    @test rosenopt(Any[copy(w),copy(v)], [adam(),adam()])
    @test rosenopt(Dict(:a=>copy(w),:b=>copy(v)), Dict(:a=>adam(),:b=>adam()))
    if gpu() >= 0
        Knet.gc()
        w = KnetArray(w) #GPU Tests
        @test rosenopt(copy(w),Sgd(lr=0.0005))
        @test rosenopt(copy(w),Momentum(lr=0.00025, gamma=0.95))
        @test rosenopt(copy(w),Nesterov(lr=0.00025, gamma=0.95))
        @test rosenopt(copy(w),Adam(lr=0.005, beta1=0.9, beta2=0.95, eps=1e-8))
        @test rosenopt(copy(w),Adagrad(lr=0.35, eps=1e-6))
        @test rosenopt(copy(w),Adadelta(lr=0.01, rho=0.5, eps=1e-6))
        @test rosenopt(copy(w),Rmsprop(lr=0.0005, rho=0.9, eps=1e-6))
        v = KnetArray(v)
        @test rosenopt((copy(w),copy(v)), (adam(),adam()))
        @test rosenopt(Any[copy(w),copy(v)], [adam(),adam()])
        @test rosenopt(Dict(:a=>copy(w),:b=>copy(v)), Dict(:a=>adam(),:b=>adam()))
    end
end

nothing
