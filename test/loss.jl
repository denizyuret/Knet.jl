include("header.jl")

@testset "loss" begin
    for f in (logp, logsumexp, softmax)
        a = rand(10,10,10)
        @test gradcheck(f,a)
        @test gradcheck(f,a,kw=(:dims=>1,))
        @test gradcheck(f,a,kw=(:dims=>2,))
        if gpu() >= 0
            k = KnetArray(a)
            @test gradcheck(f,k)
            @test gradcheck(f,k,kw=(:dims=>1,))
            @test gradcheck(f,k,kw=(:dims=>2,))
            @test isapprox(f(a),f(k))
            @test isapprox(f(a,dims=1),f(k,dims=1))
            @test isapprox(f(a,dims=2),f(k,dims=2))
        end
    end

    a = rand(10,10)
    indices = rand(1:10,10)
    @test gradcheck(nll, a, indices, kw=(:dims=>1,), args=1)
    @test gradcheck(nll, a, indices, kw=(:dims=>2,), args=1)
    if gpu() >= 0
        k = KnetArray(a)
        @test gradcheck(nll, k, indices, kw=(:dims=>1,), args=1)
        @test gradcheck(nll, k, indices, kw=(:dims=>2,), args=1)
        @test isapprox(nll(k, indices, dims=1), nll(a, indices, dims=1))
        @test isapprox(nll(k, indices, dims=2), nll(a, indices, dims=2))
    end
    @test gradcheck(logistic,a[:],a[:])
    @test gradcheck(bce,a[:],a[:])
end

