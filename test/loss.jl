include("header.jl")

@testset "loss" begin
    for f in (logp, logsumexp)
        a = rand(10,10)
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
        
        a = rand(10,10,10)
        @test gradcheck(f,a)
        @test gradcheck(f,a,kw=(:dims=>1,))
        @test gradcheck(f,a,kw=(:dims=>2,))
        @test gradcheck(f,a,kw=(:dims=>3,))
        @test gradcheck(f,a,kw=(:dims=>(1,2),))
        @test gradcheck(f,a,kw=(:dims=>(3,2),))
        @test gradcheck(f,a,kw=(:dims=>(1,3),))
        
        if gpu() >= 0
            k = KnetArray(a)
            @test gradcheck(f,k)
            @test isapprox(f(a),f(k))
            for d in [1,2,3]
                @test gradcheck(f,k,kw=(:dims=>d,))
                @test isapprox(f(a,dims=d),f(k,dims=d))
            end
            for dims in [(1,2), (1,3), (3,1), [1,3,2]]
                @test isapprox(f(a,dims=dims),f(k,dims=dims))
            end
        end

        a = rand(10,10, 10)
        for d in [1, 2, 3, (1,2), (1,3), [2,3], [1,2,3]]
            @test softmax(a, dims=d) ≈ exp.(logsoftmax(a, dims=d))
            @test all(sum(softmax(a, dims=d), dims=d) .≈ 1)
            if gpu() > 0
                k = KnetArray(a)
                @test softmax(k, dims=d) ≈ exp.(logsoftmax(k, dims=d))
                @test all(sum(softmax(k, dims=d), dims=d) .≈ 1)
            end
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
