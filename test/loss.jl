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
        @test gradcheck(f,a,1)
        @test gradcheck(f,a,2)
        @test gradcheck(f,a,3)
        @test gradcheck(f,a,(1,2))
        @test gradcheck(f,a,(3,2))
        @test gradcheck(f,a,(1,3))
        
        if gpu() >= 0
            k = KnetArray(a)
            @test gradcheck(f,k)
            @test isapprox(f(a),f(k))
            for d in [1,2,3]
                @test gradcheck(f,k,d)
                @test isapprox(f(a,d),f(k,d))
            end
            for dims in [(1,2), (1,3), (3,1), [1,3,2]]
                @test isapprox(f(a,dims),f(k,dims))
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
end

