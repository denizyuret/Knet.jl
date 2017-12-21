include("header.jl")

@testset "loss" begin
    for f in (logp, logsumexp)
        a = rand(10,10)
        @test gradcheck(f,a)
        @test gradcheck(f,a,1)
        @test gradcheck(f,a,2)
        @test gradcheck(f,a,(1,2))

        if gpu() >= 0
            k = KnetArray(a)
            @test gradcheck(f,k)
            @test gradcheck(f,k,1)
            @test gradcheck(f,k,2)
            @test isapprox(f(a),f(k))
            @test isapprox(f(a,1),f(k,1))
            @test isapprox(f(a,2),f(k,2))
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
            for dims in [(1,2), (1,3), (3,1)]
                @test isapprox(f(a,dims),f(k,dims))
            end
        end
    end
end

