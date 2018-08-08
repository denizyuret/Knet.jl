include("header.jl")

@testset "loss" begin
    # for f in (logp, logsumexp)
    for f in (logp,)
        a = rand(10,10)
        @test gradcheck(f,a)
        if VERSION < v"0.7.0-DEV.4064"
            @test gradcheck(f,a,1)
            @test gradcheck(f,a,2)
        else
            # @test gradcheck(f,a, dims = 1)
            # @test gradcheck(f,a, dims = 2)
        end
            if gpu() >= 0
            k = KnetArray(a)
            # @test gradcheck(f,k)
            # if VERSION < v"0.7.0-DEV.4064"
            #     @test gradcheck(f,k,1)
            #     @test gradcheck(f,k,2)
            # else
            #     @test gradcheck(f,k, dims = 1)
            #     @test gradcheck(f,k, dims = 2)
            # end
            # @test isapprox(f(a),f(k))
            # @test isapprox(f(a,1),f(k,1))
            # @test isapprox(f(a,2),f(k,2))
        end
    end
end

