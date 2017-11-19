include("header.jl")

@testset "loss" begin
    for f in (logp, logsumexp)
        a = rand(10,10)
        @test gradcheck(f,a)
        @test gradcheck(f,a,1)
        @test gradcheck(f,a,2)
        if gpu() >= 0
            k = KnetArray(a)
            @test gradcheck(f,k)
            @test gradcheck(f,k,1)
            @test gradcheck(f,k,2)
            @test isapprox(f(a),f(k))
            @test isapprox(f(a,1),f(k,1))
            @test isapprox(f(a,2),f(k,2))
        end
    end
end

