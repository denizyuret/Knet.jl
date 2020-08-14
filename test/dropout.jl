using Test, Random
using Knet.Ops20: dropout
using AutoGrad: gradcheck
using CUDA: CUDA, functional
using Knet.KnetArrays: KnetArray

@testset "dropout" begin
    dropout1(x,p)=dropout(x,p;seed=1,drop=true)
    a = rand(Random.MersenneTwister(2),100,100)
    @test gradcheck(dropout1,a,0.5; args=1)
    if CUDA.functional()
        k = KnetArray(a)
        @test gradcheck(dropout1,k,0.5; args=1)
        # This fails because seeds work differently on cpu vs gpu
        # @test isapprox(dropout1(k,0.5),dropout1(a,0.5))
        @test isapprox(sum(abs2,dropout1(k,0.5)), sum(abs2,dropout1(a,0.5)), rtol=0.1)
    end
end

