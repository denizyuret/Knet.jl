using Test
using Random: randn
using AutoGrad: @gcheck, Param
using Knet.Ops21: gelu, mmul

# Allow the user to override this:
if !isdefined(Main, :testparam)
    testparam(x...) = Param(randn(x...))
end

@testset "ops21" begin

    @testset "activation" begin
        x4 = testparam(5,4,3,2)
        @gcheck gelu.(x4)
    end

    @testset "mmul" begin
        x4 = testparam(5,4,3,2)
        w4 = testparam(6,5,4,3)
        @gcheck mmul(w4,x4,dims=3)
        x3 = testparam(3,4,5)
        w3 = testparam(5,4,3)
        @gcheck mmul(w3,x3)
    end
end
