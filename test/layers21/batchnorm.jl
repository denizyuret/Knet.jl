using Test, Random, CUDA, Knet, AutoGrad, Statistics, Knet.Layers21

@testset "layers21/batchnorm" begin

    x, s, b, s2, b2, s3, b3 = (Param(CUDA.randn(Float64,x...)) for x in ((5,4,3,2),(1,1,3,1),(1,1,3,1),(5,4,3,1),(5,4,3,1),(5,1,1,1),(5,1,1,1)))

    # @gcheck needs Inference and Training to compute the same function
    # For this they need to use the same mean/var
    # @gcheck sums the output of non-scalar functions by default
    # This has the effect of multiplying scale with a 0 mean array
    # So we'll use getindex(y,1) instead
    function bnormtest(x,s,b; dims=(1,2,4), o...)
        m = mean(value(x); dims)
        v = var(value(x); mean=m,corrected=false,dims)
        bn = BatchNorm(; dims, mean=m, var=v, bias=b, scale=s, o...)
        getindex(bn(x), 1)
    end

    @test bnormtest(x,s,b) ≈ value(@diff(bnormtest(x,s,b)))
    @test @gcheck bnormtest(x,s,b)
    @test @gcheck bnormtest(x,s2,b2; dims=4)
    @test @gcheck bnormtest(x,s3,b3; dims=(2,3,4))
    @test @gcheck bnormtest(x,s,b; epsilon = 0)
    @test @gcheck bnormtest(x,s,b; momentum = 0.99)
end
