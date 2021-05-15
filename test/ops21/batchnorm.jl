using Test, Random, AutoGrad, Statistics
using Knet.Ops21: batchnorm
using Knet.KnetArrays: KnetArray
using CUDA: CUDA, CuArray

@testset "ops21/batchnorm" begin

    # @gcheck needs Inference and Training to compute the same function
    # For this they need to use the same mean/var
    # @gcheck sums the output of non-scalar functions by default
    # This has the effect of multiplying scale with a 0 mean array
    # So we'll use getindex(y,1) instead
    function bnormtest(; dims=(1,2,4), epsilon=1e-5, momentum=0.9, atype=Array, seed=-1)
        if seed > 0; Random.seed!(seed); end
        x = randn(5,4,3,2)
        m(x) = mean(x; dims)
        v(x) = var(x; dims, corrected=false)
        bsize = size(m(x))
        b = randn(bsize...)
        s = randn(bsize...)
        ax,ab,as = atype.((x,b,s))
        px,pb,ps = Param.((ax,ab,as))
        if atype == Array
            r1 = r2 = true
        else
            r1 = isapprox(Array(batchnorm(px, m(ax), v(ax), pb, ps; epsilon, momentum)), batchnorm(x, m(x), v(x), b, s; epsilon, momentum))
            r2 = isapprox(Array(batchnorm(px, m(ax), v(ax), pb, ps; epsilon, momentum, training=true)), batchnorm(x, m(x), v(x), b, s; epsilon, momentum, training=true))
        end
        r3 = isapprox(batchnorm(px, m(ax), v(ax), pb, ps; epsilon, momentum)[1], value(@diff(batchnorm(px, m(ax), v(ax), pb, ps; epsilon, momentum)[1])))
        r4 = @gcheck(batchnorm(px, m(ax), v(ax), pb, ps; epsilon, momentum)[1])
        r1 && r2 && r3 && r4
    end

    @test bnormtest()
    @test bnormtest(; dims=4)
    @test bnormtest(; dims=(2,3,4))
    @test bnormtest(; epsilon = 0)
    @test bnormtest(; momentum = 0.99)
    if CUDA.functional()
        @test bnormtest(; atype=CuArray)
        @test bnormtest(; dims=4, atype=CuArray)
        @test bnormtest(; dims=(2,3,4), atype=CuArray)
        @test bnormtest(; atype=KnetArray)
        @test bnormtest(; dims=4, atype=KnetArray)
        @test bnormtest(; dims=(2,3,4), atype=KnetArray)
    end
end
