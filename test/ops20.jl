using Test
using Random: randn
using AutoGrad: @gcheck, Param, value
using Knet.Ops20: elu, relu, selu, sigm, dropout, bmm, conv4, conv4w, conv4x, deconv4, mat, pool, poolx, unpool

@testset "ops20" begin
    x = Param(randn(8,8,2,3))
    @test @gcheck elu.(x)
    @test @gcheck relu.(x)
    @test @gcheck selu.(x)
    @test @gcheck sigm.(x)
    @test @gcheck dropout(x, 0.5; seed=1, drop=true)

    b = Param(randn(8,6,2,3))
    pd(x) = permutedims(x,(2,1,3,4))
    @test @gcheck bmm(x,b)
    @test @gcheck bmm(x,pd(b),transB=true)
    @test @gcheck bmm(pd(x),b,transA=true)
    @test @gcheck bmm(pd(x),pd(b),transA=true,transB=true)
    @test (bmm(x,b) ≈ bmm(x,pd(b),transB=true) ≈ bmm(pd(x),b,transA=true) ≈ bmm(pd(x),pd(b),transA=true,transB=true))

    function convtest(w,x;o...)
        y = Param(conv4(w,x;o...))
        dy = Param(randn(size(y)))
        ((@gcheck conv4(w,x;o...)) &&
         (@gcheck conv4w(w,x,dy;o...)) &&
         (@gcheck conv4x(w,x,dy;o...)) &&
         (@gcheck deconv4(w,y;o...)))
    end
    
    w = Param(randn(3,3,2,4))
    @test convtest(w, x; padding=0, stride=1, dilation=1, mode=0, alpha=1, group=1)
    @test convtest(w, x; padding=1, stride=1, dilation=1, mode=0, alpha=1, group=1)
    @test convtest(w, x; padding=0, stride=2, dilation=1, mode=0, alpha=1, group=1)
    @test convtest(w, x; padding=0, stride=1, dilation=2, mode=0, alpha=1, group=1)
    @test convtest(w, x; padding=0, stride=1, dilation=1, mode=1, alpha=1, group=1)
    @test convtest(w, x; padding=0, stride=1, dilation=1, mode=0, alpha=2, group=1)
    # Grouped convolutions not yet implemented in NNlib, see https://github.com/JuliaGPU/CuArrays.jl/pull/523
    @test_skip convtest(w, x; padding=0, stride=1, dilation=1, mode=0, alpha=1, group=2) 


    function pooltest(x; o...)
        y = Param(pool(x; o...))
        dy = Param(randn(size(y)))
        ((@gcheck pool(x; o...)) &&
         # @gcheck poolx(x,y,dy) misleads because NNlib gets confused when x&y don't change together.
         (@gcheck poolx(value(x), value(y), dy; o...)) &&
         (@gcheck unpool(x; o...)) &&
         true)
    end

    @test pooltest(x; window=2, stride=2, padding=0, mode=0, maxpoolingNanOpt=1, alpha=1)
    @test pooltest(x; window=3, stride=2, padding=0, mode=0, maxpoolingNanOpt=1, alpha=1)
    @test pooltest(x; window=2, stride=3, padding=0, mode=0, maxpoolingNanOpt=1, alpha=1)
    # Pool padding is buggy: https://github.com/FluxML/NNlib.jl/issues/229
    @test_skip pooltest(x; window=2, stride=2, padding=1, mode=0, maxpoolingNanOpt=1, alpha=1)
    @test pooltest(x; window=2, stride=2, padding=0, mode=1, maxpoolingNanOpt=1, alpha=1)
    @test pooltest(x; window=2, stride=2, padding=0, mode=2, maxpoolingNanOpt=1, alpha=1)
    @test pooltest(x; window=2, stride=2, padding=0, mode=3, maxpoolingNanOpt=1, alpha=1)
    @test pooltest(x; window=2, stride=2, padding=0, mode=0, maxpoolingNanOpt=0, alpha=1)
    @test pooltest(x; window=2, stride=2, padding=0, mode=0, maxpoolingNanOpt=1, alpha=2)

    @test @gcheck mat(x)
    @test size(mat(x)) == (128,3)
    @test size(mat(x,dims=4)) == (384,1)
    @test size(mat(x,dims=3)) == (128,3)
    @test size(mat(x,dims=2)) == (64,6)
    @test size(mat(x,dims=1)) == (8,48)
    @test size(mat(x,dims=0)) == (1,384)
end

# TODO:
# test unpool, unpoolx, loss.jl, rnn.jl, batchnorm.jl
