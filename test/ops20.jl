using Test
using Random: randn
using AutoGrad: @gcheck, Param
using Knet.Ops20: elu, relu, selu, sigm, dropout, bmm, conv4, conv4w, conv4x, deconv4, mat

@testset "ops20" begin
    x = Param(randn(3,4,5))
    @test @gcheck elu.(x)
    @test @gcheck relu.(x)
    @test @gcheck selu.(x)
    @test @gcheck sigm.(x)
    @test @gcheck dropout(x, 0.5; seed=1, drop=true)

    y = Param(randn(4,2,5))
    pd(x) = permutedims(x,(2,1,3))
    @test @gcheck bmm(x,y)
    @test @gcheck bmm(x,pd(y),transB=true)
    @test @gcheck bmm(pd(x),y,transA=true)
    @test @gcheck bmm(pd(x),pd(y),transA=true,transB=true)
    @test (bmm(x,y) ≈ bmm(x,pd(y),transB=true) ≈ bmm(pd(x),y,transA=true) ≈ bmm(pd(x),pd(y),transA=true,transB=true))

    w = Param(randn(3,3,2,4))
    x = Param(randn(8,8,2,3))
    for (p,s,d,m,a,g) in
        ((0,1,1,0,1,1),
         (1,1,1,0,1,1),
         (0,2,1,0,1,1),
         (0,1,2,0,1,1),
         (0,1,1,1,1,1),
         (0,1,1,0,2,1),
         # (0,1,1,0,1,2), # Grouped convolutions not yet implemented in NNlib, see https://github.com/JuliaGPU/CuArrays.jl/pull/523
         )
        y = Param(conv4(w,x,padding=p,stride=s,dilation=d,mode=m,alpha=a,group=g))
        @test @gcheck conv4(w,x,padding=p,stride=s,dilation=d,mode=m,alpha=a,group=g)
        @test @gcheck conv4w(w,x,y,padding=p,stride=s,dilation=d,mode=m,alpha=a,group=g)
        @test @gcheck conv4x(w,x,y,padding=p,stride=s,dilation=d,mode=m,alpha=a,group=g)
        @test @gcheck deconv4(w,y,padding=p,stride=s,dilation=d,mode=m,alpha=a,group=g)
    end    

    @test @gcheck mat(x)
    @test size(mat(x)) == (128,3)
    @test size(mat(x,dims=4)) == (384,1)
    @test size(mat(x,dims=3)) == (128,3)
    @test size(mat(x,dims=2)) == (64,6)
    @test size(mat(x,dims=1)) == (8,48)
    @test size(mat(x,dims=0)) == (1,384)
end

# TODO:
# test pool, poolx, unpool, unpoolx, loss.jl, rnn.jl, batchnorm.jl
