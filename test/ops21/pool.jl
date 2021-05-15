using Test, Random, AutoGrad, Statistics
using Knet.Ops21: pool
using Knet.KnetArrays: KnetArray
using CUDA: CUDA, CuArray

@testset "ops21/pool" begin

    function pooltest(
        ; 
        atype = Array,
        nd = 4,
        op = maximum,
        window = 2,
        padding = 0,
        stride = window,
        propagateNaN = false,
        includePadding = false,
        channelmajor = false,
        alpha = 1,
        gcheck_test = true,
        approx_test = true,
    )
        xd = rand(5:10, nd)
        x = randn(xd...)
        kw = (; op, window, padding, stride, propagateNaN, includePadding, channelmajor, alpha)
        ax = atype(x)
        px = Param(ax)
        r1 = ((isa(ax,Array) || !approx_test) ? true :
              isapprox(pool(x; kw...), Array(pool(ax; kw...))))
        atol,rtol = (eltype(px) == Float64 ? (0.01, 0.05) : (0.1, 0.5))
        r2 = !gcheck_test ? true : @gcheck pool(px; kw...) (;rtol,atol)
        r1 && r2
    end

    @test pooltest(; )
    @test pooltest(; nd=3)
    @test pooltest(; nd=5)
    @test pooltest(; op=mean)
    @test pooltest(; window=3)
    @test pooltest(; window=typemax(Int))
    @test pooltest(; padding=1)
    @test pooltest(; stride=3)
    @test pooltest(; propagateNaN=true)
    @test pooltest(; includePadding=true, op=mean, padding=1)
    @test pooltest(; includePadding=false, op=mean, padding=1)
    @test pooltest(; channelmajor=true)
    @test pooltest(; alpha=2)

    if CUDA.functional()

        @test pooltest(; atype=CuArray)
        @test pooltest(; atype=CuArray, nd=3)
        @test pooltest(; atype=CuArray, nd=5)
        @test pooltest(; atype=CuArray, op=mean)
        @test pooltest(; atype=CuArray, window=3)
        @test pooltest(; atype=CuArray, window=typemax(Int))
        @test pooltest(; atype=CuArray, padding=1)
        @test pooltest(; atype=CuArray, stride=3)
        @test pooltest(; atype=CuArray, propagateNaN=true)
        @test pooltest(; atype=CuArray, includePadding=true, op=mean, padding=1)
        @test_skip pooltest(; atype=CuArray, includePadding=false, op=mean, padding=1)
        @test pooltest(; atype=CuArray, channelmajor=true)
        @test pooltest(; atype=CuArray, alpha=2)

        @test pooltest(; atype=KnetArray)
        @test pooltest(; atype=KnetArray, nd=3)
        @test pooltest(; atype=KnetArray, nd=5)
        @test pooltest(; atype=KnetArray, op=mean)
        @test pooltest(; atype=KnetArray, window=3)
        @test pooltest(; atype=KnetArray, window=typemax(Int))
        @test pooltest(; atype=KnetArray, padding=1)
        @test pooltest(; atype=KnetArray, stride=3)
        @test pooltest(; atype=KnetArray, propagateNaN=true)
        @test pooltest(; atype=KnetArray, includePadding=true, op=mean, padding=1)
        @test_skip pooltest(; atype=KnetArray, includePadding=false, op=mean, padding=1)
        @test pooltest(; atype=KnetArray, channelmajor=true)
        @test pooltest(; atype=KnetArray, alpha=2)

    end
end
