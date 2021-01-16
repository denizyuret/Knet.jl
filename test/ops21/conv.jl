using Test, Random, AutoGrad, Statistics
using Knet.Ops21: conv, relu
using Knet.KnetArrays: KnetArray
using CUDA: CUDA, CuArray

@testset "ops21/conv" begin

    function convtest(
        ; atype = Array,
        nd = 4,
        activation = nothing,
        alpha = 1,
        beta = 0,
        dilation = 1,
        flipkernel = false,
        group = 1,
        padding = 0,
        stride = 1,
        usebias = false,
        usez = false,
        gcheck_only = false,
    )
        cx, cy = 3*group, 2*group
        xd = rand(5:10, nd); xd[nd-1] = cx
        x = randn(xd...)
        w = randn((3*ones(Int,nd-2))..., cx÷group, cy)
        b = usebias ? randn(ones(Int,nd-2)...,cy,1) : nothing
        z = usez ? randn!(conv(w,x;activation,alpha,beta,dilation,flipkernel,group,padding,stride)) : nothing
        ax,aw,ab,az = (i->i===nothing ? nothing : atype(i)).((x,w,b,z))
        px,pw,pb,pz = (i->i===nothing ? nothing : Param(i)).((ax,aw,ab,az))
        r1 = ((isa(aw,Array) || gcheck_only) ? true :
              isapprox(conv(w,x;activation,alpha,beta,dilation,flipkernel,group,padding,stride,bias=b,z=z),
                       Array(conv(aw,ax;activation,alpha,beta,dilation,flipkernel,group,padding,stride,bias=ab,z=az))))
        r2 = @gcheck conv(pw,px;activation,alpha,beta,dilation,flipkernel,group,padding,stride,bias=pb,z=pz)
        r1 && r2
    end

    @test convtest(; )
    @test convtest(; nd=3)
    @test convtest(; nd=5)
    @test convtest(; activation=relu)
    @test convtest(; alpha=2)
    @test convtest(; beta=2, usez=true)
    @test convtest(; dilation=2)
    @test convtest(; flipkernel=true)
    @test convtest(; padding=1)
    @test convtest(; stride=2)
    @test convtest(; usebias=true)
    @test_skip convtest(; group=2) # TODO: when group is fixed remove gcheck_only from below

    if CUDA.functional()

        @test convtest(; atype=CuArray, )
        @test convtest(; atype=CuArray, nd=3)
        @test convtest(; atype=CuArray, nd=5)
        @test convtest(; atype=CuArray, activation=relu)
        @test convtest(; atype=CuArray, alpha=2)
        @test convtest(; atype=CuArray, beta=2, usez=true)
        @test convtest(; atype=CuArray, dilation=2)
        @test convtest(; atype=CuArray, flipkernel=true)
        @test convtest(; atype=CuArray, padding=1)
        @test convtest(; atype=CuArray, stride=2)
        @test convtest(; atype=CuArray, usebias=true)
        @test convtest(; atype=CuArray, group=2, gcheck_only=true)
        
        @test convtest(; atype=KnetArray, )
        @test convtest(; atype=KnetArray, nd=3)
        @test convtest(; atype=KnetArray, nd=5)
        @test convtest(; atype=KnetArray, activation=relu)
        @test convtest(; atype=KnetArray, alpha=2)
        @test convtest(; atype=KnetArray, beta=2, usez=true)
        @test convtest(; atype=KnetArray, dilation=2)
        @test convtest(; atype=KnetArray, flipkernel=true)
        @test convtest(; atype=KnetArray, padding=1)
        @test convtest(; atype=KnetArray, stride=2)
        @test convtest(; atype=KnetArray, usebias=true)
        @test convtest(; atype=KnetArray, group=2, gcheck_only=true)

    end
end
