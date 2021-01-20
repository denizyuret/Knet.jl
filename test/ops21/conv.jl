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
        channelmajor = false,
        crosscorrelation = false,
        dilation = 1,
        group = 1,
        padding = 0,
        stride = 1,
        usebias = false,
        usez = false,
        gcheck_test = true,
        approx_test = true,
    )
        cx, cy = 3*group, 2*group
        xd = rand(5:10, nd); xd[nd-1] = cx
        x = randn(xd...)
        w = randn((3*ones(Int,nd-2))..., cx÷group, cy)
        b = usebias ? randn(ones(Int,nd-2)...,cy,1) : nothing
        z = usez ? randn!(conv(w,x;activation,alpha,beta,channelmajor=false,crosscorrelation,dilation,group,padding,stride)) : nothing
        cm(x) = (x===nothing ? nothing : permutedims(x, (nd-1,(1:nd-2)...,nd)))
        if channelmajor; (x,w,b,z) = cm.((x,w,b,z)); end
        ax,aw,ab,az = (i->i===nothing ? nothing : atype(i)).((x,w,b,z))
        px,pw,pb,pz = (i->i===nothing ? nothing : Param(i)).((ax,aw,ab,az))
        r1 = ((isa(aw,Array) || !approx_test) ? true :
              isapprox(conv(w,x;bias=b,z=z,activation,alpha,beta,channelmajor,crosscorrelation,dilation,group,padding,stride),
                       Array(conv(aw,ax;bias=ab,z=az,activation,alpha,beta,channelmajor,crosscorrelation,dilation,group,padding,stride))))
        atol,rtol = (eltype(pw) == Float64 ? (0.01, 0.05) : (0.1, 0.5))
        r2 = !gcheck_test ? true : @gcheck conv(pw,px;bias=pb,z=pz,activation,alpha,beta,channelmajor,crosscorrelation,dilation,group,padding,stride) (;rtol,atol)
        r1 && r2
    end

    @test convtest(; )
    @test convtest(; nd=3)
    @test convtest(; nd=5)
    @test convtest(; activation=relu)
    @test convtest(; alpha=2)
    @test convtest(; beta=2, usez=true)
    @test convtest(; channelmajor=true)
    @test convtest(; crosscorrelation=true)
    @test convtest(; dilation=2)
    @test convtest(; group=2) 
    @test convtest(; padding=1)
    @test convtest(; stride=2)
    @test convtest(; usebias=true)

    if CUDA.functional()

        @test convtest(; atype=CuArray, )
        @test convtest(; atype=CuArray, nd=3)
        @test convtest(; atype=CuArray, nd=5)
        @test convtest(; atype=CuArray, activation=relu)
        @test convtest(; atype=CuArray, alpha=2)
        @test convtest(; atype=CuArray, beta=2, usez=true)
        # cudnn8 does not support Float64 with channelmajor
        @test convtest(; atype=CuArray{Float32}, channelmajor=true, gcheck_test=false)
        @test convtest(; atype=CuArray, crosscorrelation=true)
        @test convtest(; atype=CuArray, dilation=2)
        @test convtest(; atype=CuArray, group=2)
        @test convtest(; atype=CuArray, padding=1)
        @test convtest(; atype=CuArray, stride=2)
        @test convtest(; atype=CuArray, usebias=true)
        
        @test convtest(; atype=KnetArray, )
        @test convtest(; atype=KnetArray, nd=3)
        @test convtest(; atype=KnetArray, nd=5)
        @test convtest(; atype=KnetArray, activation=relu)
        @test convtest(; atype=KnetArray, alpha=2)
        @test convtest(; atype=KnetArray, beta=2, usez=true)
        @test convtest(; atype=KnetArray{Float32}, channelmajor=true, gcheck_test=false)
        @test convtest(; atype=KnetArray, crosscorrelation=true)
        @test convtest(; atype=KnetArray, dilation=2)
        @test convtest(; atype=KnetArray, group=2)
        @test convtest(; atype=KnetArray, padding=1)
        @test convtest(; atype=KnetArray, stride=2)
        @test convtest(; atype=KnetArray, usebias=true)

        # Multiple use of the same conv weight
        w = Param(CUDA.randn(Float64, 3,3,3,3))
        x = Param(CUDA.randn(Float64, 8,8,3,1))
        @test @gcheck conv(w,conv(w,x))
    end
end
