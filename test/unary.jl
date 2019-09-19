include("header.jl")
using SpecialFunctions
using Knet: reluback, sigmback, tanhback, invxback, eluback, seluback
using Knet: lgamma, loggamma    # TODO: delete after everyone has SpecialFunctions 0.8

@testset "unary" begin

    function frand(f,t,d...)
        r = rand(t,d...) .* t(0.5) .+ t(0.25)
        if in(f,(acosh,asec))
            return 1 ./ r
        else
            return r
        end
    end

    bcast(f)=(x->broadcast(f,x))

    unary_fns = Any[]
    for f in Knet.unary_ops
        if isa(f,Tuple); f=f[2]; end
        push!(unary_fns, eval(Meta.parse(f)))
    end

    skip_grads = [trigamma]
    for f in unary_fns
        f in skip_grads && continue
        #@show f
        bf = bcast(f)
        for t in (Float32, Float64)
            #@show f,t
            sx = frand(f,t)
            @test isa(f(sx),t)
            @test gradcheck(f, sx)
            for n in (1,(1,1),2,(2,1),(1,2),(2,2))
                #@show f,t,n
                ax = frand(f,t,n)
                @test gradcheck(bf, ax)
                if gpu() >= 0
                    gx = KnetArray(ax)
                    cy = bf(ax)
                    gy = bf(gx)
                    @test isapprox(cy,Array(gy))
                    @test gradcheck(bf, gx)
                end
            end
        end
    end

    # Issue #456: 2nd derivative for MLP
    for trygpu in (false, true)
        trygpu && gpu() < 0 && continue
        (x,y,dy) = randn.((10,10,10))
        if trygpu; (x,y,dy) = KnetArray.((x,y,dy)); end
        (x,y,dy) = Param.((x,y,dy))
        for f in (relu,sigm,tanh,invx,selu,elu)
            f1(x) = f.(x); @test @gcheck f1(x)
            f1i(x,i) = f1(x)[i]; @test @gcheck f1i(x,1)
            g1i(x,i) = grad(f1i)(x,i); @test @gcheck g1i(x,1)
            g1ij(x,i,j) = g1i(x,i)[j]; @test @gcheck g1ij(x,1,1)
            h1ij(x,i,j) = grad(g1ij)(x,i,j); if h1ij(x,1,1) != nothing; @test @gcheck h1ij(x,1,1); end
        end
        @test @gcheck reluback.(dy,y)
        @test @gcheck sigmback.(dy,y)
        @test @gcheck tanhback.(dy,y)
        @test @gcheck invxback.(dy,y)
        @test @gcheck seluback.(dy,y)
        @test @gcheck  eluback.(dy,y)
    end
end

nothing
