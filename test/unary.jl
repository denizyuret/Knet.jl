if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using Knet

function frand(f,t,d...)
    r = rand(t,d...)*t(0.8)+t(0.1)
    if in(f,(acosh,asec))
        return 1 ./ r
    else
        return r
    end
end

unary_fns = Any[]
for f in Knet.unary_ops
    if isa(f,Tuple); f=f[2]; end
    push!(unary_fns, eval(parse(f)))
end
push!(unary_fns, logp)
push!(unary_fns, x->isa(x,Number)?zero(x):logp(x,1))
push!(unary_fns, x->isa(x,Number)?zero(x):logp(x,2))

@testset "unary" begin
    for f in unary_fns
        for t in (Float32, Float64)
            # @show f,t
            sx = frand(f,t)
            @test isa(f(sx),t)
            @test gradcheck(f, sx)
            for n in (1,(1,1),2,(2,1),(1,2),(2,2))
                # @show f,t,n
                ax = frand(f,t,n)
                @test gradcheck(f, ax)
                if gpu() >= 0
                    gx = KnetArray(ax)
                    cy = f(ax)
                    gy = f(gx)
                    @test isapprox(cy,Array(gy))
                    @test gradcheck(f, gx)
                end
            end
        end
    end
end

nothing

