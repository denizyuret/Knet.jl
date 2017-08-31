include("header.jl")
srand(42)

function frand(f,t,d...)
    r = rand(t,d...)*t(0.5)+t(0.25)
    if in(f,(acosh,asec))
        return 1 ./ r
    else
        return r
    end
end

if VERSION >= v"0.6-"
    bcast(f)=(x->broadcast(f,x))
else
    bcast(f)=f
end

unary_fns = Any[]
for f in Knet.unary_ops
    if isa(f,Tuple); f=f[2]; end
    push!(unary_fns, eval(parse(f)))
end

@testset "unary" begin
    for f in unary_fns
        @show f
        bf = bcast(f)
        for t in (Float32, Float64)
            @show f,t
            sx = frand(f,t)
            @test isa(f(sx),t)
            @test gradcheck(f, sx)
            for n in (1,(1,1),2,(2,1),(1,2),(2,2))
                @show f,t,n
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
    # Non-broadcasting functions
    a = rand(10,10)
    @test gradcheck(dropout,a,0.5,kwargs=Dict(:seed=>1))
    @test gradcheck(logp,a)
    @test gradcheck(logp,a,1)
    @test gradcheck(logp,a,2)
    if gpu() >= 0
        k = KnetArray(a)
        @test gradcheck(dropout,k,0.5,kwargs=Dict(:seed=>1))
        @test gradcheck(logp,k)
        @test gradcheck(logp,k,1)
        @test gradcheck(logp,k,2)
    end
end

nothing

