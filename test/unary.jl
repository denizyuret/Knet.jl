include("header.jl")
using SpecialFunctions
Random.seed!(42)

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

@testset "unary" begin
    for f in unary_fns
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
end

nothing

