using Base.Test, Knet

function frand(f,t,d...)
    r = rand(t,d...)*t(0.8)+t(0.1)
    if in(f,(acosh,asec))
        return 1 ./ r
    else
        return r
    end
end

cuda1flist = Any[]
for f in Knet.cuda1
    if isa(f,Tuple); f=f[2]; end
    push!(cuda1flist, eval(parse(f)))
end
push!(cuda1flist, logp)
push!(cuda1flist, x->isa(x,Number)?zero(x):logp(x,1))
push!(cuda1flist, x->isa(x,Number)?zero(x):logp(x,2))

@testset "cuda1" begin
    for f in cuda1flist
        for t in (Float32, Float64)
            # @show f,t
            sx = frand(f,t)
            @test isa(f(sx),t)
            @test gradcheck(f, sx; rtol=0.01)
            for n in (1,(1,1),2,(2,1),(1,2),(2,2))
                # @show f,t,n
                ax = frand(f,t,n)
                @test gradcheck(f, ax; rtol=0.01)
                if gpu() >= 0
                    gx = KnetArray(ax)
                    cy = f(ax)
                    gy = f(gx)
                    @test isapprox(cy,Array(gy))
                    @test gradcheck(f, gx; rtol=0.01)
                end
            end
        end
    end
end

nothing

