using Base.Test, Knet

cuda1flist = Any[]
for f in Knet.cuda1
    if isa(f,Tuple); f=f[2]; end
    push!(cuda1flist, eval(parse(f)))
end
push!(cuda1flist, logp)
push!(cuda1flist, x->isa(x,Number)?zero(x):logp(x,1))
push!(cuda1flist, x->isa(x,Number)?zero(x):logp(x,2))

@testset "cuda1" begin
    for t in (Float32, Float64)
        sx = exp(rand(t))
        for n in (1,(1,1),2,(2,1),(1,2),(2,2))
            ax = exp(rand(t,n))
            if gpu() >= 0; gx = KnetArray(ax); end
            for f in cuda1flist
                # @show t,n,f
                @test gradcheck(f, ax)
                @test gradcheck(f, sx)
                @test isa(f(sx),t)
                if gpu() >= 0
                    cy = f(ax)
                    gy = f(gx)
                    @test isapprox(cy,Array(gy))
                    @test gradcheck(f, gx)
                end
            end
        end
    end
end # cuda1

nothing

