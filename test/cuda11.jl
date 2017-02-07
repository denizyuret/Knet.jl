using Base.Test, Knet

rand11(f,t,d...)=rand(t,d...)*t(0.8)+t(0.1)

cuda11flist = Any[]
for f in Knet.cuda11
    if isa(f,Tuple); f=f[2]; end
    in(f, ("invxback", "reluback", "sigmback", "tanhback")) && continue
    push!(cuda11flist, eval(parse(f)))
end

size11 = (1,(1,1),2,(2,1),(1,2),(2,2))

@testset "cuda11" begin
    for f in cuda11flist
        f1(x) = f(x[1],x[2])
        for t in (Float32, Float64)
            for n1 in size11, n2 in size11
                # @show f,t,n1,n2
                a1 = rand11(f,t,n1)
                a2 = rand11(f,t,n2)+t(1)
                @test gradcheck(f1, [a1, a2])
                if gpu() >= 0
                    g1 = KnetArray(a1)
                    g2 = KnetArray(a2)
                    @test isapprox(Array{t}(f(a1,a2)),Array{t}(f(g1,g2)))
                    @test gradcheck(f1, [g1, g2])
                end
            end
        end
    end
end

nothing
