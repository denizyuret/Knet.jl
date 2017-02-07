using Base.Test, Knet

rand10(f,t,d...)=rand(t,d...)*t(0.8)+t(0.1)

cuda10flist = Any[+,-,*,/,\,.^]
for f in Knet.cuda10
    if isa(f,Tuple); f=f[2]; end
    if f=="stoa" || f=="atos"; continue; end
    push!(cuda10flist, eval(parse(f)))
end

no_sa = [ /, ]
no_as = [ \, ]

@testset "cuda10" begin
    for f in cuda10flist
        f1(x) = f(x[1],x[2])
        for t in (Float32, Float64)
            sx = rand10(f,t)+t(1)
            for n in (1,(1,1),2,(2,1),(1,2),(2,2))
                # @show f,t,n
                ax = rand10(f,t,n)
                !in(f,no_as) && (@test gradcheck(f1, [ax, sx]))
                !in(f,no_sa) && (@test gradcheck(f1, [sx, ax]))
                if gpu() >= 0
                    gx = KnetArray(ax)
                    !in(f,no_as) && (@test isapprox(Array{t}(f(ax,sx)),Array{t}(f(gx,sx))))
                    !in(f,no_sa) && (@test isapprox(Array{t}(f(sx,ax)),Array{t}(f(sx,gx))))
                    !in(f,no_sa) && (@test gradcheck(f1, [sx, gx]))
                    !in(f,no_as) && (@test gradcheck(f1, [gx, sx]))
                end
            end
        end
    end
end

nothing
