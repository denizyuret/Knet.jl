include("header.jl")

rand11(f,t,d...)=rand(t,d...)*t(0.8)+t(0.1)
size11 = (1,(1,1),2,(2,1),(1,2),(2,2))
# max and min do not have broadcasting versions defined in Base
# 0.5 and 0.6 use max.(x,y) syntax, 0.4 can also using @compat
# Fix this as part of general 0.6 compat work
# For now we will temporarily define them so the tests work
Base.max(x::Array,y::Array)=broadcast(max,x,y)
Base.min(x::Array,y::Array)=broadcast(min,x,y)
# These are helper functions for gradients and rpow is used to define Array.^Number
# The former is tested during gradcheck, rpow is tested with .^ operation
exclude = ("invxback", "reluback", "sigmback", "tanhback", "rpow")

broadcast_fns = Any[]
for f in Knet.broadcast_ops
    if isa(f,Tuple); f=f[2]; end
    in(f, exclude) && continue
    push!(broadcast_fns, eval(parse(f)))
end

@testset "broadcast" begin
    for f in broadcast_fns
        f1(x) = f(x[1],x[2])
        for t in (Float32, Float64)
            # Array broadcast
            for n1 in size11, n2 in size11
                # @show f,t,n1,n2
                a1 = rand11(f,t,n1)
                a2 = rand11(f,t,n2)+t(1)
                @test gradcheck(f1, Any[a1, a2])
                if gpu() >= 0
                    g1 = KnetArray(a1)
                    g2 = KnetArray(a2)
                    @test isapprox(Array{t}(f(a1,a2)),Array{t}(f(g1,g2)))
                    @test gradcheck(f1, Any[g1, g2])
                end
            end
            # Scalar broadcast
            for n in size11
                # @show f,t,n,0
                a = rand11(f,t,n)
                s = rand11(f,t)+t(1)
                @test gradcheck(f1, Any[a, s])
                @test gradcheck(f1, Any[s, a])
                if gpu() >= 0
                    g = KnetArray(a)
                    @test isapprox(Array{t}(f(a,s)),Array{t}(f(g,s)))
                    @test isapprox(Array{t}(f(s,a)),Array{t}(f(s,g)))
                    @test gradcheck(f1, Any[g, s])
                    @test gradcheck(f1, Any[s, g])
                end
            end
        end
    end
end

nothing
