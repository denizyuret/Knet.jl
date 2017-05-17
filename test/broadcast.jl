include("header.jl")

rand11(f,t,d...)=rand(t,d...)*t(0.8)+t(0.1)
# we need symetric ones as well to test compare operations
#broadcast dim sizes chosen in the lower limits of given kernels
size12 = (((513,1025),(1,1025)),((1,1025),(513,1025)),#cuda13 vector-Ndim, first dim
          ((256,1),(256,1024)),((256,1024),(256,1)),#cuda14 vector-Ndim, other than first dim
          ((8,8,16,4),(8,8,1,4)),((8,8,16,4),(8,8,16,4)),#cuda16 3,4,5 dims generalised
          ((5,1,2,2,4,4,2),(5,5,1,2,4,4,1)),((5,5,1,2,4,4,1),(5,1,2,2,4,4,2)))#cuda17  more than 5 dim, generalised

size11 = (1,(1,1),2,(2,1),(1,2),(2,2))
# These are helper functions for gradients and rpow is used to define Array.^Number
# The former is tested during gradcheck, rpow is tested with .^ operation
exclude11 = ("invxback", "reluback", "sigmback", "tanhback", "rpow")

broadcast_fns = Any[]
for f in Knet.broadcast_ops
    if isa(f,Tuple); f=f[2]; end
    in(f, exclude11) && continue
    push!(broadcast_fns, eval(parse(f)))
end

srand(42)

@testset "broadcast" begin
    for f in broadcast_fns
        f1(x) = f(x[1],x[2])
        for t in (Float32, Float64)
            # multidim array broadcast
            # vector broadcast which is size bigger than 127 (more detail in src/broadcast.jl)
            if f == .+          # this takes too much time to perform on all functions
                for (n1,n2) in size12
                    #@show f,t,n1,n2
                    a1 = rand11(f,t,n1)
                    a2 = rand11(f,t,n2)+t(1)
                    if !(f in (max,min) && n1 != n2)
                        if t == Float64 # Float32 does not have enough precision for large arrays
                            @test gradcheck(f1, Any[a1, a2]; rtol=0.01)
                        end
                    end
                    if gpu() >= 0
                        g1 = KnetArray(a1)
                        g2 = KnetArray(a2)
                        @test isapprox(Array{t}(broadcast(f,a1,a2)),Array{t}(f(g1,g2)))
                        if t == Float64
                            @test gradcheck(f1, Any[g1, g2]; rtol=0.01)
                        end
                    end
                end
            end
            # Array-vector broadcast
            for n1 in size11, n2 in size11
                #@show f,t,n1,n2
                a1 = rand11(f,t,n1)
                a2 = rand11(f,t,n2)+t(1)
                if !(f in (max,min) && n1 != n2)      # max and min do not have broadcasting (different sized) versions defined in Base
                    @test gradcheck(f1, Any[a1, a2])  # 0.5 and 0.6 use max.(x,y) syntax, 0.4 can also using @compat
                end                                   # TODO: Fix this as part of general 0.6 compat work
                if gpu() >= 0
                    g1 = KnetArray(a1)
                    g2 = KnetArray(a2)
                    @test isapprox(Array{t}(broadcast(f,a1,a2)),Array{t}(f(g1,g2)))
                    @test gradcheck(f1, Any[g1, g2])
                end
            end
            # Scalar broadcast
            for n in size11
                #@show f,t,n,0
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
