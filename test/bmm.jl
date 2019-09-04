include("header.jl")
using Statistics, LinearAlgebra

sizes = [((2,4,3),(4,1,3)),((2,4,5),(4,8,5)),((2,8,4),(8,2,4))]
@testset "bmm" begin
    for t in (Float32, Float64)
        for s in sizes
            a = t(0.1)*randn(t, s[1]...)
            b = t(0.1)*randn(t, s[2]...)
            bmmul(w)=bmm(w[1],w[2])
            @test gradcheck(bmmul, (a,b))
            if gpu() >= 0
                c = bmm(a, b)
                ka = KnetArray(a)
                kb = KnetArray(b)
                kc = bmm(ka, kb)
                @test isapprox(c, Array(kc))
                @test gradcheck(bmmul, (ka,kb))
            end
        end
    end
end #end of testset

# suppress the return
nothing
