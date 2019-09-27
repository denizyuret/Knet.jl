include("header.jl")
using Statistics, LinearAlgebra

sizes = [((2,4,3),(4,1,3)),((2,4,5),(4,8,5)),((2,8,4,3),(8,2,4,3))]
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
    # Issue #495: transpose support
    using Knet: bmm!
    A = KnetArray(rand(3,4,10))
    At = permutedims(A,(2,1,3))
    B = KnetArray(rand(4,5,10))
    Bt = permutedims(B,(2,1,3))
    C = KnetArray(zeros(3,5,10))
    ϵ = 1e-9
    C1 = bmm!('N','N',1.0,A,B,0.0,copy(C))
    @test mean(abs,bmm!('N','T',1.0,A,Bt,0.0,copy(C))-C1)  <  1e-9
    @test mean(abs,bmm!('T','N',1.0,At,B,0.0,copy(C))-C1)  <  1e-9
    @test mean(abs,bmm!('T','T',1.0,At,Bt,0.0,copy(C))-C1) <  1e-9
end #end of testset

# suppress the return
nothing
