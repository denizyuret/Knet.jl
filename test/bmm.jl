include("header.jl")
using Statistics, LinearAlgebra

sizes = [((2,4,3),(4,1,3)),((2,4,5),(4,8,5)),((2,8,4,3),(8,2,4,3))]
@testset "bmm" begin
    for t in (Float32, Float64)
        for s in sizes
            a = t(0.1)*randn(t, s[1]...)
            b = t(0.1)*randn(t, s[2]...)
            bmmul(w)=bmm(w[1],w[2])
            w = [a,b]
            @test gradcheck(bmmul, w)
            if gpu() >= 0
                c = bmm(a, b)
                ka = KnetArray(a)
                kb = KnetArray(b)
                kc = bmm(ka, kb)
                @test isapprox(c, Array(kc))
                w = [ka,kb]
                @test gradcheck(bmmul, w)
            end
        end
    end
    # Issue #495: transpose support
    ϵ =  1e-9
    A  = KnetArray(rand(3,4,10))
    At = permutedims(A,(2,1,3))
    B  = KnetArray(rand(4,5,10))
    Bt = permutedims(B,(2,1,3))
    C  = KnetArray(rand(3,5,10))
    C =  bmm(A,B)
    @test mean(abs,bmm(A,Bt;transB=true) - C) < ϵ
    @test mean(abs,bmm(At,B;transA=true) - C) < ϵ
    @test mean(abs,bmm(At,Bt;transA=true,transB=true) - C) < ϵ
end #end of testset

# suppress the return
nothing
