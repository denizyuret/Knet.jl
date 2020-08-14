using Statistics, LinearAlgebra
using Knet.Ops20: bmm
using Knet.KnetArrays: KnetArray
using CUDA: CUDA, functional

sizes = [((2,4,3),(4,1,3)),((2,4,5),(4,8,5)),((2,8,4,3),(8,2,4,3))]
@testset "bmm" begin
    for t in (Float32, Float64)
        for s in sizes
            a = t(0.1)*randn(t, s[1]...)
            b = t(0.1)*randn(t, s[2]...)
            bmmul1(w)=bmm(w[1],w[2])
            bmmul2(w)=bmm(w[1],w[2]; transA=true)
            bmmul3(w)=bmm(w[1],w[2]; transB=true)
            bmmul4(w)=bmm(w[1],w[2]; transA=true, transB=true)
            pm(w) = ndims(w)==3 ? permutedims(w, (2,1,3)) : permutedims(w, (2,1,3,4)) 
            w = [a,b]
            @test gradcheck(bmmul1, w)
            w = [pm(a),b]
            @test gradcheck(bmmul2, w)
            w = [a,pm(b)]
            @test gradcheck(bmmul3, w)
            w = [pm(a),pm(b)]
            @test gradcheck(bmmul4, w)
            if CUDA.functional()
                c = bmm(a, b)
                ka = KnetArray(a)
                kb = KnetArray(b)
                kc = bmm(ka, kb)
                @test isapprox(c, Array(kc))
                w = [ka,kb]
                @test gradcheck(bmmul1, w)
            end
        end
    end
    if CUDA.functional()
        # Issue #495: transpose support
        ϵ =  1e-9
        A  = KnetArray(rand(3,4,10))
        At = permutedims(A,(2,1,3))
        B  = KnetArray(rand(4,5,10))
        Bt = permutedims(B,(2,1,3))
        C =  bmm(A,B)
        @test bmm(A,Bt;transB=true) ≈ C
        @test bmm(At,B;transA=true) ≈ C
        @test bmm(At,Bt;transA=true,transB=true) ≈ C
    end
end #end of testset

# suppress the return
nothing
