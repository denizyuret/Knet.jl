include("header.jl")
srand(42)
nsample(a,n)=collect(a)[randperm(length(a))[1:n]]
using Base.BLAS: axpy!

@testset "linalg" begin
    for t in (Float32,Float64)
        a = rand(t,3,5)
        b = rand(t,5,2)
        mmul(w)=w[1]*w[2]
        @test gradcheck(mmul, (a,b))
        if gpu() >= 0
            c = a * b
            ka = KnetArray(a)
            kb = KnetArray(b)
            kc = ka * kb
            @test isapprox(c, Array(kc))
            @test gradcheck(mmul, (ka,kb))
        end

        if gpu() >= 0
            # cannot gradcheck axpy!, scal! overwriting
            d = rand(t,3,5)
            kd = KnetArray(d)
            r = rand()
            @test isapprox(axpy!(r,a,d), axpy!(r,ka,kd))
            @test isapprox(scale!(r,d), scale!(r,kd))
        end

        @test gradcheck(transpose, a)
        at = a'
        bt = b'
        s = a[1]
        mmul1(w)=w[1]'*w[2]
        mmul2(w)=w[1]*w[2]'
        mmul3(w)=w[1]'*w[2]'
        @test gradcheck(mmul1, Any[at,b])
        @test gradcheck(mmul2, Any[a,bt])
        @test gradcheck(mmul3, Any[at,bt])
        @test gradcheck(mmul1, Any[a,s])
        @test gradcheck(mmul1, Any[s,b])
        @test gradcheck(mmul2, Any[a,s])
        @test gradcheck(mmul2, Any[s,b])
        @test gradcheck(mmul3, Any[a,s])
        @test gradcheck(mmul3, Any[s,b])
        if gpu() >= 0
            kat = ka'
            kbt = kb'
            @test isapprox(at, Array(kat))
            @test gradcheck(transpose, ka)
            @test isapprox(kat'*kb, at'*b)
            @test isapprox(ka*kbt', a*bt')
            @test isapprox(kat'*kbt', at'*bt')
            @test gradcheck(mmul1, Any[kat,kb])
            @test gradcheck(mmul2, Any[ka,kbt])
            @test gradcheck(mmul3, Any[kat,kbt])
            @test gradcheck(mmul1, Any[ka,s])
            @test gradcheck(mmul1, Any[s,kb])
            @test gradcheck(mmul2, Any[ka,s])
            @test gradcheck(mmul2, Any[s,kb])
            @test gradcheck(mmul3, Any[ka,s])
            @test gradcheck(mmul3, Any[s,kb])
        end

        @test gradcheck(mat, a)
        if gpu() >= 0
            @test isapprox(mat(a), Array(mat(ka)))
            @test gradcheck(mat, ka)
        end

        for p in collect(permutations(1:2))
            p2(x) = permutedims(x,p)
            @test gradcheck(p2, a)
            if gpu() >= 0
                @test isapprox(p2(a), Array(p2(ka)))
                @test gradcheck(p2, ka)
            end
        end

        a3 = rand(2,3,4)
        if gpu() >= 0; k3 = KnetArray(a3); end
        for p in collect(permutations(1:3))
            p3(x) = permutedims(x,p)
            @test gradcheck(p3, a3)
            if gpu() >= 0
                @test isapprox(p3(a3), Array(p3(k3)))
                @test gradcheck(p3, k3)
            end
        end

        a4 = rand(2,3,4,5)
        if gpu() >= 0; k4 = KnetArray(a4); end
        for p in nsample(permutations(1:4),6)
            p4(x) = permutedims(x,p)
            @test gradcheck(p4, a4)
            if gpu() >= 0
                @test isapprox(p4(a4), Array(p4(k4)))
                @test gradcheck(p4, k4)
            end
        end

        a5 = rand(2,3,4,5,6)
        if gpu() >= 0; k5 = KnetArray(a5); end
        for p in nsample(permutations(1:5),6)
            p5(x) = permutedims(x,p)
            @test gradcheck(p5, a5)
            if gpu() >= 0
                @test isapprox(p5(a5), Array(p5(k5)))
                @test gradcheck(p5, k5)
            end
        end
    end
end

nothing
