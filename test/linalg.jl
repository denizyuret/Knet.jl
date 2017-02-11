using Knet, Base.Test

@testset "linalg" begin
    for t in (Float32,Float64)
        a = rand(t,3,5)
        b = rand(t,5,2)
        c = a * b
        ka = KnetArray(a)
        kb = KnetArray(b)
        kc = ka * kb
        @test isapprox(c, Array(kc))
        mmul(w)=w[1]*w[2]
        @test gradcheck(mmul, (a,b))
        @test gradcheck(mmul, (ka,kb))

        d = rand(t,3,5)
        kd = KnetArray(d)
        r = rand()
        axpy!(r,a,d)
        axpy!(r,ka,kd)
        @test isapprox(d, Array(kd))

        t = a'
        kt = ka'
        @test isapprox(t, Array(kt))
        @test gradcheck(transpose, a)
        @test gradcheck(transpose, ka)

        @test isapprox(mat(a), Array(mat(ka)))
        @test gradcheck(mat, a)
        @test gradcheck(mat, ka)

        for p in ([1,2], [2,1])
            p2(x) = permutedims(x,p)
            @test isapprox(p2(a), Array(p2(ka)))
            @test gradcheck(p2, a)
            @test gradcheck(p2, ka)
        end

        a3 = rand(2,3,4)
        k3 = KnetArray(a3)
        for p in ([1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1])
            p3(x) = permutedims(x,p)
            @test isapprox(p3(a3), Array(p3(k3)))
            @test gradcheck(p3, a3)
            @test gradcheck(p3, k3)
        end
    end
end

nothing
