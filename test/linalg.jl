if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using Knet

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
            # cannot gradcheck axpy!, overwriting
            d = rand(t,3,5)
            kd = KnetArray(d)
            r = rand()
            axpy!(r,a,d)
            axpy!(r,ka,kd)
            @test isapprox(d, Array(kd))
        end

        @test gradcheck(transpose, a)
        if gpu() >= 0
            t = a'
            kt = ka'
            @test isapprox(t, Array(kt))
            @test gradcheck(transpose, ka)
        end

        @test gradcheck(mat, a)
        if gpu() >= 0
            @test isapprox(mat(a), Array(mat(ka)))
            @test gradcheck(mat, ka)
        end

        for p in ([1,2], [2,1])
            p2(x) = permutedims(x,p)
            @test gradcheck(p2, a)
            if gpu() >= 0
                @test isapprox(p2(a), Array(p2(ka)))
                @test gradcheck(p2, ka)
            end
        end

        a3 = rand(2,3,4)
        if gpu() >= 0; k3 = KnetArray(a3); end
        for p in ([1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1])
            p3(x) = permutedims(x,p)
            @test gradcheck(p3, a3)
            if gpu() >= 0
                @test isapprox(p3(a3), Array(p3(k3)))
                @test gradcheck(p3, k3)
            end
        end
    end
end

nothing
