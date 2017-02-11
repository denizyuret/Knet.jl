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
        f1(w)=w[1]*w[2]
        @test gradcheck(f1, (a,b))
        @test gradcheck(f1, (ka,kb))
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
    end
end

nothing
