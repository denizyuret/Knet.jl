using Test,Knet

@testset "dropout" begin
    dropout1(x,p)=dropout(x,p;seed=1)
    a = rand(10,10)
    @test gradcheck(dropout1,a,0.5)
    if gpu() >= 0
        k = KnetArray(a)
        @test gradcheck(dropout1,k,0.5)
        # This fails because seeds work differently on cpu vs gpu
        # @test isapprox(dropout1(k,0.5),dropout1(a,0.5))
        @test isapprox(vecnorm(dropout1(k,0.5)), vecnorm(dropout1(a,0.5)), rtol=0.1)
    end
end

