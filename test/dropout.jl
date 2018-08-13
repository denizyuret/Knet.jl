include("header.jl")

@testset "dropout" begin
    dropout1(x,p)=dropout(x,p;seed=2)
    a = rand(100,100)
    @test gradcheck(dropout1,a,0.5)
    if gpu() >= 0
        k = KnetArray(a)
        @test gradcheck(dropout1,k,0.5)
        # This fails because seeds work differently on cpu vs gpu
        # @test isapprox(dropout1(k,0.5),dropout1(a,0.5))
        @test isapprox(sum(abs2,dropout1(k,0.5)), sum(abs2,dropout1(a,0.5)), rtol=0.01)
    end
end

