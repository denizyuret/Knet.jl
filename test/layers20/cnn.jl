@testset "cnn" begin

    arrtype = Knet.Layers20.arrtype

    @testset "conv" begin
        m = Conv(height=3, width=3, inout=3=>5, stride=1, padding=1, mode=1)
        x = arrtype(zeros(10,10,3,2))
        y = m(x)
        @test size(y) == (10,10,5,2)

        m = Conv(height=3, width=3, inout=3=>5, stride=1, padding=1, mode=1, binit=nothing, activation=nothing)
        x = arrtype(zeros(10,10,3,2))
        y = m(x)
        @test size(y) == (10,10,5,2)

        m = DeConv(height=3, width=3, inout=5=>3, stride=2, padding=1, mode=1)
        x = arrtype(zeros(10,10,5,2))
        y = m(x)
        @test size(y) == (19,19,3,2)

    end
end
