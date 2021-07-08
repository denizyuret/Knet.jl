@testset "rnn" begin
    arrtype = Knet.Layers20.arrtype
    x    = arrtype(zeros(10,1))
    ind  = rand(1:10)
    x[ind,1] = 1.0
    l = LSTM(input=10,hidden=5,embed=5)
    l([ind]).y
    l(x).y
    l = SRNN(input=10,hidden=5,embed=5)
    l([ind]).y
    l(x).y
    l = GRU(input=10,hidden=5,embed=5)
    l([ind]).y
    l(x).y
    @test true
end
