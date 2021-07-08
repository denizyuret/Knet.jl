@testset "loss" begin

    arrtype = Knet.Layers20.arrtype
    x = arrtype(randn(10,10))
    indices = rand(1:10,10)
    l1 = CrossEntropyLoss(dims=2)
    l1(x,indices);
    @test true

    #https://github.com/denizyuret/Knet.jl/pull/374
    #x = arrtype(rand(10))
    #indices = rand([0,1],10)
    #l2 = BCELoss()
    #l2(x,indices)
    #@test true

    #x = arrtype(randn(10))
    #indices = rand([-1,1],10)
    #l3 = LogisticLoss()
    #l3(x,indices)
    #@test true

end
