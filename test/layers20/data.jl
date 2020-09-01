import Knet.Layers20: IndexedDict
@testset "data" begin
    data = "this is an example data with example words"
    vocab = IndexedDict{String}()
    append!(vocab,split(data))
    vocab["example"]
    vocab[4]
    vocab[[1,2,3,4]]
    vocab[["example","this"]]
    push!(vocab,"a-new-word")
    vocab["a-new-word"]
    length(vocab)
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
