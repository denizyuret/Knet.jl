@testset "mlp" begin
     atype = Knet.Layers20.arrtype
     x =atype(randn(10,2))
     m = MLP(10,5,2;winit=randn,binit=zeros)
     y = m(x)
     @test true
end
