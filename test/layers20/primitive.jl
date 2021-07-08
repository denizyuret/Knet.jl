@testset "primitive" begin

    arrtype = Knet.Layers20.arrtype

    @testset "Multiply" begin
        inputDim = 10; outputDim = 2;
        m = Multiply(input=inputDim, output=outputDim, winit=randn)
        x = arrtype(zeros(inputDim, 1))
        y1 = m(x)
        y2 = arrtype(zeros(outputDim, 1))
        @test y1==y2
    end


    @testset "Embed" begin
        m  = Embed(input=10, output=2, winit=randn)
        x  = arrtype(zeros(10,1))
        ind = rand(1:10)
        x[ind,1] = 1.0
        y1 = m(x)
        y2 = m([ind])
        @test y1==y2
    end

    @testset "Bias" begin
        m  = Bias(10)
        @test size(m.b) == (10,)
        @test sum(m.b) == 0
     end

    @testset "Linear" begin
        inputDim=10; outputDim=3
        m = Linear(input=inputDim, output=outputDim, winit=randn,binit=zeros)
        x = arrtype(zeros(10,2))
        y = m(x)
        @test true
    end

    @testset "Dense" begin
        inputDim=10; outputDim=3
        m = Dense(input=inputDim, output=outputDim, winit=randn,binit=zeros)
        x = arrtype(zeros(10,2))
        y = m(x)
        @test true
    end
end
