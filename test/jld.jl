using Test, FileIO, JLD2
using Knet.Ops20: RNN

@testset "JLD" begin

    #needed for load macro test: https://github.com/simonster/JLD2.jl/blob/cc56a4d6da116d6172a4ea89f4bec9d17154a0ba/test/loadsave.jl#L5L17
    fn = tempname()*".jld2"
    save(fn,"model",RNN(1,1))

    @eval begin
        function macro_load()
            @load $fn
            model2 = RNN(1,1)
            return typeof(model)==typeof(model2)
         end
    end

    #@test macro_load()


    function macro_save()
        model = RNN(1,1)
        @save fn model
        true
     end

    @test macro_save()
    @test macro_load()

    function fun_sl()
        model = RNN(1,1)
        save(fn,"model",model)
        model2 = load(fn,"model")
        typeof(model)==typeof(model2)
    end

    @test fun_sl()

    function collections_sl()
        model  = RNN(1,1)
        model2 = RNN(1,1)
        save(fn,"model",[model,model2])
        models = load(fn,"model")
        test1 = typeof(models[1])==typeof(model)
        save(fn,"model",(model,model2))
        models = load(fn,"model")
        test2 = typeof(models[1])==typeof(model)
        save(fn,"model",Dict("model"=>model,"model2"=>model2))
        models = load(fn,"model")
        test3 = typeof(models["model"])==typeof(model)
        save(fn,Dict("model"=>model,"model2"=>model2))
        models = load(fn)
        test3 = typeof(models["model"])==typeof(model)
        test1 && test2 && test3
    end

    @test collections_sl()

    rm(fn)
end
