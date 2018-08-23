include("header.jl")

@testset "JLD" begin

    #needed for load macro test: https://github.com/simonster/JLD2.jl/blob/cc56a4d6da116d6172a4ea89f4bec9d17154a0ba/test/loadsave.jl#L5L17
    fn = joinpath(tempdir(), "test.jld2")
    Knet.save(fn,"model",rnninit(1,1))

    @eval begin
        function macro_load()
            Knet.@load $fn
            model2 = rnninit(1,1)
            return all(typeof.(model).==typeof.(model2))
         end
    end

    #@test macro_load()


    function macro_save()
        model = rnninit(1,1)
        Knet.@save fn model
        return true
     end

    @test macro_save()
    @test macro_load()

    function fun_sl()
        model = rnninit(1,1)
        Knet.save(fn,"model",model)
        model2 = Knet.load(fn,"model")
        return all(typeof.(model).==typeof.(model))
    end

    @test fun_sl()

end
