using CUDA, Knet, JLD2, FileIO, Test

@testset "jld2convert" begin
    f = joinpath(tempdir(), "test.jld2")
    struct CuStruct; a::CuArray; end
    struct KnetStruct; a::KnetArray; end

    for (atype, stype) in ((KnetArray, KnetStruct), (CuArray, CuStruct))

        # Single Array
        x = atype(rand(4))
        save(f, "x", x)
        @test x == load(f, "x")

        # In Array container
        a = [x, x, x]
        save(f, "a", a)
        @test a == load(f, "a")

        # In Tuple container
        t = (x, x, x)
        save(f, "t", t)
        @test t == load(f, "t")

        # In Dict container
        d = Dict(1=>x)
        save(f, "d", d)
        @test d == load(f, "d")

        # In Struct container
        s = stype(x)
        save(f, "s", s)
        @test x == load(f, "s").a
    end        
end
