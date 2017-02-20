include("header.jl")

# Test KnetArray operations: cat, convert, copy, display, eachindex,
# eltype, endof, fill!, first, getindex, hcat, isempty, length,
# linearindexing, ndims, ones, pointer, rand!, reshape, setindex!,
# similar, size, stride, strides, summary, vcat, vec, zeros

if gpu() >= 0
    @testset "karray" begin
        a = rand(3,4)
        k = KnetArray(a)

        # getindex, setindex!
        # Index types: Integer, CartesianIndex, Vector{Int}, Array{Int}, EmptyArray, a:c, a:b:c, Colon, Bool
        # See http://docs.julialang.org/en/latest/manual/arrays.html#man-supported-index-types-1
        # TODO: check out http://docs.julialang.org/en/latest/manual/arrays.html#Cartesian-indices-1
        @testset "indexing" begin
            @test a == k
            for i in ((3,), (2,3), (:,), (:,:), (3:5,), (1:2,3:4), )
                # @show i
                @test a[i...] == k[i...]
                tmp = a[i...]; a[i...] = 0; k[i...] = 0
                @test a == k
                a[i...] = tmp; k[i...] = tmp
                @test gradcheck(getindex, a, i...)
                @test gradcheck(getindex, k, i...)
            end
            # make sure end works
            @test a[2:end] == k[2:end]
            @test a[2:end,2:end] == k[2:end,2:end]
        end

        # Unsupported indexing etc.:
        # @test_broken a[3:2:9] == Array(k[3:2:9]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::StepRange{Int64,Int64})
        # @test_broken a[1:2:3,1:3:4] == Array(k[1:2:3,1:3:4]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::StepRange{Int64,Int64}, ::StepRange{Int64,Int64})
        # @test_broken a[[1,5,3]] == Array(k[[1,5,3]]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::Array{Int64,1})
        # @test_broken a[[3,1],[4,2]] == Array(k[[3,1],[4,2]]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::Array{Int64,1}, ::Array{Int64,1})
        # @test_broken a[[]] == Array(k[[]]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::Array{Any,1})
        # @test_broken a[a.>0.5] == Array(k[k.>0.5]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::Knet.KnetArray{Float64,2})
        # @test_broken a[a.>0.5] == Array(k[a.>0.5]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::BitArray{2})
        # @test_broken cat((1,2),a,a) == Array(cat((1,2),k,k)) # cat only impl for i=1,2

        # AbstractArray interface
        @testset "abstractarray" begin

            for f in (copy, endof, first, isempty, length, ndims, ones, vec, zeros, 
                      a->(eachindex(a);0), a->(eltype(a);0), a->(Base.linearindexing(a);0),
                      a->collect(Float64,size(a)), a->collect(Float64,strides(a)), 
                      a->cat(1,a,a), a->cat(2,a,a), a->hcat(a,a), a->vcat(a,a), 
                      a->reshape(a,2,6), a->reshape(a,(2,6)), 
                      a->size(a,1), a->size(a,2),
                      a->stride(a,1), a->stride(a,2), )

                # @show f
                @test f(a) == f(k)
                @test gradcheck(f, a)
                @test gradcheck(f, k)
            end

            @test convert(Array{Float32},a) == convert(KnetArray{Float32},k)
            @test fill!(similar(a),pi) == fill!(similar(k),pi)
            @test fill!(similar(a,(2,6)),pi) == fill!(similar(k,(2,6)),pi)
            @test fill!(similar(a,2,6),pi) == fill!(similar(k,2,6),pi)
            @test isa(pointer(k), Ptr{Float64})
            @test isa(pointer(k,3), Ptr{Float64})
            @test isempty(KnetArray(Float32,0))
            @test rand!(copy(a)) != rand!(copy(k))
            @test k == k
            @test a == k
            @test k == a
            @test isapprox(k,k)
            @test isapprox(a,k)
            @test isapprox(k,a)
            @test a == copy!(similar(a),k)
            @test k == copy!(similar(k),a)
            @test k == copy!(similar(k),k)
            @test k == copy(k)
            @test pointer(k) != pointer(copy(k))
            @test k == deepcopy(k)
            @test pointer(k) != pointer(deepcopy(k))
        end

        @testset "cpu2gpu" begin
            # cpu/gpu xfer with grad support
            @test gradcheck(x->Array(sin(KnetArray(x))),a)
            @test gradcheck(x->KnetArray(sin(Array(x))),k)
        end
    end
end

nothing
