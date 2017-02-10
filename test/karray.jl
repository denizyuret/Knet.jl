using Base.Test, Knet

# Test KnetArray operations: cat, convert, copy, display, eachindex,
# elsize, eltype, endof, fill!, first, getindex, hcat, isempty,
# length, linearindexing, ndims, ones, pointer, rand!, reshape,
# setindex!, similar, size, stride, strides, summary, vcat, vec, zeros

@testset "karray" begin
    a = rand(3,4)
    k = KnetArray(a)
    @test a == Array(k)

    # getindex, setindex!
    # Index types: Integer, CartesianIndex, Vector{Int}, Array{Int}, EmptyArray, a:c, a:b:c, Colon, Bool
    # See http://docs.julialang.org/en/latest/manual/arrays.html#man-supported-index-types-1
    # TODO: check out http://docs.julialang.org/en/latest/manual/arrays.html#Cartesian-indices-1
    @test a[3] == k[3]
    @test (a[3] = 1; k[3] = 1; a == Array(k))
    @test a[2,3] == k[2,3]
    @test (a[2,3] = 2; k[2,3] = 2; a == Array(k))
    @test a[:] == Array(k[:])
    @test (a[:] = 3; k[:] = 3; a == Array(k))
    @test a[:,:] == Array(k[:,:])
    @test (a[:,:] = 4; k[:,:] = 4; a == Array(k))
    a = rand(3,4); k = KnetArray(a)
    @test a[3:5] == Array(k[3:5])
    @test (a[3:5] = 1; k[3:5] = 1; a == Array(k))
    @test a[1:2,3:4] == Array(k[1:2,3:4])
    @test (a[1:2,3:4] = 2; k[1:2,3:4] = 2; a == Array(k))
    @test a[2:end] == Array(k[2:end])
    @test (a[2:end] = 3; k[2:end] = 3; a == Array(k))
    @test a[2:end,2:end] == Array(k[2:end,2:end])
    @test (a[2:end,2:end] = 4; k[2:end,2:end] = 4; a == Array(k))
    a = rand(3,4); k = KnetArray(a)

    # Unsupported indexing:
    # @test_broken a[3:2:9] == Array(k[3:2:9]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::StepRange{Int64,Int64})
    # @test_broken a[1:2:3,1:3:4] == Array(k[1:2:3,1:3:4]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::StepRange{Int64,Int64}, ::StepRange{Int64,Int64})
    # @test_broken a[[1,5,3]] == Array(k[[1,5,3]]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::Array{Int64,1})
    # @test_broken a[[3,1],[4,2]] == Array(k[[3,1],[4,2]]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::Array{Int64,1}, ::Array{Int64,1})
    # @test_broken a[[]] == Array(k[[]]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::Array{Any,1})
    # @test_broken a[a.>0.5] == Array(k[k.>0.5]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::Knet.KnetArray{Float64,2})
    # @test_broken a[a.>0.5] == Array(k[a.>0.5]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::BitArray{2})

    # AbstractArray interface
    # @test_broken cat((1,2),a,a) == Array(cat((1,2),k,k)) # cat only impl for i=1,2
    @test cat(1,a,a) == Array(cat(1,k,k)) 
    @test cat(2,a,a) == Array(cat(2,k,k))
    @test convert(Array{Float32},a) == Array(convert(KnetArray{Float32},k))
    @test copy(a) == Array(copy(k))
    @test eachindex(a) == eachindex(k)
    @test Base.elsize(a) == Base.elsize(k)
    @test eltype(a) == eltype(k)
    @test endof(a) == endof(k)
    @test fill!(similar(a),pi) == Array(fill!(similar(k),pi))
    @test fill!(similar(a,(2,6)),pi) == Array(fill!(similar(k,(2,6)),pi))
    @test fill!(similar(a,2,6),pi) == Array(fill!(similar(k,2,6),pi))
    @test first(a) == first(k)
    @test hcat(a,a) == Array(hcat(k,k))
    @test isa(pointer(k), Ptr{Float64})
    @test isa(pointer(k,3), Ptr{Float64})
    @test isempty(a) == isempty(k)
    @test isempty(KnetArray(Float32,0))
    @test length(a) == length(k)
    @test Base.linearindexing(a) == Base.linearindexing(k)
    @test ndims(a) == ndims(k)
    @test ones(a) == Array(ones(k))
    @test rand!(copy(a)) != Array(rand!(copy(k)))
    @test reshape(a,(2,6)) == Array(reshape(k,(2,6)))
    @test reshape(a,2,6) == Array(reshape(k,2,6))
    @test size(a) == size(k)
    @test size(a,1) == size(k,1)
    @test size(a,2) == size(k,2)
    @test stride(a,1) == stride(k,1)
    @test stride(a,2) == stride(k,2)
    @test strides(a) == strides(k)
    @test vcat(a,a) == Array(vcat(k,k))
    @test vec(a) == Array(vec(k))
    @test zeros(a) == Array(zeros(k))

    # cpu/gpu xfer with grad support
    @test gradcheck(x->gpu2cpu(sin(cpu2gpu(x))),a)
    @test gradcheck(x->cpu2gpu(sin(gpu2cpu(x))),k)

end
