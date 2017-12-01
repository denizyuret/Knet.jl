include("header.jl")

# http://docs.julialang.org/en/latest/manual/arrays.html#man-supported-index-types-1
if VERSION < v"0.5.0"
    Base.IteratorsMD.CartesianIndex(i::Int...)=CartesianIndex(i)
end

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
        # check out http://docs.julialang.org/en/latest/manual/arrays.html#Cartesian-indices-1
        @testset "indexing" begin
            @test a == k                     		# Supported index types:
            for i in ((:,), (:,:),                      # Colon, Tuple{Colon}
                      (3,), (2,3),              	# Int, Tuple{Int}
                      (3:5,), (1:2,3:4),                # UnitRange, Tuple{UnitRange}
                      (2,:), (:,2),                     # Int, Colon
                      (1:2,:), (:,1:2),                 # UnitRange,Colon
                      (1:2,2), (2,1:2),                 # Int, UnitRange
                      (1:2:3,),                         # StepRange
                      (1:2:3,:), (:,1:2:3),             # StepRange,Colon
                      ([1,3],), ([2,2],),               # Vector{Int}
                      ([1,3],:), (:,[1,3]),             # Vector{Int},Colon
                      ([2,2],:), (:,[2,2]),             # Repeated index
                      ([],),                            # Empty Array
                      ((a.>0.5),),                      # BitArray
                      ([1 3; 2 4],),                    # Array{Int}
                      (CartesianIndex(3,),), (CartesianIndex(2,3),), # CartesianIndex
                      (if VERSION >= v"0.5.0"
                           [(:,a[1,:].>0.5),(a[:,1].>0.5,:),  # BitArray2 # FAIL for julia4
                            ([CartesianIndex(2,2), CartesianIndex(2,1)],)] # Array{CartesianIndex} # FAIL for julia4
                       else [] end)...
                      )
                # @show i
                @test a[i...] == k[i...]
                ai = a[i...]
                a[i...] = 0
                k[i...] = 0
                @test a == k
                a[i...] = ai
                k[i...] = ai
                @test a == k
                @test gradcheck(getindex, a, i...)
                @test gradcheck(getindex, k, i...)
            end
            # make sure end works
            @test a[2:end] == k[2:end]
            @test a[2:end,2:end] == k[2:end,2:end]
            # k.>0.5 returns KnetArray{T}, no Knet BitArrays yet
            @test a[a.>0.5] == k[k.>0.5]
        end

        # Unsupported indexing etc.:
        # @test_broken a[1:2:3,1:3:4] == Array(k[1:2:3,1:3:4]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::StepRange{Int64,Int64}, ::StepRange{Int64,Int64})
        # @test_broken a[[3,1],[4,2]] == Array(k[[3,1],[4,2]]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::Array{Int64,1}, ::Array{Int64,1})
        # @test_broken cat((1,2),a,a) == Array(cat((1,2),k,k)) # cat only impl for i=1,2

        # AbstractArray interface
        @testset "abstractarray" begin

            for f in (copy, endof, first, isempty, length, ndims, ones, vec, zeros, 
                      a->(eachindex(a);0), a->(eltype(a);0), # a->(Base.linearindexing(a);0),
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
            if VERSION >= v"0.6.0"
                @test gradcheck(x->Array(sin.(KnetArray(x))),a)
                @test gradcheck(x->KnetArray(sin.(Array(x))),k)
            else
                @test gradcheck(x->Array(sin(KnetArray(x))),a)
                @test gradcheck(x->KnetArray(sin(Array(x))),k)
            end
        end

        @testset "reshape" begin
            a = KnetArray(Float32, 2, 2, 2)
            
            @test size(reshape(a, 4, :)) == size(reshape(a, (4, :)) == (4, 2)
            @test size(reshape(a, :, 4)) == size(reshape(a, (:, 4)) == (2, 4)
            @test size(reshape(a, :, 1, 4)) == (2, 1,  4)
        end
    end
end

nothing
