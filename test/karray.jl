include("header.jl")

# http://docs.julialang.org/en/latest/manual/arrays.html#man-supported-index-types-1

# Test KnetArray operations: cat, convert, copy, display, eachindex,
# eltype, endof, fill!, first, getindex, hcat, isempty, length,
# linearindexing, ndims, ones, pointer, rand!, reshape, setindex!,
# similar, size, stride, strides, summary, vcat, vec, zeros

if gpu() >= 0
    @testset "karray" begin
        a2 = rand(3,4)
        k2 = KnetArray(a2)
        a3 = rand(3,4,5) 
        k3 = KnetArray(a3)
        
        # getindex, setindex!
        # Index types: Integer, CartesianIndex, Vector{Int}, Array{Int}, EmptyArray, a:c, a:b:c, Colon, Bool
        # See http://docs.julialang.org/en/latest/manual/arrays.html#man-supported-index-types-1
        # check out http://docs.julialang.org/en/latest/manual/arrays.html#Cartesian-indices-1
        @testset "indexing 2d" begin
            @test a2 == k2                     		# Supported index types:
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
                      ((a2.>0.5),),                      # BitArray
                      ([1 3; 2 4],),                    # Array{Int}
                      (CartesianIndex(3,),),            # CartesianIndex
                      (CartesianIndex(2,3),),           
                      (:,a2[1,:].>0.5),                  # BitArray2 
                      (a2[:,1].>0.5,:),  
                      ([CartesianIndex(2,2), CartesianIndex(2,1)],) # Array{CartesianIndex} 
                      )
                # @show i
                @test a2[i...] == k2[i...]
                ai = a2[i...]
                a2[i...] = 0
                k2[i...] = 0
                @test a2 == k2
                a2[i...] = ai
                k2[i...] = ai
                @test a2 == k2
                @test gradcheck(getindex, a2, i...)
                @test gradcheck(getindex, k2, i...)
            end
            # make sure end works
            @test a2[2:end] == k2[2:end]
            @test a2[2:end,2:end] == k2[2:end,2:end]
            # k2.>0.5 returns KnetArray{T}, no Knet BitArrays yet
            @test a2[a2.>0.5] == k2[k2.>0.5]

             # Unsupported indexing etc.:
            # @test_broken a2[1:2:3,1:3:4] == Array(k2[1:2:3,1:3:4]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::StepRange{Int64,Int64}, ::StepRange{Int64,Int64})
            # @test_broken a2[[3,1],[4,2]] == Array(k2[[3,1],[4,2]]) # MethodError: no method matching getindex(::Knet.KnetArray{Float64,2}, ::Array{Int64,1}, ::Array{Int64,1})
            # @test_broken cat((1,2),a2,a2) == Array(cat((1,2),k2,k2)) # cat only impl for i=1,2
        end

       
        @testset "indexing 3d" begin
            @test a3 == k3                     		# Supported index types:
            for i in (
                    (:,), 
                    (:,:,:),                      # Colon, Tuple{Colon}
                    (3,), (2,3,2),              	# Int, Tuple{Int}
                    (3:5,), 
                    (1:2,3:4,3),                # UnitRange, Tuple{UnitRange}
                    (2,:,:), (1,:,2),                     # Int, Colon
                    (1:2,:,1), (1:3,:,1:2),(:,1:2,2),   # UnitRange,Colon
                    (1:2,2,2), (2,2:2,1:2),                 # Int, UnitRange
                    # (1:2:3,),                         # StepRange
                    # (1:2:3,:), (:,1:2:3),             # StepRange,Colon
                    # ([1,3],), ([2,2],1,1),               # Vector{Int}
                    # ([1,3],:), (:,[1,3]),             # Vector{Int},Colon
                    # ([2,2],:), (:,[2,2]),             # Repeated index
                    # ([],),                            # Empty Array
                    ((a3.>0.5),),                      # BitArray
                    # ([1 3; 2 4],),                    # Array{Int}
                    (CartesianIndex(3,),),            # CartesianIndex
                    (CartesianIndex(2,3,4),),           
                    # (:,a3[1,:].>0.5),                  # BitArray2 
                    # (a3[:,1].>0.5,:),  
                    # ([CartesianIndex(2,2), CartesianIndex(2,1)],) # Array{CartesianIndex} 
                    )
                # @show i
                @test a3[i...] == k3[i...]
                ai = a3[i...]
                a3[i...] = 0
                k3[i...] = 0
                @test a3 == k3
                a3[i...] = ai
                k3[i...] = ai
                @test a3 == k3
                @test gradcheck(getindex, a3, i...)
                @test gradcheck(getindex, k3, i...)
            end
            # make sure end works
            @test a3[2:end] == k3[2:end]
            @test a3[2:end,2:end,2:end] == k3[2:end,2:end,2:end]
            # k2.>0.5 returns KnetArray{T}, no Knet BitArrays yet
            @test a3[a3.>0.5] == k3[k3.>0.5]
        end
        # AbstractArray interface
        @testset "abstractarray" begin

            for f in (copy, endof, first, isempty, length, ndims, ones, vec, zeros, 
                      a2->(eachindex(a2);0), a2->(eltype(a2);0), # a2->(Base.linearindexing(a2);0),
                      a2->collect(Float64,size(a2)), a2->collect(Float64,strides(a2)), 
                      a2->cat(1,a2,a2), a2->cat(2,a2,a2), a2->hcat(a2,a2), a2->vcat(a2,a2), 
                      a2->reshape(a2,2,6), a2->reshape(a2,(2,6)), 
                      a2->size(a2,1), a2->size(a2,2),
                      a2->stride(a2,1), a2->stride(a2,2), )

                # @show f
                @test f(a2) == f(k2)
                @test gradcheck(f, a2)
                @test gradcheck(f, k2)
            end

            @test convert(Array{Float32},a2) == convert(KnetArray{Float32},k2)
            @test fill!(similar(a2),pi) == fill!(similar(k2),pi)
            @test fill!(similar(a2,(2,6)),pi) == fill!(similar(k2,(2,6)),pi)
            @test fill!(similar(a2,2,6),pi) == fill!(similar(k2,2,6),pi)
            @test isa(pointer(k2), Ptr{Float64})
            @test isa(pointer(k2,3), Ptr{Float64})
            @test isempty(KnetArray(Float32,0))
            @test rand!(copy(a2)) != rand!(copy(k2))
            @test k2 == k2
            @test a2 == k2
            @test k2 == a2
            @test isapprox(k2,k2)
            @test isapprox(a2,k2)
            @test isapprox(k2,a2)
            @test a2 == copy!(similar(a2),k2)
            @test k2 == copy!(similar(k2),a2)
            @test k2 == copy!(similar(k2),k2)
            @test k2 == copy(k2)
            @test pointer(k2) != pointer(copy(k2))
            @test k2 == deepcopy(k2)
            @test pointer(k2) != pointer(deepcopy(k2))
        end

        @testset "cpu2gpu" begin
            # cpu/gpu xfer with grad support
            @test gradcheck(x->Array(sin.(KnetArray(x))),a2)
            @test gradcheck(x->KnetArray(sin.(Array(x))),k2)
        end

        @testset "reshape" begin
            a2 = KnetArray(Float32, 2, 2, 2)
            
            @test size(reshape(a2, 4, :)) == size(reshape(a2, (4, :))) == (4, 2)
            @test size(reshape(a2, :, 4)) == size(reshape(a2, (:, 4))) == (2, 4)
            @test size(reshape(a2, :, 1, 4)) == (2, 1,  4)
        end        
    end
end

nothing
