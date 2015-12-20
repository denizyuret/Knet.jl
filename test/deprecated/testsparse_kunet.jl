using Compat
using Base.Test
using CUDArt
using KUnet
density = 0.4

iseq01(a::KUsparse,b::SparseMatrixCSC)=(for f in fieldnames(a); iseq01(a.(f),b.(f)) || (warn("$f mismatch:\n$(a.(f))\n$(b.(f))"); return false); end; return true)
iseq01(a::BaseArray,b::BaseArray)=(to_host(a)==to_host(b))
iseq01(a,b)=(a==b)

for A in (CudaArray, Array)
    for T in (Float64, Float32, Int64, Int32)
        m = rand(1:20)
        n = rand(1:20)
        @show (A,T,m,n)
        a = sprand(m,n,density,rand,T)
        s = convert(KUsparse{A}, copy(a))

        @test atype(s)==A
        @test iseq01(s,a)
        for fname in (:eltype, :length, :ndims, :size, :isempty)
            @test (@eval $fname($s)==$fname($a))
        end
        @test size(s,1)==size(a,1)
        @test size(s,2)==size(a,2)

        # copy
        s1 = cpucopy(s); @test iseq01(s1,a)
        s2 = gpucopy(s); @test iseq01(s2,a)
        s3 = copy(s); @test iseq01(s3,a)
        # b = sprand(m,rand(1:20),density,rand,T)
        # @show map(size, (a,b,s,s1,s2,s3))
        # copy!(s3,b); @test iseq01(s3,b)
        # copy!(s3,s); @test iseq01(s3,a)

        # cslice!
        r1 = rand(1:ccount(a))
        r2 = rand(1:ccount(a))
        r1 <= r2 || ((r1,r2)=(r2,r1))
        s3 = copy(s)
        cslice!(s3, a, r1:r2)
        @test iseq01(s3, a[:,r1:r2])

        # ccat!
        b = sprand(m,rand(1:20),density,rand,T)
        da = convert(KUsparse{A}, copy(a))
        db = convert(KUsparse{A}, copy(b))
        ccat!(da, db)
        @test iseq01(da, [a b])

        # uniq!
        for i=1:5; ccat!(da, db); end
        uniq!(da)
        @test iseq01(da, unique([a b], 2))
    end
end
