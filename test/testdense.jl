using Base.Test
using CUDArt
using KUnet

for N in 1:5
    for A in (CudaArray, Array)
        for T in (Float64, Float32, Int64, Int32)
            S = tuple(rand(1:20,N)...)
            @show (A,T,S)
            a = convert(A{T}, rand(T,S))
            b = convert(A{T}, rand(T,csize(a,rand(1:20))))
            d = KUdense(a)
            @test atype(d)==A
            @test d.arr == a
            @test pointer(d) == pointer(d.arr) == pointer(d.ptr)
            for fname in (:eltype, :length, :ndims, :size, :strides, :isempty)
                @test (@eval $fname($d)==$fname($a))
            end
            for fname in (:size, :stride)
                for n in 1:N
                    @test (@eval $fname($d,$n)==$fname($a,$n))
                end
            end

            if isa(a,Array)
                r = rand(1:length(d))
                a[r] = d[r] = 42
                @test a[r] == d[r] == 42
                rr = map(x->rand(1:x), size(d))
                a[rr...] = d[rr...] = 53
                @test a[rr...] == d[rr...] == 53
            end

            dd = copy(d)
            copy!(dd, b)
            @test dd.arr == b
            copy!(dd, d)
            @test dd.arr == d.arr
            @test pointer(dd) == pointer(dd.arr) == pointer(dd.ptr)
            @test pointer(dd) != pointer(d)
            
            q = csize(d, rand(1:ccount(d)))
            qq = map(x->(1:x), q)
            resize!(d, q)
            @test to_host(d.arr) == to_host(a)[qq...]
            
            copy!(d, a)
            setseed(42); rand!(a)
            setseed(42); rand!(d)
            @test d.arr == a
            if isa(eltype(a), FloatingPoint)
                setseed(42); randn!(a)
                setseed(42); randn!(d)
                @test d.arr == a
            end

            cd = cpucopy(d)
            @test pointer(cd) == pointer(cd.arr) == pointer(cd.ptr)
            @test pointer(cd) != pointer(d)
            gd = gpucopy(d)
            @test pointer(gd) == pointer(gd.arr) == pointer(gd.ptr)
            @test pointer(gd) != pointer(d)
            ca = cpucopy(a)
            @test cd.arr == ca
            ga = gpucopy(a)
            @test gd.arr == ga
        end
    end
end