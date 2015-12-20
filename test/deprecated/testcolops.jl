using Knet, CUDArt, Base.Test

iseq02(a,b)=(convert(Array,a)==convert(Array,b))

for B in (
          SubArray,
          SparseMatrixCSC,
          KUsparse{Array},
          KUsparse{CudaArray},
          Array, 
          CudaArray,
          KUdense{Array},
          KUdense{CudaArray},
          )
    b0 = sprand(10,100,.1)
    b = (B==SubArray ? 
         sub(full(b0),:,:) :
         convert(B, copy(b0)))
    for A in (
              CudaArray,
              Array,
              )
        a0 = sprand(10,20,.5)
        a = issparse(b) ? convert(KUsparse{A}, copy(a0)) : convert(KUdense{A}, copy(a0))
        println(map(typeof,(a,b)))

        # CSLICE
        @test iseq02(cslice!(copy(a), b, 3:8), b0[:,3:8])

        # CCAT
        r = rand(1:ccount(b), 10)
        @test iseq02(ccat!(copy(a), b, r), hcat(a0, b0[:,r]))

        # CCOPY
        if !issparse(b)
            bb = ccopy!(copy(b), 5, a) # copies a into b starting at b column 5
            @test iseq02(bb, hcat(b0[:,1:4], a0, b0[:,25:end]))
            
            # CADD
            if atype(a) == atype(b)
                cc = cadd!(copy(b), 5, a)  # adds a into b starting at b column 5
                @test iseq02(cc, hcat(b0[:,1:4], (a0+b0[:,5:24]), b0[:,25:end]))
            end
        end

        # UNIQ
        aa = copy(a)
        for i=1:4; ccat!(aa, a); end
        w0 = convert(KUdense, rand(size(aa)))
        w1 = copy(w0)
        uniq!(aa, w1)
        @test iseq02(aa, a)
        @test size(aa)==size(w1)
        v0 = convert(Array, w0)
        v1 = convert(Array, w1)
        for j=1:ccount(v1)
            vj = v0[:,j]
            for i=1:4; vj += v0[:,j+i*ccount(v1)]; end
            @test iseq02(vj, v1[:,j])
        end
    end
end
