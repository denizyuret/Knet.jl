using KUnet, CUDArt, Base.Test
CUDArt.to_host(x)=x
a = sprand(3,5,.5)
x = sparse(CudaArray(full(a)))
x2 = gpucopy(a)
for n in names(a)
    println((n, a.(n), to_host(x.(n)), to_host(x2.(n))))
end

b = transpose(a)
y = transpose(x)
for n in names(b)
    println((n, b.(n), to_host(y.(n))))
end

# a = sprand(3,1000,.1)
# b = transpose(a)
# ga = gpucopy(a)
# gb = transpose(ga)
# @test b == cpucopy(gb)
