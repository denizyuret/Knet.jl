using Knet
using CUDArt
using CUSPARSE
using Base.Test

a = rand(Float32,5,3)
b = convert(SparseMatrixCSC{Float32}, sprand(1000,3,.005))
b[10,:] = 1
c = a*b'
sc = sparse(c)
da = CudaArray(a)
db = CudaSparseMatrixCSC(b)
cc = similar(sc)
rand!(cc.nzval)
dc1 = CudaSparseMatrixCSR(cc)
Knet.A_mul_Bt_slow!(dc1, da, db)
sc1 = Knet.to_host2(dc1)
c1 = full(sc1)
@test @show c == c1
dc2 = CudaSparseMatrixCSR(cc)
Knet.A_mul_Bt!(dc2, da, db)
sc2 = Knet.to_host2(dc2)
c2 = full(sc2)
@test @show c == c2

T=1000
I=1000
J=20
K=10000


foo1(c,a,b)=(for t=1:T; Knet.A_mul_Bt_slow!(c,a,b); end)
foo2(c,a,b)=(for t=1:T; Knet.A_mul_Bt!(c,a,b); end)

a = rand(Float32, I, J)
b = convert(SparseMatrixCSC{Float32}, sprand(K,J,1/K))
c = a*b'
sc = sparse(c)
da = CudaArray(a)
db = CudaSparseMatrixCSC(b)
cc = similar(sc)
rand!(cc.nzval)
dc1 = CudaSparseMatrixCSR(cc)
dc2 = CudaSparseMatrixCSR(cc)
@date foo1(dc1, da, db)
@date foo1(dc1, da, db)
@date foo2(dc2, da, db)
@date foo2(dc2, da, db)

sc1 = Knet.to_host2(dc1)
c1 = full(sc1)
@test @show c == c1

sc2 = Knet.to_host2(dc2)
c2 = full(sc2)
@test @show c == c2
