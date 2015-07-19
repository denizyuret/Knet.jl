using KUnet, CUDArt, Base.Test

a=rand(3,5)

# KUdense
b=convert(KUdense,a)
c=convert(Array,b)
@test a==c

# GPU
d=gpucopy(a)
e=cpucopy(d)
@test a==e

# KUdense + GPU
f=gpucopy(b)
g=cpucopy(f)
h=convert(Array,g)
@test a==h

# GPU + KUdense
i=convert(KUdense,d)
j=cpucopy(i)
k=convert(Array,j)
@test a==k

aa=sprand(3,5,.5)

# Sparse
bb=convert(Sparse,aa)
cc=convert(SparseMatrixCSC,bb)
@test aa==cc

# KUsparse
dd=convert(KUsparse,aa)
ee=convert(SparseMatrixCSC,dd)
@test aa==ee

# Sparse + KUsparse
ff=convert(KUsparse, bb)
gg=convert(SparseMatrixCSC,ff)
@test aa==gg

# KUsparse + Sparse
hh=convert(Sparse, dd)
ii=convert(SparseMatrixCSC,hh)
@test ii==aa

# Sparse + GPU
jj=gpucopy(bb)
kk=cpucopy(jj)
ll=convert(SparseMatrixCSC,kk)
@test ll==aa

# KUsparse + GPU
mm=gpucopy(dd)
nn=cpucopy(mm)
oo=convert(SparseMatrixCSC,nn)
@test oo==aa

# Sparse + KUsparse + GPU
pp=gpucopy(ff)
qq=cpucopy(pp)
rr=convert(SparseMatrixCSC,qq)
@test rr==aa

# KUsparse + Sparse + GPU
ss=gpucopy(hh)
tt=cpucopy(ss)
uu=convert(SparseMatrixCSC,tt)
@test uu==aa

# No GPU only conversion

