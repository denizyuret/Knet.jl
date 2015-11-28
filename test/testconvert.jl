using Knet, CUDArt, Base.Test

a=rand(3,5)

info("KUdense")
b=convert(KUdense,a)
c=convert(Array,b)
@test a==c

info("GPU")
d=gpucopy(a)
e=cpucopy(d)
@test a==e

info("KUdense + GPU")
f=gpucopy(b)
g=cpucopy(f)
h=convert(Array,g)
@test a==h

info("GPU + KUdense")
i=convert(KUdense,d)
j=cpucopy(i)
k=convert(Array,j)
@test a==k

aa=sprand(3,5,.5)

info("Sparse")
bb=convert(Sparse,aa)
cc=convert(SparseMatrixCSC,bb)
@test aa==cc

info("KUsparse")
dd=convert(KUsparse,aa)
ee=convert(SparseMatrixCSC,dd)
@test aa==ee

info("Sparse + KUsparse")
ff=convert(KUsparse, bb)
gg=convert(SparseMatrixCSC,ff)
@test aa==gg

info("KUsparse + Sparse")
hh=convert(Sparse, dd)
ii=convert(SparseMatrixCSC,hh)
@test ii==aa

info("Sparse + GPU")
jj=gpucopy(bb)
kk=cpucopy(jj)
ll=convert(SparseMatrixCSC,kk)
@test ll==aa

info("KUsparse + GPU")
mm=gpucopy(dd)
nn=cpucopy(mm)
oo=convert(SparseMatrixCSC,nn)
@test oo==aa

info("Sparse + KUsparse + GPU")
pp=gpucopy(ff)
qq=cpucopy(pp)
rr=convert(SparseMatrixCSC,qq)
@test rr==aa

info("KUsparse + Sparse + GPU")
ss=gpucopy(hh)
tt=cpucopy(ss)
uu=convert(SparseMatrixCSC,tt)
@test uu==aa

# info("No GPU only conversion")

