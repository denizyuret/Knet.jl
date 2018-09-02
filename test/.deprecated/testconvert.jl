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

info("KUsparse")
dd=convert(KUsparse,aa)
ee=convert(SparseMatrixCSC,dd)
@test aa==ee

info("KUsparse + GPU")
mm=gpucopy(dd)
nn=cpucopy(mm)
oo=convert(SparseMatrixCSC,nn)
@test oo==aa

# info("No GPU only conversion")

