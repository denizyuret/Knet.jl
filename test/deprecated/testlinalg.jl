using Knet, CUDArt, CUSPARSE, Base.Test
include("isapprox.jl")

# function A_mul_B!{T}(C::KUdense{CudaArray,T,2}, A::CudaArray{T,2}, B::CudaSparseMatrixCSC{T})
a = rand(3,5)
b = sprand(5,2,.5)
c = a*b
A = CudaArray(a)
B = CudaSparseMatrixCSC(b)
C = KUdense(CudaArray(similar(c)))
A_mul_B!(C,A,B)
@test @show c == convert(Array,C)
error("ok")

# Operations needed:
# mmul forw: A_mul_B!(y, w, x)		A_mul_Bs!(y, w, x): cpu/gpu
# mmul back: A_mul_Bt!(dw, dy, x)	A_mul_Bst!(dw, dy, x): cpu/gpu
# mmul back: At_mul_B!(dx, w, dy)	no dx: only initial input can be sparse
# kper forw: At_mul_B!(k, s, x)		Ast_mul_Bs!(k, s, x): cpu/gpu

sprand32(m,n,r)=convert(SparseMatrixCSC{Float64,Int32},sprand(m,n,r))

a0 = sprand32(100,100,.1)
b0 = sprand32(100,100,.1)
c0 = sprand32(100,100,.1)

a1 = full(a0)
b1 = full(b0)
c1 = full(c0)

info("Array = Array * Array")
ab = a1 * b1
atb = a1' * b1
abt = a1 * b1'

info("SparseMatrixCSC = SparseMatrixCSC * SparseMatrixCSC")
@test isapprox(ab, full(a0 * b0))
@test isapprox(atb, full(a0' * b0))
@test isapprox(abt, full(a0 * b0'))

info("CudaArray = CudaArray * CudaArray")
a2 = gpucopy(a1)
b2 = gpucopy(b1)
c2 = gpucopy(c1)
@test isapprox(ab, A_mul_B!(c2, a2, b2))
@test isapprox(atb, At_mul_B!(c2, a2, b2))
@test isapprox(abt, A_mul_Bt!(c2, a2, b2))

info("KUdense{Array} = KUdense{Array} * KUdense{Array}")
a3 = KUdense(copy(a1))
b3 = KUdense(copy(b1))
c3 = KUdense(copy(c1))
@test isapprox(ab, A_mul_B!(c3, a3, b3))
@test isapprox(atb, At_mul_B!(c3, a3, b3))
@test isapprox(abt, A_mul_Bt!(c3, a3, b3))

info("KUdense{Array} = Array * KUdense{Array}")
a3p = copy(a1)
@test isapprox(ab, A_mul_B!(c3, a3p, b3))
@test isapprox(atb, At_mul_B!(c3, a3p, b3))

info("Array = KUdense{Array} * KUdense{Array}")
c3r = copy(c1)
@test isapprox(abt, A_mul_Bt!(c3r, a3, b3))

info("KUdense{CudaArray} = KUdense{CudaArray} * KUdense{CudaArray}")
a4 = KUdense(copy(a2))
b4 = KUdense(copy(b2))
c4 = KUdense(copy(c2))
@test isapprox(ab, A_mul_B!(c4, a4, b4))
@test isapprox(atb, At_mul_B!(c4, a4, b4))
@test isapprox(abt, A_mul_Bt!(c4, a4, b4))

info("KUdense{CudaArray} = CudaArray * KUdense{CudaArray}")
a4p = copy(a2)
@test isapprox(ab, A_mul_B!(c4, a4p, b4))
@test isapprox(atb, At_mul_B!(c4, a4p, b4))

info("CudaArray = KUdense{CudaArray} * KUdense{CudaArray}")
c4r = copy(c2)
@test isapprox(abt, A_mul_Bt!(c4r, a4, b4))

info("KUdense{Array} = KUsparse{Array} * KUsparse{Array}")
a5 = convert(KUsparse, copy(a0))
b5 = convert(KUsparse, copy(b0))
c5 = convert(KUdense, copy(c1))
@test isapprox(atb, At_mul_B!(c5, a5, b5))

info("Array = KUsparse{Array} * KUsparse{Array}")
c5r = copy(c1)
@test isapprox(atb, At_mul_B!(c5r, a5, b5))

info("Array = KUdense{Array} * KUsparse{Array}")
a16 = convert(KUdense, copy(a1))
b16 = convert(KUsparse, copy(b0))
c16a = copy(c1)
@test isapprox(abt, A_mul_Bt!(c16a, a16, b16))

info("KUdense{Array} = Array * KUsparse{Array}")
a16p = copy(a1)
c16 = convert(KUdense, copy(c1))
@test isapprox(ab, A_mul_B!(c16, a16p, b16))

info("Array = Array * KUsparse{Array}")
a6p = copy(a1)
b6 = convert(KUsparse, copy(b0))
c6r = copy(c1)
@test isapprox(ab,   A_mul_B!(c6r, a6p, b6))
@test isapprox(abt, A_mul_Bt!(c6r, a6p, b6))

info("KUdense{CudaArray} = KUsparse{CudaArray} * KUsparse{CudaArray}")
a7 = convert(KUsparse{CudaArray}, copy(a0))
b7 = convert(KUsparse{CudaArray}, copy(b0))
c7 = convert(KUdense{CudaArray}, copy(c1))
@test isapprox(atb, At_mul_B!(c7, a7, b7))

info("CudaArray = KUsparse{CudaArray} * KUsparse{CudaArray}")
c7r = convert(CudaArray, copy(c1))
@test isapprox(atb, At_mul_B!(c7r, a7, b7))

info("CudaArray = KUdense{CudaArray} * KUsparse{CudaArray}")
a18 = convert(KUdense{CudaArray}, copy(a1))
b18 = convert(KUsparse{CudaArray}, copy(b0))
c18a = copy(c2)
@test isapprox(abt, A_mul_Bt!(c18a, a18, b18))

info("KUdense{CudaArray} = CudaArray * KUsparse{CudaArray}")
a18p = copy(a2)
c18 = convert(KUdense{CudaArray}, copy(c1))
@test isapprox(ab,   A_mul_B!(c18, a18p, b18))

info("CudaArray = CudaArray * KUsparse{CudaArray}")
a8p = convert(CudaArray, copy(a1))
b8 = convert(KUsparse{CudaArray}, copy(b0))
c8r = convert(CudaArray, copy(c1))
@test isapprox(ab,   A_mul_B!(c8r, a8p, b8))
@test isapprox(abt, A_mul_Bt!(c8r, a8p, b8))

