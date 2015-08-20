using KUnet, CUDArt, Base.Test
include("isapprox.jl")

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

info("Array")
ab = a1 * b1
atb = a1' * b1
abt = a1 * b1'

info("SparseMatrixCSC")
@test isapprox(ab, full(a0 * b0))
@test isapprox(atb, full(a0' * b0))
@test isapprox(abt, full(a0 * b0'))

info("CudaArray")
a2 = gpucopy(a1)
b2 = gpucopy(b1)
c2 = gpucopy(c1)
@test isapprox(ab, A_mul_B!(c2, a2, b2))
@test isapprox(atb, At_mul_B!(c2, a2, b2))
@test isapprox(abt, A_mul_Bt!(c2, a2, b2))

info("KUdense{Array}")
a3 = KUdense(copy(a1))
b3 = KUdense(copy(b1))
c3 = KUdense(copy(c1))
a3p = copy(a1)
c3r = copy(c1)
@test isapprox(ab, A_mul_B!(c3, a3p, b3))
@test isapprox(atb, At_mul_B!(c3, a3p, b3))
@test isapprox(abt, A_mul_Bt!(c3r, a3, b3))

info("KUdense{CudaArray}")
a4 = KUdense(copy(a2))
b4 = KUdense(copy(b2))
c4 = KUdense(copy(c2))
a4p = copy(a2)
c4r = copy(c2)
@test isapprox(ab, A_mul_B!(c4, a4p, b4))
@test isapprox(atb, At_mul_B!(c4, a4p, b4))
@test isapprox(abt, A_mul_Bt!(c4r, a4, b4))

info("Sparse{Array} Sparse{Array}")
a5 = convert(Sparse, copy(a0))
b5 = convert(Sparse, copy(b0))
c5 = copy(c1)
# @test isapprox(ab,   A_mul_B!(c5, a5, b5))
@test isapprox(atb, At_mul_B!(c5, a5, b5))
# @test isapprox(abt, A_mul_Bt!(c5, a5, b5))

info("Array Sparse{Array}")
a6 = copy(a1)
b6 = convert(Sparse, copy(b0))
c6 = copy(c1)
@test isapprox(ab,   A_mul_B!(c6, a6, b6))
# @test isapprox(atb, At_mul_B!(c6, a6, b6))
@test isapprox(abt, A_mul_Bt!(c6, a6, b6))

info("Sparse{CudaArray} Sparse{CudaArray}")
a7 = convert(Sparse{CudaArray}, copy(a0))
b7 = convert(Sparse{CudaArray}, copy(b0))
c7 = convert(CudaArray, copy(c1))
# @test isapprox(ab,   A_mul_B!(c7, a7, b7))
@test isapprox(atb, At_mul_B!(c7, a7, b7))
# @test isapprox(abt, A_mul_Bt!(c7, a7, b7))

info("CudaArray Sparse{CudaArray}")
a8 = convert(CudaArray, copy(a1))
b8 = convert(Sparse{CudaArray}, copy(b0))
c8 = convert(CudaArray, copy(c1))
@test isapprox(ab,   A_mul_B!(c8, a8, b8))
# @test isapprox(atb, At_mul_B!(c8, a8, b8))
@test isapprox(abt, A_mul_Bt!(c8, a8, b8))

info("KUsparse{Array} KUsparse{Array}")
a15 = convert(KUsparse, copy(a0))
b15 = convert(KUsparse, copy(b0))
c15 = convert(KUdense, copy(c1))
# @test isapprox(ab,   A_mul_B!(c15, a15, b15))
@test isapprox(atb, At_mul_B!(c15, a15, b15))
# @test isapprox(abt, A_mul_Bt!(c15, a15, b15))

info("KUdense{Array} KUsparse{Array}")
a16 = convert(KUdense, copy(a1))
a16p = copy(a1)
b16 = convert(KUsparse, copy(b0))
c16 = convert(KUdense, copy(c1))
c16a = copy(c1)
@test isapprox(ab,   A_mul_B!(c16, a16p, b16))
# @test isapprox(atb, At_mul_B!(c16, a16, b16))
@test isapprox(abt, A_mul_Bt!(c16a, a16, b16))

info("KUsparse{CudaArray} KUsparse{CudaArray}")
a17 = convert(KUsparse{CudaArray}, copy(a0))
b17 = convert(KUsparse{CudaArray}, copy(b0))
c17 = convert(KUdense{CudaArray}, copy(c1))
# @test isapprox(ab,   A_mul_B!(c17, a17, b17))
@test isapprox(atb, At_mul_B!(c17, a17, b17))
# @test isapprox(abt, A_mul_Bt!(c17, a17, b17))

info("KUdense{CudaArray} KUsparse{CudaArray}")
a18 = convert(KUdense{CudaArray}, copy(a1))
a18p = copy(a2)
b18 = convert(KUsparse{CudaArray}, copy(b0))
c18 = convert(KUdense{CudaArray}, copy(c1))
c18a = copy(c2)
@test isapprox(ab,   A_mul_B!(c18, a18p, b18))
# @test isapprox(atb, At_mul_B!(c18, a18, b18))
@test isapprox(abt, A_mul_Bt!(c18a, a18, b18))

