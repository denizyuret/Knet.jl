using CUDArt,CUBLAS,CUSPARSE,KUnet,Base.Test

csc2csr{T}(x::SparseMatrixCSC{T})=CudaSparseMatrixCSR{T}(CudaArray(convert(Vector{Cint},x.colptr)), CudaArray(convert(Vector{Cint},x.rowval)), CudaArray(x.nzval), (x.n,x.m), convert(Cint,length(x.nzval)), device())

function LinAlg.A_mul_B!{T}(Y::StridedMatrix{T}, X::StridedMatrix{T}, A::SparseMatrixCSC{T})
    mX, nX = size(X)
    nX == A.m || throw(DimensionMismatch())
    fill!(Y,0)
    rowval = A.rowval
    nzval = A.nzval
    @inbounds for multivec_row=1:mX, col = 1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        Y[multivec_row, col] += X[multivec_row, rowval[k]] * nzval[k]
    end
    Y
end

# nh = 1000
# nd = 100000
# nb = 100

nh = 1500
nd = 10000
nb = 20

info("1000x CPU forw: 0.7ms")
w = x = y = nothing; gc()
w = rand(Float32,  nh, nd)      # N
x = speye(Float32, nd, nb)      # N
rand!(x.rowval, 1:nd)
y = Array(Float32, nh, nb)      # N
@time for i=1:1000              # 0.5ms
    A_mul_B!(y, w, x)
end
@test @show to_host(y)==to_host(w)[:,x.rowval]

info("1000x GPU forw using KUsparse A_mul_B!(y,w,x) 2.5ms")
w1 = x1 = y1 = nothing; gc()
w1 = CudaArray(w)               # N
x1 = convert(KUsparse{CudaArray,Float32}, x) # N
y1 = fill!(CudaArray(y),0)                   # N
@time for i=1:1000
    A_mul_B!(y1, w1, x1)        # 2.5ms
    gpusync()
end
@test @show to_host(y1)==y

info("10000x GPU forw using csrmm!(x',w',y') 0.12ms")
w2 = x2 = y2 = nothing; gc()
x2 = CudaSparseMatrixCSR(x')    # T
w2 = CudaArray(w')              # T
# CUBLAS.geam!('T','T',1f0,w1,0f0,w1,w2)
y2 = fill!(CudaArray(y'),0)     # T
@time for i=1:10000              # 0.08ms
    CUSPARSE.csrmm!('N',1f0,x2,w2,0f0,y2,'O')
    gpusync()
end
@test @show to_host(y2)==y'

info("10000x GPU forw using csrmm2!(x',w,y') 0.04ms")
w3 = x3 = y3 = nothing; gc()
x3 = x2                         # T
w3 = w1                         # N
y3 = similar(y2)                # T
@time for i=1:10000              # 0.04ms
    CUSPARSE.csrmm2!('N','T',1f0,x3,w3,0f0,y3,'O')
    gpusync()
end
@test @show to_host(y3)==y'

info("10000x GPU forw using csrmm2!(x',w,y')+transpose 0.05ms")
w4 = x4 = y4 = nothing; gc()
x4 = csc2csr(x)                 # T: direct copy possible
w4 = w1                         # N
y4 = similar(y1)                # N
z4 = similar(y2)                # T
@time for i=1:10000              # 0.05ms
    CUSPARSE.csrmm2!('N','T',1f0,x4,w4,0f0,z4,'O')
    CUBLAS.geam!('T','T',1f0,z4,0f0,z4,y4) # need temp variable :(
    gpusync()
end
@test @show to_host(y4)==y

# going back: dw = dy * x', we want sparse dw, so we need to use csrgemm.
# we already have x' sparse and csr.
# we convert dy from dense to csr.  dw needs lots of allocation which hopefully can be avoided.

info("1x CPU back: dense x sparse (single iter) 280ms")
dw = dy = nothing; gc()
dy = rand!(similar(y))
@time dw = dy * x'

info("1000x CPU back: all sparse 3.26ms")
dw0 = dy0 = nothing; gc()
dy0 = sparse(dy)
@time for i=1:1000
    dw0 = dy0 * x'
end
@test @show full(dw0) == dw

info("1000x GPU back (all sparse) using dw=gemm(dy,x') 8.55ms")
dw5 = x5 = dy5 = nothing; gc()
x5 = csc2csr(x)                 # T
dy5 = sparse(CudaArray(dy)) # N
dw5 = nothing
@time for i=1:1000
    dw5 = CUSPARSE.gemm('N','N',dy5,x5,'O','O','O')
    gpusync()
end
@test @show full(to_host(dw5)) == dw

info("1000x GPU back (all sparse) using gemm!(dy,x',dw) 7.26ms")
dw6 = x6 = dy6 = nothing; gc()
x6 = csc2csr(x)                 # T
dy6 = sparse(CudaArray(dy))     # N
dw6 = CudaSparseMatrixCSR(spzeros(Float32,size(w)...)) # N
@time for i=1:1000
    CUSPARSE.gemm!('N','N',dy6,x6,dw6,'O')
    gpusync()
end
@test @show full(to_host(dw6)) == dw

info("1000x GPU back (all sparse) using dw'=gemm(x,dy') 3.39ms")
dw7a = x7a = dy7a = nothing; gc()
x7a = CudaSparseMatrixCSR(x)                 # N
dy7a = sparse(CudaArray(dy')) # T
dw7a = nothing                # T
@time for i=1:1000
    dw7a = CUSPARSE.gemm('N','N',x7a,dy7a,'O','O','O')
    gpusync()
end
@test @show full(to_host(dw7a)) == dw'

info("1000x GPU back (all sparse) using gemm!(x,dy',dw') 2.13ms")
dw7 = x7 = dy7 = nothing; gc()
x7 = CudaSparseMatrixCSR(x)                 # N
dy7 = sparse(CudaArray(dy'))                # T
dw7 = CudaSparseMatrixCSR(spzeros(Float32,size(w')...)) # T
@time for i=1:1000              # 2.22s
    CUSPARSE.gemm!('N','N',x7,dy7,dw7,'O')
    gpusync()
end
@test @show full(to_host(dw7)) == dw'

# dw'= x * dy'
info("1000x GPU back (sparse x dense) using csrmm!(x,dy',dw') 22.3ms")
dw8 = x8 = dy8 = nothing; gc()
x8 = CudaSparseMatrixCSR(x)                 # N
dy8 = CudaArray(dy')                        # T
dw8 = fill!(CudaArray(w'),0)                # T
@time for i=1:100              # 22.3s
    CUSPARSE.csrmm!('N',1f0,x8,dy8,0f0,dw8,'O')
    gpusync()
end
@test @show full(to_host(dw8)) == dw'

# dw = dy * x'
info("1000x GPU back (all dense) using gemm!(dy,x',dw) 16.6ms")
dw9 = x9 = dy9 = nothing; gc()
dy9 = CudaArray(dy)             # N
x9 = CudaArray(full(x'))        # T
dw9 = fill!(CudaArray(w),0)     # N
@time for i=1:100              # 16.6s
    A_mul_B!(dw9,dy9,x9)
    gpusync()
end
@test @show full(to_host(dw9)) == dw
