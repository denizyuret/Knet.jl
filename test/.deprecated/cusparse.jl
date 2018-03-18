using CUDArt,CUBLAS,CUSPARSE,Knet,Base.Test
using Knet: gemm!, gpusync

# include("isapprox.jl")

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

# Experiment 1:
# nh = 1000
# nd = 100000
# nb = 100

# Experiment 2:
nh = 1500
nd = 10000
nb = 20

info("1000x CPU forw: 0.2ms")
w = x = y = nothing; gc()
w = rand(Float32,  nh, nd)      # N
x = sponehot(Float32, nd, nb)      # N
rand!(x.rowval, 1:nd)
y = Array(Float32, nh, nb)      # N
@time for i=1:1000              # 1:0.5ms, 2:0.2ms
    A_mul_B!(y, w, x)
end
@test @show to_host(y)==to_host(w)[:,x.rowval]

# info("1000x GPU forw using KUsparse A_mul_B!(y,w,x) 2.5ms")
# w1 = x1 = y1 = nothing; gc()
# w1 = CudaArray(w)               # N
# x1 = convert(KUsparse{CudaArray,Float32}, x) # N
# y1 = fill!(CudaArray(y),0)                   # N
# @time for i=1:1000
#     A_mul_B!(y1, w1, x1)        # 1:2.5ms
#     gpusync()
# end
# @test @show to_host(y1)==y
# CUBLAS.geam!('T','T',1f0,w1,0f0,w1,w2)

# if false 

info("10000x GPU forw using csrmm!(x',w',y') 0.056ms")
w2 = x2 = y2 = nothing; gc()
x2 = CudaSparseMatrixCSR(x')    # T
w2 = CudaArray(w')              # T
y2 = fill!(CudaArray(y'),0)     # T
@time for i=1:10000             # 1:0.08ms 2:0.056ms
    CUSPARSE.csrmm!('N',1f0,x2,w2,0f0,y2,'O')
    gpusync()
end
@test @show to_host(y2)==y'

info("10000x GPU forw using csrmm2!(x',w,y') 0.036ms")
w3 = x3 = y3 = nothing; gc()
x3 = x2                         # T
w3 = CudaArray(w)               # N
y3 = similar(y2)                # T
@time for i=1:10000             # 1:0.04ms, 2:0.036ms
    CUSPARSE.csrmm2!('N','T',1f0,x3,w3,0f0,y3,'O')
    gpusync()
end
@test @show to_host(y3)==y'

info("10000x GPU forw using csrmm2!(x',w,y')+transpose 0.046ms")
w4 = x4 = y4 = nothing; gc()
x4 = csc2csr(x)                 # T: direct copy possible
w4 = w3                         # N
y4 = fill!(CudaArray(y),0)      # N
z4 = similar(y2)                # T
@time for i=1:10000             # 1:0.05ms 2:0.046ms
    CUSPARSE.csrmm2!('N','T',1f0,x4,w4,0f0,z4,'O')
    CUBLAS.geam!('T','T',1f0,z4,0f0,z4,y4) # need temp variable :(
    gpusync()
end
@test @show to_host(y4)==y

info("10000x GPU forw using A_mul_B!(y,w,x) 0.074ms")
w4a = x4a = y4a = nothing; gc()
x4a = CudaSparseMatrixCSC(x)     # N
w4a = w3                         # N
y4a = CudaArray(similar(y)) # N
@time for i=1:10000             # 
    A_mul_B!(y4a, w4a, x4a)
    gpusync()
end
@test @show convert(Array,y4a)==y

# end # if false

#### going back: dw = dy * x', 

# we want sparse dw, so we need to use csrgemm.
# new design: we have x in csc, which is equiv to x' csr
# we'll have dw and dy also in csc and run A_mul_Bt(dw,dy,x)

# we already have x' sparse and csr.
# we convert dy from dense to csr.  dw needs lots of allocation which hopefully can be avoided.

# The true cost should include: any transposes, sparsifications,
# densifications, allocations, and the final update dw+=iw, w+=dw
# we always have x sparse, dy,w dense. we have a choice for iw and dw.
# if seqlen=t=20, we have 20 (iw=dy*x'; dw+=iw) followed by one w+=dw.
# D=10K, h=1K, b=10, t=10
# x:(D,b), nnz(x)=b, D>>b, avoid densification if possible.
# w,dw,iw:(h,D), nnz(iw):h*b, nnz(dw):h*b*t
# y,dy:(h,b)

#    ........x............................................
#    ..................x..................................
#    ..x..................................................
# ...
# ...
# ...
# ...
# ...



info("1x CPU back: dense x sparse (single iter) 41ms")
dw = dy = nothing; gc()
dy = rand!(similar(y))
@time dw = dy * x'              # 1:280ms 2:41.6ms

# if false

info("1000x CPU back: all sparse 0.78ms")
dw0 = dy0 = nothing; gc()
dy0 = sparse(dy)
@time for i=1:1000              # 1:3.26ms 2:0.78ms
    dw0 = dy0 * x'
end
@test @show full(dw0) == dw

# dw = dy * x'
info("1000x GPU back (all sparse) using dw=gemm(dy,x') 1.08ms")
dw5 = x5 = dy5 = nothing; gc()
x5 = csc2csr(x)                 # T
dy5 = sparse(CudaArray(dy)) # N
dw5 = nothing
@time for i=1:1000              # 1:8.55ms 2:1.08ms
    dw5 = CUSPARSE.gemm('N','N',dy5,x5,'O','O','O')
    gpusync()
end
@test @show full(to_host(dw5)) == dw

# end # if false

# dw = dy * x'
info("1000x GPU back (all sparse) using gemm!(dy,x',dw) 0.72ms")
dw6 = x6 = dy6 = nothing; gc()
x6 = csc2csr(x)                 # T
dy6 = sparse(CudaArray(dy))     # N
dw6 = CudaSparseMatrixCSR(spzeros(Float32,size(w)...)) # N
@time for i=1:1000              # 1:7.26ms 2:0.72ms
    gemm!('N','N',dy6,x6,dw6)
    gpusync()
end
@test @show full(to_host(dw6)) == dw

# dw = dy * x'
info("1000x GPU back (all sparse) using A_mul_Bt (including sparse(dy)) 1.28ms")
dw6a = x6a = dy6a = nothing; gc()
x6a = CudaSparseMatrixCSC(x)
dy6a = CudaArray(dy)     # N
dw6a = CudaSparseMatrixCSR(spzeros(Float32,size(w)...)) # N
@time for i=1:1000              # 2:1.28ms
    A_mul_Bt!(dw6a,dy6a,x6a)
    gpusync()
end
@test @show full(to_host(dw6a)) == dw

# dw = dy * x'
info("1000x GPU back (all sparse) using A_mul_Bt (including sparse(dy) and axpy) 2.21ms")
dw6b = x6b = dy6b = nothing; gc()
x6b = CudaSparseMatrixCSC(x)
dy6b = CudaArray(dy)     # N
dw6b = CudaSparseMatrixCSR(spzeros(Float32,size(w)...)) # N
iw6b = CudaSparseMatrixCSR(spzeros(Float32,size(w)...)) # N
@time for i=1:1000              # 2:2.21ms
    A_mul_Bt!(iw6b,dy6b,x6b)
    Base.axpy!(1,iw6b,dw6b)
    gpusync()
end
@test @show full(to_host(iw6b)) == dw
@show full(to_host(dw6b)) == 1000*full(to_host(iw6b))
@show isapprox(full(to_host(dw6b)), 1000*full(to_host(iw6b)))

# if false

# dw = dy * x'
info("1000x GPU back (all sparse) using dw'=gemm(x,dy') 2.39ms")
dw7a = x7a = dy7a = nothing; gc()
x7a = CudaSparseMatrixCSR(x)                 # N
dy7a = sparse(CudaArray(dy')) # T
dw7a = nothing                # T
@time for i=1:1000            # 1:3.39ms 2:2.39ms
    dw7a = CUSPARSE.gemm('N','N',x7a,dy7a,'O','O','O')
    gpusync()
end
@test @show full(to_host(dw7a)) == dw'

info("1000x GPU back (all sparse) using gemm!(x,dy',dw') 2.09ms")
dw7 = x7 = dy7 = nothing; gc()
x7 = CudaSparseMatrixCSR(x)                 # N
dy7 = sparse(CudaArray(dy'))                # T
dw7 = CudaSparseMatrixCSR(spzeros(Float32,size(w')...)) # T
@time for i=1:1000              # 1:2.13ms 2:2.09ms
    gemm!('N','N',x7,dy7,dw7)
    gpusync()
end
@test @show full(to_host(dw7)) == dw'

# dw'= x * dy'
info("1000x GPU back (sparse x dense)->dense using csrmm!(x,dy',dw') 3.40ms")
dw8 = x8 = dy8 = nothing; gc()
x8 = CudaSparseMatrixCSR(x)                 # N
dy8 = CudaArray(dy')                        # T
dw8 = fill!(CudaArray(w'),0)                # T
@time for i=1:1000              # 1:22.3ms 2:3.40ms
    CUSPARSE.csrmm!('N',1f0,x8,dy8,0f0,dw8,'O')
    gpusync()
end
@test @show to_host(dw8) == dw'

# dw = dy * x'
info("1000x GPU back (all dense) using gemm!(dy,x',dw) 1.59ms")
dw9 = x9 = dy9 = nothing; gc()
dy9 = CudaArray(dy)             # N
x9 = CudaArray(full(x'))        # T
dw9 = fill!(CudaArray(w),0)     # N
@time for i=1:1000              # 1:16.6s 2:1.59ms
    A_mul_B!(dw9,dy9,x9)
    gpusync()
end
@test @show to_host(dw9) == dw

# end # if false
