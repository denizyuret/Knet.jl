# (c) Deniz Yuret, June 19, 2015
# This is a standalone (single layer) implementation of the averaged
# kernel perceptron algorithm.
# Based on the k_perceptron_multi_train.m in the DOGMA library 
# (http://dogma.sourceforge.net) by Francesco Orabona which references:
#     - Crammer, K., & Singer Y. (2003).
#       Ultraconservative Online Algorithms for Multiclass Problems.
#       Journal of Machine Learning Research 3, (pp. 951-991).
# Averaging optimization trick with w0/w1/w2 based on:
# http://ciml.info/dl/v0_9/ciml-v0_9-ch03.pdf.

type KPerceptron <: Layer
    nclass	# number of output classes
    kernel      # kernel function
    p           # kernel parameter array
    s           # support vectors
    x           # input minibatch
    k           # kernel matrix
    y           # model output
    u           # number of training instances
    w0          # regular weights
    w1          # w2-u*w0 (kept up to date during training)
    w2          # summed weights (computed before prediction)
    dw0         # new weights
    dw1         # new weights
    dj          # indices for new support vectors
    dn          # number of new support vectors
    KPerceptron(nclass,kernel,kparams=nothing)=new(nclass,kernel,kparams)
end

function forw(l::KPerceptron, x; predict=false, o...)
    initforw(l, x, predict)    
    l.k = l.kernel(l.x, l.s, l.p, l.k)          # l.s generally larger, so we will transpose l.x, e.g. k=x'*s
    w = (predict ? l.w2 : l.w0)                 # w2 averaged, w0 regular weights
    A_mul_Bt!(l.y, w, l.k)                      # l.y = w * l.k'
end

function back(l::KPerceptron, z; returndx=false, o...)
    returndx && error("KPerceptron does not know how to return dx")
    (y,z) = initback(l,z)
    @inbounds for j=1:size(z,2)
        (cz,cy,ymax,zmax) = (0,0,typemin(eltype(y)),typemin(eltype(z)))
        @inbounds for i=1:l.nclass
            z[i,j] > zmax && ((cz,zmax) = (i,z[i,j])) # find the correct answer
            y[i,j] > ymax && ((cy,ymax) = (i,y[i,j])) # find the model answer
        end
        if cz != cy # if model answer is not correct l.x[:,j] becomes a new support vector
            l.dn += one(l.dn)
            l.dj[l.dn] = j
            u = l.u + j - 1
            l.dw1[cz,j] = -u
            l.dw1[cy,j] = u
            l.dw0[cz,j] = 1
            l.dw0[cy,j] = -1
        end
    end
end

function update(l::KPerceptron; o...) # 198
    l.w2 = nothing # make sure w2 is reset when w0,w1,u changes
    l.u += size(l.x,2)
    l.s  = hcat!(l.s,  l.x,   l.dj, l.dn)
    l.w0 = hcat!(l.w0, l.dw0, l.dj, l.dn)
    l.w1 = hcat!(l.w1, l.dw1, l.dj, l.dn)
end

# hcat!(a,b,vj,nj)=[a b[:,vj[1:nj]]]

hcat!{Tv,Ti<:Integer}(a::Matrix{Tv}, b::Matrix{Tv}, vj::Vector{Ti}, nj::Integer)=[a b[:,vj[1:nj]]]

function hcat!{Tv,Ti<:Integer}(a::SparseMatrixCSC{Tv}, b::SparseMatrixCSC{Tv}, vj::Vector{Ti}, nj::Integer)
    # a: m, n, colptr, rowval, nzval
    # colptr[i]: starting index (in rowval,nzval) of column i
    # colptr[n+1]: nz+1
    @assert size(a,1) == size(b,1)
    @inbounds for i=1:nj
        j = vj[i]  # concat b[:,j]
        b0 = b.colptr[j]
        b1 = b.colptr[j+1]-1
        nz = b1-b0+1
        a.colptr = push!(a.colptr, a.colptr[end]+nz)
        if nz > 0
            a.rowval = append!(a.rowval, b.rowval[b0:b1])
            a.nzval = append!(a.nzval, b.nzval[b0:b1])
        end
    end
    a.n += nj
    return a
end


function hcat!(a::CudaMatrix, b::KUnetArray, vj::Vector, nj::Integer)
    @assert size(a,1) == size(b,1)
    @assert eltype(a) == eltype(b)
    (nrows,ncols) = size(a)
    c = CudaArray(eltype(a), nrows, ncols+nj)   # TODO: is there realloc?
    copy!(c, 1, a, 1, length(a))
    nc = length(a)+1
    for i=1:nj
        nb = (vj[i]-1)*nrows+1
        copy!(c, nc, b, nb, nrows)
        nc += nrows
    end
    return c
end

function initforw(l::KPerceptron, x::KUnetArray, predict)
    if !isdefined(l,:s)                         # first initialization
        similar!(l,:s,x,size(x,1),0)      	# s matches x in location, sparseness, eltype, orientation
        wtype = gpu() ? CudaArray : Array       # w matches x in location and eltype but is dense
        xtype = eltype(x)
        l.w0 = wtype(xtype, l.nclass, 0)        # should we allocate extra space for expansion?
        l.w1 = wtype(xtype, l.nclass, 0)
        l.w2 = nothing
        l.u = zero(xtype)
    end
    l.x = x                                     # x can be cpu/gpu dense/sparse
    @assert typeof(l.x) == typeof(l.s)          # x and s have the same type
    @assert size(l.x, 1) == size(l.s, 1)        # and same orientation
    @assert isongpu(l.x) == isongpu(l.w0)       # w has the same location as x
    @assert eltype(l.x) == eltype(l.w0)         # w has the same eltype as x
    @assert size(l.w0, 2) == size(l.s, 2)       # w has same number of cols as s
    similar!(l,:y,l.w0,(size(l.w0,1),size(l.x,2)))
    similar!(l,:k,l.w0,(size(l.x,2),size(l.s,2)))
    if predict && (l.w2 == nothing)
        # l.w2 = l.u * l.w0 + l.w1
        l.w2 = axpy!(length(l.w0), l.u, l.w0, 1, copy(l.w1), 1)
        # making sure we don't get overflow, especially with integer types
        # @assert maximum(abs(l.w2)) < sqrt(typemax(eltype(l.w2)))
    end
end

function initback(l::KPerceptron, z, y=l.y)
    @assert size(z) == size(y)
    isongpu(z) && (z = cpucopy(z))
    isongpu(y) && (y = cpucopy(y))
    similar!(l,:dw0,y)
    similar!(l,:dw1,y)
    similar!(l,:dj,y,Int32,size(z,2))
    l.dn = zero(Int32)
    fill!(l.dw0,zero(eltype(l.dw0)))
    fill!(l.dw1,zero(eltype(l.dw1)))
    return (y,z)
end

# Some common kernels
# http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications

kgauss0(x, s, p, k)=exp(-p[1] * broadcast(+, sum(x.^2,1).', broadcast(+, sum(s.^2,1), -2*(x.' * s))))
kpoly0(x, s, p, k)=((x.' * s + p[1]) .^ p[2])
klinear0(x, s, p, k)=(x.' * s)

# More efficient implementations:
# using NumericExtensions # this slows kgauss down!
# using InplaceOps

function kgauss(x, s, p, k)         # 2582
    k = klinear(x, s, p, k) # 1741
    xx = sum(x.^2,1) # 10
    ss = sum(s.^2,1) # 419 Can be cached
    return exp(-p[1] * broadcast!(+, k, xx', broadcast!(+, k, ss, -2*k)))
    # return exp(-p[1] * broadcast(+, xx', broadcast(+, ss, -2*k)))
    # return exp!(-p[1] * broadcast!(+, k, xx', broadcast!(+, k, ss, -2*k)))
end

# function kgauss2(x::SparseMatrixCSC, s::SparseMatrixCSC, p, k)         # 2582
#     k = klinear(x, s, p, k) # 1741
#     xx = sum(x.^2,1) # 10
#     ss = sum(s.^2,1) # 419 Can be cached
#     k = broadcast!(+, k, xx', broadcast!(+, k, ss, -2*k))
#     g = -p[1]
#     @in1! k .* g
#     return exp!(k)
# end

# # This is much slower than kgauss and kgauss0

# function kgauss1(x::SparseMatrixCSC, s::SparseMatrixCSC, p, k)
#     k = klinear(x, s, p, k) # 1741
#     xx = sum(x.^2,1) # 10
#     ss = sum(s.^2,1) # 419 Can be cached
#     # return exp(-p[1] * broadcast(+, xx', broadcast(+, ss, -2*k))) # 412
#     g = p[1]
#     @inbounds @simd for i=1:size(k,1)
#         @inbounds @simd for j=1:size(k,2)
#             k[i,j] = exp(-g * (xx[i] + ss[j] - 2*k[i,j]))
#         end
#     end
#     return k
# end

function kpoly(x, s, p, k)
    k = klinear(x, s, p, k)                                               # 1670
    (c,d) = p
    @inbounds @simd for i=1:length(k); k[i] = (k[i] + c).^d; end  # 1413
    return k
end

# Why is kpoly slower than kgauss?  This is not faster either:
function kpoly1(x, s, p, k)
    k = klinear(x, s, p, k)
    return (k + p[1]).^p[2]
end

# Better would be to define this in terms of A(t?)_mul_B! so operators work
# klinear(x, s, p, k)=gemm!('T','N',one(eltype(x)),x,s,zero(eltype(k)),k) # k=x'*s
klinear(x, s, p, k)=At_mul_B!(k, x, s)

import Base: A_mul_Bt,  At_mul_B
import Base: A_mul_Bt!, At_mul_B!, A_mul_B!

# cpu/sparse

At_mul_B!(k::Matrix, x::SparseMatrixCSC, s::SparseMatrixCSC)=A_mul_B!(k,x',s)

function A_mul_B!(k::Matrix, x::SparseMatrixCSC, s::SparseMatrixCSC) # 1607
    @assert size(k)==(size(x,1), size(s,2))
    fill!(k, zero(eltype(k)))
    @inbounds @simd for scol=1:size(s,2)
        @inbounds @simd for sp=s.colptr[scol]:(s.colptr[scol+1]-1)
            srow = s.rowval[sp]
            sval = s.nzval[sp]  # 133
            @inbounds @simd for xp=x.colptr[srow]:(x.colptr[srow+1]-1)
                xrow = x.rowval[xp] # 63
                xval = x.nzval[xp]  # 217
                yinc = xval * sval  # 245
                k[xrow,scol] += yinc # 789
            end
        end
    end
    return k
end

# gpu/sparse: all the cpucopy items need implementing
At_mul_B!{T}(k::CudaMatrix{T}, x::CudaSparseMatrixCSC{T}, s::CudaSparseMatrixCSC{T})=A_mul_B!(k,x.',s)
transpose(x::CudaSparseMatrixCSC)=gpucopy(cpucopy(x)')
A_mul_B!{T}(k::CudaMatrix{T}, x::CudaSparseMatrixCSC{T}, s::CudaSparseMatrixCSC{T})=copy!(k, full(cpucopy(x)*cpucopy(s)))
hcat!{T}(x::CudaSparseMatrixCSC{T}, s::CudaSparseMatrixCSC{T},vj,nj)=gpucopy(hcat!(cpucopy(x),cpucopy(s),cpucopy(vj),nj))
kpoly(x::CudaSparseMatrixCSC, s::CudaSparseMatrixCSC, p, k::CudaArray)=copy!(k, kpoly(cpucopy(x),cpucopy(s),p,cpucopy(k)))
kgauss(x::CudaSparseMatrixCSC, s::CudaSparseMatrixCSC, p, k::CudaArray)=copy!(k, kgauss(cpucopy(x),cpucopy(s),p,cpucopy(k)))

# gpu/dense
A_mul_Bt!{T}(k::CudaMatrix{T}, x::CudaMatrix{T}, s::CudaMatrix{T})=gemm!('N','T',one(T),x,s,zero(T),k)
At_mul_B!{T}(k::CudaMatrix{T}, x::CudaMatrix{T}, s::CudaMatrix{T})=gemm!('T','N',one(T),x,s,zero(T),k)
kpoly(x::CudaArray, s::CudaArray, p, k::CudaArray)=gpucopy(kpoly(cpucopy(x),cpucopy(s),p,cpucopy(k)))
kgauss(x::CudaArray, s::CudaArray, p, k::CudaArray)=gpucopy(kgauss(cpucopy(x),cpucopy(s),p,cpucopy(k)))

