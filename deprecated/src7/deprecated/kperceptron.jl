# TODO: deprecate KUsparse -> CudaSparseMatrixCSC
# TODO: reimplement kperceptron in new net language

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

type KPerceptron <: Loss
    nclass	# number of output classes
    kernel      # kernel function
    p           # kernel parameter tuple
    s           # support vectors
    x           # input minibatch
    k           # kernel matrix
    y           # model output
    u           # number of training updates
    u2          # the last time w2 was calculated
    w0          # regular weights
    w1          # w2-u*w0 (kept up to date during training)
    w2          # summed weights (computed before prediction)
    dw0         # new weights
    dw1         # new weights
    dj          # indices for new support vectors
    dn          # number of new support vectors
    KPerceptron(nclass,kernel,kparams=nothing)=new(nclass,kernel,kparams)
end

function forw(l::KPerceptron, x; train=true, o...)
    initforw(l, x, train)    
    l.k = l.kernel(l.k, l.x, l.s, l.p)          # l.s generally larger, so we will transpose l.x, e.g. k=x'*s
    w = (train ? l.w0 : l.w2)                   # w2 averaged, w0 regular weights
    A_mul_Bt!(l.y, w, l.k)                      # l.y = w * l.k'
    return l.y
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

function update!(l::KPerceptron; o...) # 198
    l.u += size(l.x,2)
    dj = sub(l.dj, 1:l.dn)
    l.s  = ccat!(l.s,  l.x,   dj)
    l.w0 = ccat!(l.w0, l.dw0, dj)
    l.w1 = ccat!(l.w1, l.dw1, dj)
    l.k  = resize!(l.k, (size(l.k,1),size(l.s,2)))
end

# hcat!{T}(a::Matrix{T}, b::Matrix{T}, vj=(1:size(b,2)), nj=length(vj))=[a b[:,vj[1:nj]]]
# size!(a::Array, d::Dims)=(size(a)==d ? a : Array(eltype(a),d))

# To preserve the behavior and minimize the space, get rid of everything except:
# nclass, kernel, p, s, u, w0, w1
# Also filter the support vectors to keep only the unique ones.
function strip!(l::KPerceptron)
    l.x=l.k=l.y=l.dw0=l.dw1=l.dj=l.dn=nothing
    (l.s,l.w0,l.w1)=uniq!(l.s,l.w0,l.w1)
    return l
end

# We only support KUsparse/KUdense for x, s; KUdense for all else.

function initforw(l::KPerceptron, x, train)
    @assert isa(x, KUsparse) || isa(x, KUdense)
    if !isdefined(l,:s)                         # first initialization
        similar!(l,:s,x,size(x,1),0)      	# s matches x in location, sparseness, eltype, orientation
        l.w0 = KUdense(atype(x),eltype(x),(l.nclass, 0)) # w matches x in location and eltype but is dense
        l.w1 = copy(l.w0)
        l.w2 = copy(l.w0)
        l.u = l.u2 = 0
    end
    l.x = x                                     # x can be cpu/gpu dense/sparse
    @assert typeof(l.x) == typeof(l.s) 
    @assert atype(l.x) == atype(l.w0)
    @assert eltype(l.x) == eltype(l.s) == eltype(l.w0)
    @assert size(l.s) == (size(l.x,1), size(l.w0,2))
    similar!(l,:y,l.w0,(size(l.w0,1),size(l.x,2)))
    similar!(l,:k,l.w0,(size(l.x,2),size(l.s,2)))
    if !train && (l.u2 != l.u)
        copy!(l.w2, l.w1)
        axpy!(l.u, l.w0, l.w2)
        l.u2 = l.u
    end
end

function initback(l::KPerceptron, z::KUdense, y::KUdense=l.y)
    @assert size(z) == size(y)
    z = to_host(z.arr)
    y = to_host(y.arr)
    similar!(l,:dw0,y)
    similar!(l,:dw1,y)
    similar!(l,:dj,y,Int32,size(y,2))
    l.dn = 0
    fill!(l.dw0,0)
    fill!(l.dw1,0)
    return (y,z)
end

# Some common kernels
# http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications

klinear0(k, x, s, p)=(x.' * s)
kpoly0(k, x, s, p)=((x.' * s + p[1]) .^ p[2])
kgauss0(k, x, s, p)=exp(-p[1] * broadcast(+, sum(x.^2,1).', broadcast(+, sum(s.^2,1), -2*(x.' * s))))

# More efficient implementations:

klinear(k, x, s, p)=At_mul_B!(k, x, s)          # k=x'*s

function kpoly(k, x, s, p)
    k = klinear(k, x, s, p)                                               # 1670
    kpolymap(k, p[1], p[2])
    return k
end

function kpolymap(k::Array, c, d)
    @inbounds @simd for i=1:length(k)
        k[i] = (k[i] + c).^d
    end
    return k
end

kpolymap(k::KUdense, c, d)=(kpolymap(k.arr, c, d); k)

# This is a more efficient implementation that covers kpoly + kpolymap
# Do we need gpu kpolymap if we use this?
kpoly{A<:CudaArray}(k::KUdense{A}, x::KUsparse{A}, s::KUsparse{A}, p)=(kpoly(k.arr, convert(KUsparse,x), convert(KUsparse,s), p); k)

kgauss(k::KUdense, x::KUdense, s::KUdense, p)=(kgauss(k.arr, x.arr, s.arr, p); k)
kgauss{A<:BaseArray}(k::KUdense{A}, x::KUsparse{A}, s::KUsparse{A}, p)=(kgauss(k.arr, convert(KUsparse, x), convert(KUsparse, s), p); k)
kgauss(k::Array, x::KUsparse{Array}, s::KUsparse{Array}, p)=(kgauss(k, convert(SparseMatrixCSC,x), convert(SparseMatrixCSC,s), p); k)

function kgauss(k::AbstractArray, x::AbstractArray, s::AbstractArray, p)         # 2582
    k = klinear(k, x, s, p) # 1741
    xx = sum(x.^2,1) # 10
    ss = sum(s.^2,1) # 419 Can be cached
    # return exp(-p[1] * broadcast!(+, k, xx', broadcast!(+, k, ss, -2*k)))
    k1,k2 = size(k); p1 = p[1]
    @inbounds @simd for i=1:k1
        @inbounds @simd for j=1:k2
            k[i,j] = exp(-p1 * (xx[i] + ss[j] - 2*k[i,j]))
        end
    end
    return k
end

if GPU

function kgauss(k::CudaArray{Float32}, x::CudaArray{Float32}, s::CudaArray{Float32}, p)
    @assert size(x,1)==size(s,1)
    @assert size(k)==(size(x,2),size(s,2))
    k = klinear(k, x, s, p)
    x2 = CudaArray(Float32, size(x,2))
    s2 = CudaArray(Float32, size(s,2))
    ccall((:kgauss32sum,libkunet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat}),size(x,1),size(x,2),x,x2)
    ccall((:kgauss32sum,libkunet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat}),size(s,1),size(s,2),s,s2)
    ccall((:kgauss32map,libkunet),Void,(Cint,Cint,Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat},Cfloat),
          size(x,2),size(s,2),x2,s2,k,p[1])
    gpusync()
    free(x2); free(s2)
    return k
end

function kgauss(k::CudaArray{Float64}, x::CudaArray{Float64}, s::CudaArray{Float64}, p)
    @assert size(x,1)==size(s,1)
    @assert size(k)==(size(x,2),size(s,2))
    k = klinear(k, x, s, p)
    x2 = CudaArray(Float64, size(x,2))
    s2 = CudaArray(Float64, size(s,2))
    ccall((:kgauss64sum,libkunet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble}),size(x,1),size(x,2),x,x2)
    ccall((:kgauss64sum,libkunet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble}),size(s,1),size(s,2),s,s2)
    ccall((:kgauss64map,libkunet),Void,(Cint,Cint,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Cdouble),
          size(x,2),size(s,2),x2,s2,k,p[1])
    free(x2); free(s2)
    gpusync()
    return k
end

function kgauss(k::CudaArray{Float32}, x::KUsparse{CudaArray,Float32}, s::KUsparse{CudaArray,Float32}, p)
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:kgauss32,libkunet),Void,
          (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Cfloat),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k,p[1])
    gpusync()
    return k
end

function kgauss(k::CudaArray{Float64}, x::KUsparse{CudaArray,Float64}, s::KUsparse{CudaArray,Float64}, p)
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:kgauss64,libkunet),Void,
          (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Cdouble),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k,p[1])
    gpusync()
    return k
end

function kpolymap(k::CudaArray{Float32}, c, d)
    ccall((:kpolymap32,libkunet),Void,
          (Cint,Ptr{Cfloat},Cfloat,Cfloat),
          length(k),k,c,d)
    gpusync()
    return k
end

function kpolymap(k::CudaArray{Float64}, c, d)
    ccall((:kpolymap64,libkunet),Void,
          (Cint,Ptr{Cdouble},Cdouble,Cdouble),
          length(k),k,c,d)
    gpusync()
    return k
end

function kpoly(k::CudaArray{Float32}, x::KUsparse{CudaArray,Float32}, s::KUsparse{CudaArray,Float32}, p)
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:kpoly32,libkunet),Void,
          (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Cfloat,Cfloat),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k,p[1],p[2])
    gpusync()
    return k
end

function kpoly(k::CudaArray{Float64}, x::KUsparse{CudaArray,Float64}, s::KUsparse{CudaArray,Float64}, p)
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:kpoly64,libkunet),Void,
          (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Cdouble,Cdouble),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k,p[1],p[2])
    gpusync()
    return k
end

end # if GPU


# Failed optimization experiments:

# using NumericExtensions # but this slows kgauss down!
# using InplaceOps # maybe later for more readable code...

# # Why is kpoly slower than kgauss?  This is not faster either:
# function kpoly1(k, x, s, p)
#     k = klinear(k, x, s, p)
#     return (k + p[1]).^p[2]
# end

# function kgauss2(x::SparseMatrixCSC, s::SparseMatrixCSC, p, k)         # 2582
#     k = klinear(k, x, s, p) # 1741
#     xx = sum(x.^2,1) # 10
#     ss = sum(s.^2,1) # 419 Can be cached
#     k = broadcast!(+, k, xx', broadcast!(+, k, ss, -2*k))
#     g = -p[1]
#     @in1! k .* g
#     return exp!(k)
# end

# # This is much slower than kgauss and kgauss0

# function kgauss1(x::SparseMatrixCSC, s::SparseMatrixCSC, p, k)
#     k = klinear(k, x, s, p) # 1741
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

# function kgauss2(k, x, s, p)                    # buggy: does not take into account cells where one matrix is 0
#     k2 = kgauss(x, s, p, copy(k))
#     x = x'
#     @assert size(k)==(size(x,1), size(s,2))
#     fill!(k, zero(eltype(k)))
#     @inbounds @simd for scol=1:size(s,2)
#         @inbounds @simd for sp=s.colptr[scol]:(s.colptr[scol+1]-1)
#             srow = s.rowval[sp]
#             sval = s.nzval[sp]  # 133
#             @inbounds @simd for xp=x.colptr[srow]:(x.colptr[srow+1]-1)
#                 xrow = x.rowval[xp] # 63
#                 xval = x.nzval[xp]  # 217
#                 kinc = (xval - sval)^2
#                 k[xrow,scol] += kinc
#             end
#         end
#     end
#     g = p[1]
#     @inbounds @simd for i=1:length(k); k[i] = exp(-g*k[i]); end
#     isempty(k) || (@show maximum(abs(k-k2)))
#     return k
# end

# buggy for the same reason
# function kgauss1(x::Sparse{CudaArray,Float32}, s::Sparse{CudaArray,Float32}, p::Vector{Float32}, k::CudaArray{Float32})
#     t = x' # do this somewhere else?
#     @assert size(k)==(size(t,1), size(s,2))
#     isempty(k) && return k
#     fill!(k, zero(eltype(k)))
#     ccall((:kgauss32,libkunet),Void,
#           (Cint,Cint,Ptr{Float32},Ptr{Cint},Ptr{Cint},Ptr{Float32},Ptr{Cint},Ptr{Cint},Float32,Ptr{Float32}),
#           size(t,1),size(s,2),t.nzval,t.rowval,t.colptr,s.nzval,s.rowval,s.colptr,p[1],k)
#     k1 = cpucopy(k)
#     k2 = kgauss1(cpucopy(x),cpucopy(s),p,cpucopy(k))
#     isempty(k1) || (@show maximum(abs(k1-k2)))
#     return k
#     #copy!(k, kgauss(cpucopy(x),cpucopy(s),p,cpucopy(k)))
# end

# function kgauss3(x,s,p,k)  # Too slow
#     @assert size(k)==(size(x,2),size(s,2))
#     for i=1:size(x,2)
#         for j=1:size(s,2)
#             k[i,j] = zero(eltype(k)) # calculate k[i,j]
#             x1 = x.colptr[i]; x2 = x.colptr[i+1]
#             s1 = s.colptr[j]; s2 = s.colptr[j+1]
#             xr = x.rowval[x1]; sr = s.rowval[s1]
#             xv = x.nzval[x1]; sv = s.nzval[s1]
#             while (x1 < x2 || s1 < s2)
#                 xr = (x1 < x2 ? x.rowval[x1] : size(x,1)+1)
#                 sr = (s1 < s2 ? s.rowval[s1] : size(s,1)+1)
#                 if xr == sr
#                     xv = x.nzval[x1]; sv = s.nzval[s1]
#                     k[i,j] += (xv-sv)^2
#                     x1 += 1; s1 += 1
#                 elseif xr < sr
#                     xv = x.nzval[x1]
#                     k[i,j] += xv^2
#                     x1 += 1
#                 elseif sr < xr
#                     sv = s.nzval[s1]
#                     k[i,j] += sv^2
#                     s1 += 1
#                 end
#             end
#         end
#     end
#     g = p[1]
#     for i=1:length(k); k[i] = exp(-g*k[i]); end
#     return k
# end

# function klinear3(x,s,p,k)  # Too slow on cpu, ok on gpu
#     @assert size(k)==(size(x,2),size(s,2))
#     @inbounds @simd for i=1:size(x,2)
#         @inbounds @simd for j=1:size(s,2)
#             kij = zero(Float64)
#             x1 = x.colptr[i]; x2 = x.colptr[i+1]
#             s1 = s.colptr[j]; s2 = s.colptr[j+1]
#             while ((x1 < x2) && (s1 < s2))
#                 xr = x.rowval[x1]; sr = s.rowval[s1]
#                 (xr < sr ? x1+=1 :
#                  sr < xr ? s1+=1 :
#                  (kij += x.nzval[x1] * s.nzval[s1]; x1+=1; s1+=1))
#             end
#             k[i,j] = kij
#         end
#     end
#     return k
# end

# klinear4(x,s,p,k)=A_mul_B!(k,x.',s)

