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

function initforw(l::KPerceptron, x::KUnetArray, predict)
    ytype = gpu() ? CudaArray : Array
    xtype = eltype(x)
    if !isdefined(l,:s)                         # first initialization
        similar!(l,:s,x,size(x,1),0)      	# s matches x in location, sparseness, eltype, orientation
        gpu() && isa(l.s, CudaArray) && (l.s = CudaDynArray(l.s))
        wtype = gpu() ? CudaDynArray : Array    # w matches x in location and eltype but is dense
        l.w0 = wtype(xtype, l.nclass, 0)        # should we allocate extra space for expansion?
        l.w1 = wtype(xtype, l.nclass, 0)
        l.w2 = nothing
        l.u = zero(xtype)
    end
    l.x = x                                     # x can be cpu/gpu dense/sparse
    @assert isongpu(l.x) == isongpu(l.s) == isongpu(l.w0)
    @assert eltype(l.x) == eltype(l.s) == eltype(l.w0)
    @assert size(l.s) == (size(l.x,1), size(l.w0,2))
    similar!(l,:y,ytype,xtype,(size(l.w0,1),size(l.x,2)))
    similar!(l,:k,ytype,xtype,(size(l.x,2),size(l.s,2)))
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

klinear0(x, s, p, k)=(x.' * s)
kpoly0(x, s, p, k)=((x.' * s + p[1]) .^ p[2])
kgauss0(x, s, p, k)=exp(-p[1] * broadcast(+, sum(x.^2,1).', broadcast(+, sum(s.^2,1), -2*(x.' * s))))

# More efficient implementations:

klinear(x, s, p, k)=At_mul_B!(k, x, s)          # k=x'*s

function kpoly(x, s, p, k)
    k = klinear(x, s, p, k)                                               # 1670
    kpolymap(k, p[1], p[2])
    return k
end

function kpolymap(k, c, d)
    @inbounds @simd for i=1:length(k)
        k[i] = (k[i] + c).^d
    end
    return k
end

function kgauss(x, s, p, k)         # 2582
    k = klinear(x, s, p, k) # 1741
    xx = sum(x.^2,1) # 10
    ss = sum(s.^2,1) # 419 Can be cached
    return exp(-p[1] * broadcast!(+, k, xx', broadcast!(+, k, ss, -2*k)))
end


if GPU

kgauss(x::CudaArray, s::CudaArray, p, k::CudaArray)=gpucopy(kgauss(cpucopy(x),cpucopy(s),p,cpucopy(k)))

function kpolymap(k::CudaArray{Float32}, c::Float32, d::Float32)
    ccall((:kpolymap32,libkunet),Void,
          (Cint,Ptr{Cfloat},Cfloat,Cfloat),
          length(k),k,c,d)
    gpusync()
    return k
end

function kpolymap(k::CudaArray{Float64}, c::Float64, d::Float64)
    ccall((:kpolymap64,libkunet),Void,
          (Cint,Ptr{Cdouble},Cdouble,Cdouble),
          length(k),k,c,d)
    gpusync()
    return k
end

function kpoly(x::CudaSparseMatrixCSC{Float32}, s::CudaSparseMatrixCSC{Float32}, p, k::CudaArray{Float32})
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:kpoly32,libkunet),Void,
          (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Cfloat,Cfloat),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k,p[1],p[2])
    gpusync()
    return k
end

function kpoly(x::CudaSparseMatrixCSC{Float64}, s::CudaSparseMatrixCSC{Float64}, p, k::CudaArray{Float64})
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:kpoly64,libkunet),Void,
          (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Cdouble,Cdouble),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k,p[1])
    gpusync()
    return k
end

function kgauss(x::CudaSparseMatrixCSC{Float32}, s::CudaSparseMatrixCSC{Float32}, p, k::CudaArray{Float32})
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:kgauss32,libkunet),Void,
          (Cint,Cint,Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Ptr{Cint},Ptr{Cint},Ptr{Cfloat},Cfloat),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k,p[1])
    gpusync()
    return k
end

function kgauss(x::CudaSparseMatrixCSC{Float64}, s::CudaSparseMatrixCSC{Float64}, p, k::CudaArray{Float64})
    @assert size(k)==(size(x,2),size(s,2))
    ccall((:kgauss64,libkunet),Void,
          (Cint,Cint,Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Cdouble),
          size(x,2),size(s,2),x.nzval,x.rowval,x.colptr,s.nzval,s.rowval,s.colptr,k,p[1])
    gpusync()
    return k
end

end # if GPU


# Failed optimization experiments:

# using NumericExtensions # but this slows kgauss down!
# using InplaceOps # maybe later for more readable code...

# # Why is kpoly slower than kgauss?  This is not faster either:
# function kpoly1(x, s, p, k)
#     k = klinear(x, s, p, k)
#     return (k + p[1]).^p[2]
# end

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

# function kgauss2(x, s, p, k)                    # buggy: does not take into account cells where one matrix is 0
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
# function kgauss1(x::CudaSparseMatrixCSC{Float32}, s::CudaSparseMatrixCSC{Float32}, p::Vector{Float32}, k::CudaArray{Float32})
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

