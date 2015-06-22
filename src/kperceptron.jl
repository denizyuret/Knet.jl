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
    z           # desired output
    u           # number of training instances
    w0          # regular weights
    w1          # u*w0-w2 (kept up to date during training)
    w2          # summed weights (computed before prediction)
    KPerceptron(nclass,kernel,kparams=nothing)=new(nclass,kernel,kparams)
end

function forw(l::KPerceptron, x::KUnetArray; predict=false, o...)
    initforw(l, x, predict)    
    l.k = l.kernel(l.x, l.s, l.p, l.k)          # l.s generally larger, so we will transpose l.x, e.g. k=x'*s
    w = (predict ? l.w2 : l.w0)                 # w2 averaged, w0 regular weights
    l.y = gemm!('N','T',one(eltype(w)),w,l.k,zero(eltype(w)),l.y) # l.y = w * l.k'
end

function initforw(l::KPerceptron, x::KUnetArray, predict)
    x = full(x)
    if !isdefined(l,:s)                         # first initialization
        similar!(l,:s,x,size(x,1),0)      	# s matches x in location, sparseness, eltype, orientation
        wtype = gpu() ? CudaArray : Array         # w matches x in location and eltype but is dense
        xtype = eltype(x)
        l.w0 = wtype(xtype, l.nclass, 0)        # TODO: allocate extra space for expansion
        l.w1 = wtype(xtype, l.nclass, 0)
        l.w2 = nothing
        l.u = zero(xtype)
    end
    l.x = x                                     # x can be cpu/gpu dense/sparse
    @assert typeof(l.x) == typeof(l.s) "typeof:$((typeof(l.x),typeof(l.s)))"          # x and s have the same type
    @assert size(l.x, 1) == size(l.s, 1)        # and same orientation
    @assert isongpu(l.x) == isongpu(l.w0) "isongpu:$((isongpu(l.x),isongpu(l.w0)))"       # w has the same location as x
    @assert eltype(l.x) == eltype(l.w0)         # w has the same eltype as x
    @assert size(l.w0, 2) == size(l.s, 2)       # w has same number of cols as s
    similar!(l,:y,l.w0,(size(l.w0,1),size(l.x,2)))
    similar!(l,:k,l.w0,(size(l.x,2),size(l.s,2)))
    if predict && (l.w2 == nothing)
        l.w2 = l.u * l.w0 - l.w1
        # making sure we don't get overflow
        @assert maximum(abs(l.w2)) < sqrt(typemax(eltype(l.w2)))
    end
end

function update(l::KPerceptron; o...) # 198
    l.w2 = nothing                              # make sure w2 is reset when w0,w1,u changes
    w = zeros(eltype(l.w0),size(l.w0,1),1)      # use small arrays for faster hcat
    w0 = similar(l.w0, size(l.w0,1), 0)         # TODO: write efficient hcat!
    w1 = similar(l.w1, size(l.w1,1), 0)
    s = similar(l.x, size(l.x,1), 0)
    @inbounds for j=1:size(l.z,2)
        (cz,cy,ymax,zmax) = (0,0,typemin(eltype(l.y)),typemin(eltype(l.z)))
        @inbounds for i=1:l.nclass
            l.z[i,j] > zmax && ((cz,zmax) = (i,l.z[i,j])) # find the correct answer
            l.y[i,j] > ymax && ((cy,ymax) = (i,l.y[i,j])) # find the model answer
        end
        if cz != cy # if model answer is not correct l.x[:,j] becomes a new support vector
            s = [s l.x[:,j]] # 57
            w[cz] = 1
            w[cy] = -1
            w0 = [w0 w]
            w[cz] = l.u
            w[cy] = -l.u
            w1 = [w1 w]
            w[cz] = w[cy] = 0
        end
        l.u += one(l.u)      # 32 increment counter regardless of update
    end
    l.s = [l.s s]            # 40
    l.w0 = [l.w0 w0]
    l.w1 = [l.w1 w1]
end

function back(l::KPerceptron, z; returndx=false, o...)
    @assert size(z) == size(l.y)
    returndx && error("KPerceptron does not know how to return dx")
    l.z = z   # just record the correct answers in l.z
end

# Some common kernels
# http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications

kgauss0(x, s, p, k)=exp(-p[1] * broadcast(+, sum(x.^2,1)', broadcast(+, sum(s.^2,1), -2*(x' * s))))
kpoly0(x, s, p, k)=((x' * s + p[1]) .^ p[2])
klinear0(x, s, p, k)=full(x' * s)

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

klinear(x, s, p, k)=gemm!('T','N',one(eltype(x)),x,s,zero(eltype(k)),k) # k=x'*s

function klinear(x::SparseMatrixCSC, s::SparseMatrixCSC, p, k) # 1607
    @assert size(k)==(size(x,2), size(s,2))
    x = x'                                                          # 77
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

