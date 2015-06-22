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

type KPerceptron <: Layer; n; k; p; s; x; y; z; w0; w1; w2; u; 
    KPerceptron(nclass,kernel,params=nothing)=new(nclass,kernel,params)
end

function forw(l::KPerceptron, x; predict=false, o...)
    initforw(l, x, predict)
    w = (predict ? l.w2 : l.w0)
    k = l.k(l.s, l.x, l.p)
    l.y = w * k'
    # gemm!('N','T',one(eltype(w)),w,k,zero(eltype(w)),l.y) # l.y = w * k' # gemm is not any faster
end

function update(l::KPerceptron; o...) # 198
    l.w2 = nothing   # make sure these are reset when w0,w1,u changes
    w = zeros(eltype(l.w0),size(l.w0,1),1)
    w0 = similar(l.w0, size(l.w0,1), 0)
    w1 = similar(l.w1, size(l.w1,1), 0)
    s = similar(l.x, size(l.x,1), 0)
    @inbounds for j=1:size(l.z,2)
        (cz,cy,ymax,zmax) = (0,0,typemin(eltype(l.y)),typemin(eltype(l.z)))
        @inbounds for i=1:l.n
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

function initforw(l::KPerceptron, x::AbstractArray, predict)
    l.x = x
    if !isdefined(l,:s)
        l.s = similar(l.x, size(l.x,1), 0)  # matches l.x in sparsity and orientation
        l.w0 = zeros(eltype(l.x), l.n, 0)   # w,b always full
        l.w1 = zeros(eltype(l.x), l.n, 0)
        l.w2 = nothing
        l.u = zero(eltype(l.x))
    end
    @assert size(l.x, 1) == size(l.s, 1)
    @assert size(l.s, 2) == size(l.w0, 2)
    similar!(l,:y,l.w0,(size(l.w0,1),size(l.x,2)))
    if predict && (l.w2 == nothing)
        l.w2 = l.u * l.w0 - l.w1
        # making sure we don't get overflow
        @assert maximum(abs(l.w2)) < sqrt(typemax(eltype(l.w2)))
    end
end

function back(l::KPerceptron, z; returndx=false, o...)
    @assert size(z) == size(l.y)
    returndx && error("KPerceptron does not know how to return dx")
    l.z = z   # just record the correct answers in l.z
end

# Some common kernels
# http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications

kgauss0(s, x, p)=exp(-p[1] * broadcast(+, sum(x.^2,1)', broadcast(+, sum(s.^2,1), -2*(x' * s))))
kpoly0(s, x, p)=((x' * s + p[1]) .^ p[2])
klinear0(s, x, p)=full(x' * s)

# More efficient implementations:

function kgauss(s::SparseMatrixCSC, x::SparseMatrixCSC, p)         # 2582
    sx = klinear(s, x, p) # 1741
    xx = sum(x.^2,1) # 10
    ss = sum(s.^2,1) # 419 Can be cached
    return exp(-p[1] * broadcast(+, xx', broadcast(+, ss, -2*sx))) # 412
end

function kpoly(s::SparseMatrixCSC, x::SparseMatrixCSC, p)
    y = klinear(s, x, p)                                               # 1670
    @inbounds @simd for i=1:length(y); y[i] = (y[i] + p[1]).^p[2]; end  # 1413
    return y
end

# Why is kpoly slower than kgauss?  This is not faster either:
function kpoly1(s::SparseMatrixCSC, x::SparseMatrixCSC, p)
    sx = klinear(s, x, p)
    return (sx + p[1]).^p[2]
end


function klinear(s::SparseMatrixCSC, x::SparseMatrixCSC, p=nothing) # 1607
    x = x'                                                          # 77
    y = zeros(eltype(x), size(x,1), size(s,2))
    @inbounds @simd for scol=1:size(s,2)
        @inbounds @simd for sp=s.colptr[scol]:(s.colptr[scol+1]-1)
            srow = s.rowval[sp]
            sval = s.nzval[sp]  # 133
            @inbounds @simd for xp=x.colptr[srow]:(x.colptr[srow+1]-1)
                xrow = x.rowval[xp] # 63
                xval = x.nzval[xp]  # 217
                yinc = xval * sval  # 245
                y[xrow,scol] += yinc # 789
            end
        end
    end
    return y
end

# This is not any faster!

function klinear1(s::SparseMatrixCSC, x::SparseMatrixCSC, p=nothing) # 1786
    x = x'                                                          # 130
    (yrows,ycols) = (size(x,1), size(s,2))
    y = Array(eltype(x), yrows, ycols)
    yval = Array(eltype(x), yrows) # 23
    @inbounds begin
        for scol=1:ycols
            fill!(yval, zero(eltype(yval)))
            for jp=s.colptr[scol]:(s.colptr[scol+1]-1)
                srow = s.rowval[jp]
                sval = s.nzval[jp]
                for kp=x.colptr[srow]:(x.colptr[srow+1]-1)
                    xrow = x.rowval[kp] # 191
                    xval = x.nzval[kp]  # 204
                    yinc = xval * sval  # 104
                    # y[xrow,scol] += yinc
                    yval[xrow] += yinc  # 1094
                end
            end
            copy!(y, (scol-1)*yrows+1, yval, 1, yrows) # 7
        end
    end
    return y
end

