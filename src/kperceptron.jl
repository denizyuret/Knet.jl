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
    KPerceptron(nclass,kernel,params)=new(nclass,kernel,params)
end

# Some common kernels
kpoly(s, x, p)=((s' * x + p[1]) .^ p[2])
krbf(s, x, p)=exp(-p[1] * broadcast(+, sum(x.^2,1), broadcast(+, sum(s.^2,1)', -2*(s' * x))))

# This is 4 times slower than ((s' * x + p[1]) .^ p[2])
function kpoly2(s::SparseMatrixCSC, x::SparseMatrixCSC, p::Vector)
    k = Array(eltype(x), size(s, 2), size(x, 2))
    @inbounds for j=1:size(k,2)
        @inbounds for i=1:size(k,1)
            # k[i,j] = (dot(s[:,i], x[:,j]) + p[1]) ^ p[2]
            s1 = s.colptr[i]; s2 = s.colptr[i+1]
            x1 = x.colptr[j]; x2 = x.colptr[j+1]
            sx = zero(eltype(x))
            while (s1<s2 && x1<x2)
                sr = s.rowval[s1]
                xr = x.rowval[x1]
                if sr < xr
                    s1 += 1
                elseif sr > xr
                    x1 += 1
                else            # sr==xr
                    sx += s.nzval[s1] * x.nzval[x1]
                    s1 += 1; x1 += 1;
                end
            end
            k[i,j] = (sx + p[1])^p[2]
        end
    end
    return k
end

function forw(l::KPerceptron, x; predict=false, o...)
    initforw(l, x, predict)
    l.y = (predict ? l.w2 : l.w0) * l.k(l.s, l.x, l.p)
end

function update(l::KPerceptron; o...)
    l.w2 = nothing   # make sure these are reset when w0,w1,u changes
    wnew = zeros(eltype(l.w0),size(l.w0,1),1)
    snew = similar(l.x, size(l.x,1), 0)
    for j=1:size(l.z,2)
        (cz,cy,ymax,zmax) = (0,0,typemin(eltype(l.y)),typemin(eltype(l.z)))
        for i=1:l.n
            l.z[i,j] > zmax && ((cz,zmax) = (i,l.z[i,j])) # find the correct answer
            l.y[i,j] > ymax && ((cy,ymax) = (i,l.y[i,j])) # find the model answer
        end
        if cz != cy # if model answer is not correct l.x[:,j] becomes a new support vector
            snew = [snew l.x[:,j]]
            wnew[cz] = 1; wnew[cy] = -1
            l.w0 = [l.w0 wnew]
            wnew[cz] = l.u; wnew[cy] = -l.u
            l.w1 = [l.w1 wnew]
            wnew[cz] = 0; wnew[cy] = 0
        end
        l.u += 1            # increment counter regardless of update
    end
    l.s = [l.s snew]
end

function initforw(l::KPerceptron, x, predict)
    l.x = x
    if !isdefined(l,:s)
        @assert KUnet.Atype==Array "KPerceptron cannot handle CudaArray yet"
        l.s = similar(l.x, size(l.x,1), 0)  # matches l.x in sparsity and orientation
        l.w0 = zeros(eltype(l.x), l.n, 0)   # w,b always full
        l.w1 = zeros(eltype(l.x), l.n, 0)
        l.w2 = nothing
        l.u = 0
    end
    @assert size(l.x, 1) == size(l.s, 1)
    @assert size(l.s, 2) == size(l.w0, 2)
    if predict && (l.w2 == nothing)
        l.w2 = l.u * l.w0 - l.w1
        # making sure we don't get overflow
        @assert maximum(abs(l.w2)) < sqrt(typemax(eltype(l.w2)))
    end
end

function back(l::KPerceptron, z; returndx=false, o...)
    returndx && error("KPerceptron does not know how to return dx")
    l.z = z   # just record the correct answers in l.z
end

