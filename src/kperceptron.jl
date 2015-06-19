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
#
# TODO: understand why adding bias screws up rbf and has no effect on poly?

type KPerceptron <: Layer; n; k; p; s; x; y; z; w0; b0; w1; b1; w2; b2; u; 
    KPerceptron(nclass,kernel,params)=new(nclass,kernel,params)
end

function forw(l::KPerceptron, x; predict=false, o...)
    initforw(l, x, predict)
    l.y = (predict ? l.w2 : l.w0) * l.k(l.s, l.x, l.p)
#    l.y = (predict ? 
#           l.w2 * K : # use averaged weights for prediction
#           l.w0 * K)  # use regular weights for training
#           l.w2 * K .+ l.b2 : # use averaged weights for prediction
#           l.w0 * K .+ l.b0)  # use regular weights for training
end

function update(l::KPerceptron; o...)
    l.w2 = l.b2 = nothing   # make sure these are reset when w0,w1,u changes
    for j=1:size(l.z,2)
        (cz,cy,ymax,zmax) = (0,0,typemin(eltype(l.y)),typemin(eltype(l.z)))
        for i=1:l.n
            l.z[i,j] > zmax && ((cz,zmax) = (i,l.z[i,j])) # find the correct answer
            l.y[i,j] > ymax && ((cy,ymax) = (i,l.y[i,j])) # find the model answer
        end
        if cz != cy                 # if model answer is not correct
            l.b0[cz] += 1           # bias for correct answer up
            l.b0[cy] -= 1           # bias for model answer down
            l.b1[cz] += l.u         # w2,b2 keep the summed weights
            l.b1[cy] -= l.u         # w1,b1 keep the diff between l.u*(w0,b0) and the (w2,b2)
            # l.x[:,j] becomes a new support vector
            l.s = [l.s l.x[:,j]]          # TODO: use snew wnew to keep reallocations small
            w = zeros(eltype(l.w0),size(l.w0,1),1)
            w[cz] = 1; w[cy] = -1
            l.w0 = [l.w0 w]
            w[cz] = l.u; w[cy] = -l.u
            l.w1 = [l.w1 w]
        end
        l.u += 1            # increment counter regardless of update
    end
end

# Some common kernels
kpoly(s, x, p)=((s' * x + p[1]) .^ p[2])
krbf(s, x, p)=exp(-p[1] * broadcast(+, sum(x.^2,1), broadcast(+, sum(s.^2,1)', -2*(s' * x))))

function initforw(l::KPerceptron, x, predict)
    l.x = x
    if !isdefined(l,:s)
        @assert KUnet.Atype==Array "KPerceptron cannot handle CudaArray yet"
        l.s = similar(l.x, size(l.x,1), 0)  # matches l.x in sparsity and orientation
        l.w0 = zeros(eltype(l.x), l.n, 0)   # w,b always full
        l.b0 = zeros(eltype(l.x), l.n, 1)
        l.w1 = zeros(eltype(l.x), l.n, 0)
        l.b1 = zeros(eltype(l.x), l.n, 1)
        l.w2 = nothing
        l.b2 = nothing
        l.u = 0
    end
    @assert size(l.x, 1) == size(l.s, 1)
    @assert size(l.s, 2) == size(l.w0, 2)
    if predict && (l.w2 == nothing)
        l.w2 = l.u * l.w0 - l.w1
        l.b2 = l.u * l.b0 - l.b1
        # making sure we don't get overflow
        @assert maximum(abs(l.w2)) < sqrt(typemax(eltype(l.w2)))
    end
end

function back(l::KPerceptron, z; returndx=false, o...)
    returndx && error("KPerceptron does not know how to return dx")
    l.z = z   # just record the correct answers in l.z
end

