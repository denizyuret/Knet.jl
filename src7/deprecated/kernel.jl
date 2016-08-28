# Kernel Op
# Input: x[nd,nx]
# Support vector matrix: s[nd,ns]
# Kernel function: K[ns,nx] = kernel(s,x)  -- a new representation for x
# Output: y[nc,nx] = w[nc,ns] * K[ns,nx]  -- like the mmul layer
# Initially: s[nd,0], w[nc,0]
#
# A subtype of kernel needs to define:
# Fields: s, w, v, x, y, dy
# Functions: kernel
# Initialize: w[nc,0]

abstract Kernel <: Op

# The forw function is just mmul applied to a transformed input:

function forw(l::Kernel, x; train=true, o...)
    initforw(l, x)
    # kernel fn is defined by subtypes of kernel and returns the kernel matrix K
    l.y = (train ? l.v : l.w) * kernel(l)
end

function back(l::Kernel, dy; returndx=false, o...)
    @assert issimilar(dy, l.y)
    l.dy = dy  # we assume this is from PercLoss: -1 for correct answer, +1 for predicted answer
    returndx && error("Kernel layers do not know how to return dx")
end

function update!(l::Kernel; o...)
    (yrows, ycols) = size(l.dy)
    (xrows, xcols) = size(l.x)
    @assert ycols == xcols
    snew = similar(l.x, (xrows, 0))
    wnew = similar(l.w, (yrows, 0))
    for j=1:ycols
        for i=1:yrows
            if l.dy[i,j] != 0
                snew=[snew l.x[:,j]] ### BIG ALLOC
                wnew=[wnew (-l.dy[:,j])]
                break
            end
        end
    end
    l.w += l.v
    l.w = [l.w wnew]
    l.v = [l.v wnew]
    l.s = [l.s snew]
end

function initforw(l::Kernel, x)
    (xrows, xcols) = size2(x)
    l.x = (size(x)==(xrows,xcols) ? x : reshape(x, xrows, xcols))
    isdefined(l,:s) || (l.s=similar(x, (xrows, 0)))
    (srows, scols) = size(l.s)
    @assert srows == xrows
    isdefined(l,:w) || (l.w=Array(eltype(x), l.n, 0))
    (wrows, wcols) = size(l.w)
    @assert scols == wcols
    # l.v is the perceptron weights, l.w is the averaged (summed) weights
    isdefined(l,:v) || (@assert isempty(l.w); l.v = l.w)
end

# TODO: explain back and update
# TODO: what if we froze the SV and used regular updates (copy sv into mmul)?

# using Base.LinAlg.BLAS: gemm!

# OK forget about cuda, just do Array and SparseArray for now.
# Use generic code, forget about efficiency.

# DONE: forw will not work for the first batch of x when sv is empty!
# Can we multiply zero dim matrices? yes.

# No need to alloc l.y, no in-place ops for sparse
# similar!(l, :y, l.sv, (scols, xcols))

# Careful: poly may turn a sparse array into dense if b!=0.
# If b==0 we can just go over nonzero entries with sparse.

# if GPU
# poly(y::AbstractCudaArray{Float32}, a, b, d)=(ccall((poly32,libkunet),Void,(Ptr{Float32},Float32,Float32,Float32),y,a,b,d);y)
# poly(y::AbstractCudaArray{Float64}, a, b, d)=(ccall((poly64,libkunet),Void,(Ptr{Float64},Float64,Float64,Float64),y,a,b,d);y)
# # TODO: test sparse on cpu/gpu
# # TODO: implement poly32/64 in cuda
# end

    # back: Kmul passes back its own dy, nonzero columns indicate mistakes
    # We do nothing here except record it
    # Actual creation of support vectors done in update!()
    # TODO: we could compute dx like we do in mmul?

# TODO: back
# q: what is dy? - use the same dy as Kmul
# a: we need indices of x vectors with wrong answers to turn into sv.
# a: that means dy not the same size as y?
# a: argue for integrating beta here? pro:more like other layers, con:repeat code for other kernels
# q: what does back do? nothing? (store incoming info?)
# q: what does update do? (add new sv?)
# q: what do we pass back? (nothing?)
# a: does primal perceptron compute dx?

# TODO: worry about efficiency of concat later (consider making sv a list?)
# See this for efficiency: https://groups.google.com/forum/#!topic/julia-users/B4OUYPFM5L8
# TODO: check if concat works for cpu/gpu, sparse/full
# TODO: gpu version

# TODO: fix the starting empty sv problem
# TODO: implement Kmul
# TODO: implement Rbfk
# TODO: see about code repetition, Only forward will be different, back and update the same.
# we can define abstract Kernel to avoid repetition.
# if we define Kernel, we can also have the beta mult in there!

# TODO: uniq sv, cotter13
