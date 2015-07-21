# (c) Deniz Yuret, June 17, 2015
#
# This is an implementation of the averaged perceptron algorithm as described in:
# http://ciml.info/dl/v0_9/ciml-v0_9-ch03.pdf. (however I ended up not using his optimization)
#
# This layer should be paired with a PercLoss layer, i.e. 
#   net=[Perceptron(n), PercLoss()]
#
# It combines the functions of an averaged mmul layer with an averaged bias layer.
# Until I figure out how to add averaging to layers in general, this will have to do.

type Perceptron <: Layer; w; b; x; y; 
    Perceptron(w::KUparam)=new(w)
    Perceptron(w::KUparam,b::KUparam)=new(w,b)
end

Perceptron(n::Integer; bias=true,o...)=(bias ? Perceptron(KUparam(n,0;o...), KUparam(0;o...)) : Perceptron(KUparam(n,0;o...)))

# Except for averaging, forw is mmul followed by bias:

function forw(l::Perceptron, x; predict=false, o...)
    (w, b) = initforw(l, x, predict)
    A_mul_B!(l.y, w, l.x)                      # l.y = w * l.x
    isdefined(l,:b) && biasforw(b, l.y)        # l.y += l.b
    return l.y
end

# back takes dy, the error gradient coming from PercLoss and just records it.
# dy has the same dimensions as the network output y: nclass, ninst
# dy[nc,nx] = -1 if nc was the correct answer for nx and we did not get it.
# dy[nc,nx] = +1 if we answered nc for nx but this was a mistake.
# dy[nc,nx] = 0  otherwise.

function back(l::Perceptron, dy; o...)
    initback(l, dy)
    A_mul_Bt!(diff(l.w), dy, l.x)
    isdefined(l,:b) && biasback(l.db, dy)
end

function update(l::Perceptron; o...)
    update(l.w; o...)
    isdefined(l,:b) && update(l.b; o...)
end

# Allocation

function initback(l::Perceptron, dy)
    @assert issimilar(dy, l.y)
    similar!(l, :dw, l.w0)
    isdefined(l,:b) && similar!(l, :db, l.b0)
end

function initforw(l::Perceptron, x, predict)
    l.x = x
    isempty(l,:w) && firstinit(l,x)
    resize!(l.y, (l.nclass,ccount(x)))
    @assert size(l.w) == (size(l.y, 1), clength(l.x))
    @assert length(l.b) == size(l.y, 1)
    w = (predict && nz(l.w,:average,false) ? l.w.avg : l.w.arr)
    b = (!isdefined(l,:b) ? nothing : predict && nz(l.b,:average,false) ? l.b.avg : l.b.arr)
    return (w, b)
end

function firstinit(l::Perceptron, x)
    l.y = yarray(x, l.nclass)

    similar!(l,:w0,l.y,l.nclass,size(l.x,1); fill=0)
    l.average && similar!(l,:w2,l.w0; fill=0)
    isdefined(l,:b) && similar!(l,:b0,l.y,l.nclass,1; fill=0)
    isdefined(l,:b) && l.average && similar!(l,:b2,l.b0; fill=0)
end

# Our input can be sparse or dense, the parameters or the output are
# always dense.

yarray(x,n)=(isa(x, SparseMatrixCSC) ? Array(eltype(x),(n,ccount(x))) :
             isa(x, Sparse) ? atype(x)(eltype(x),(n,ccount(x))) :
             isa(x, KUsparse) ? KUdense(atype(x),eltype(x),(n,ccount(x))) :
             similar(x, (n, ccount(x))))

parray(x,d)=(isa(x, SparseMatrixCSC) ? Array(eltype(x),d) :
             isa(x, Sparse) ? atype(x)(eltype(x),d) :
             isa(x, KUsparse) ? atype(x)(eltype(x),d) :
             isa(x, KUdense) ? atype(x)(eltype(x),d) :
             similar(x,d))

### DEAD CODE

#    u; u2                       # update count, and the last count when w2 was updated


    # if predict && l.average && (l.u != l.u2)
    #     # we update w1 = w2 - u*w0 during training but not w2
    #     # we only need w2 for prediction
    #     # u2 keeps track of the last time w2 was updated
    #     # if w2 is out of date, we update it here (same with b2):
    #     copy!(l.w2,l.w1); axpy!(length(l.w0), l.u, l.w0, 1, l.w2, 1) # l.w2 = l.u * l.w0 + l.w1
    #     l.bias && (copy!(l.b2,l.b1); axpy!(length(l.b0), l.u, l.b0, 1, l.b2, 1)) # l.b2 = l.u * l.b0 + l.b1
    #     l.u2 = l.u
    #     # @assert maximum(abs(l.w2)) < sqrt(typemax(eltype(l.w2))) # does not work on gpu
    # end

#     l.u = l.u2 = zero(utype(x))
# utype(x)=eltype(x)              # or should we make this Int64 or Float64?


    # update w1 b1 and u?
    # but we lost track of u for averaging, we can use blocks, we can use w2 directly
    # simplest solution we give up on the daume trick

# function update(l::Perceptron; o...)
#     y = convert(Array, l.y)
#     z = convert(Array, l.z)
#     @inbounds for j=1:size(z,2)
#         (cz,cy,ymax,zmax) = (0,0,typemin(eltype(y)),typemin(eltype(z)))
#         @inbounds for i=1:l.nclass
#             z[i,j] > zmax && ((cz,zmax) = (i,z[i,j])) # find the correct answer
#             y[i,j] > ymax && ((cy,ymax) = (i,y[i,j])) # find the model answer
#         end
#         if cz != cy                 # if model answer is not correct
#             if l.bias
#                 l.b0[cz] += one(eltype(l.b0))           # bias for correct answer up
#                 l.b0[cy] -= one(eltype(l.b0))           # bias for model answer down
#                 l.b1[cz] -= l.u
#                 l.b1[cy] += l.u
#             end
#             # The following 4 lines are eat most of the time, so define efficient function
#             addx!(one(eltype(l.x)), l.x, j, l.w0, cz)    # l.w0[cz,:] += l.x[:,j]' # weights for correct answer +x
#             addx!(-one(eltype(l.x)), l.x, j, l.w0, cy)   # l.w0[cy,:] -= l.x[:,j]' # weights for model answer -x
#             addx!(-l.u, l.x, j, l.w1, cz)                # l.w1[cz,:] -= l.u * l.x[:,j]'
#             addx!(l.u, l.x, j, l.w1, cy)                 # l.w1[cy,:] += l.u * l.x[:,j]'
#         end
#         l.u += 1            # increment counter regardless of update
#     end
# end

# function addx!(a::Number, x::SparseMatrixCSC, xcol::Integer, w::AbstractMatrix, wrow::Integer)
#     @assert size(x, 1) == size(w, 2)
#     a = convert(eltype(x), a)
#     i1 = x.colptr[xcol]
#     i2 = x.colptr[xcol+1]-1
#     @inbounds @simd for i=i1:i2
#         wcol = x.rowval[i]
#         w[wrow,wcol] += a * x.nzval[i]
#     end
#     return w
# end

# function addx!(a::Number, x::Array, xcol::Integer, w::AbstractArray, wrow::Integer)
#     @assert size(x, 1) == size(w, 2)
#     a = convert(eltype(x), a)
#     @inbounds @simd for i=1:size(x,1)
#         w[wrow,i] += a * x[i,xcol]
#     end
#     return w
# end

    # u; u2                       # update count, and the last count when w2 was updated
    # w0; w1; w2; dw              # perceptron weights: raw, delta, summed, gradient: l.w2 = l.u * l.w0 + l.w1
    # b0; b1; b2; db              # bias weights: raw, delta, summed, gradient
