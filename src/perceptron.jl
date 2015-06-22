# (c) Deniz Yuret, June 17, 2015
# This is a standalone (single layer) implementation of the averaged
# perceptron algorithm as described in:
# http://ciml.info/dl/v0_9/ciml-v0_9-ch03.pdf.

type Perceptron <: Layer; n; x; y; z; w0; b0; w1; b1; w2; b2; u; Perceptron(nclass)=new(nclass); end

function forw(l::Perceptron, x; predict=false, o...)
    initforw(l, x, predict)
    l.y = (predict ? 
          l.w2 * l.x .+ l.b2 : # use averaged weights for prediction
          l.w0 * l.x .+ l.b0)  # use regular weights for training
end

function back(l::Perceptron, z; returndx=false, o...)
    @assert size(z) == size(l.y)
    returndx && error("Perceptron does not know how to return dx")
    l.z = z   # just record the correct answers in l.z
end

function update(l::Perceptron; o...)
    l.w2 = l.b2 = nothing   # make sure these are reset when w0,w1,u changes
    @inbounds for j=1:size(l.z,2)
        (cz,cy,ymax,zmax) = (0,0,typemin(eltype(l.y)),typemin(eltype(l.z)))
        @inbounds for i=1:l.n
            l.z[i,j] > zmax && ((cz,zmax) = (i,l.z[i,j])) # find the correct answer
            l.y[i,j] > ymax && ((cy,ymax) = (i,l.y[i,j])) # find the model answer
        end
        if cz != cy                 # if model answer is not correct
            l.b0[cz] += one(eltype(l.b0))           # bias for correct answer up
            l.b0[cy] -= one(eltype(l.b0))           # bias for model answer down
            l.b1[cz] += l.u
            l.b1[cy] -= l.u
            # The following 4 lines are eat most of the time, so define efficient function
            addx!(one(eltype(l.x)), l.x, j, l.w0, cz)    # l.w0[cz,:] += l.x[:,j]' # weights for correct answer +x
            addx!(-one(eltype(l.x)), l.x, j, l.w0, cy)   # l.w0[cy,:] -= l.x[:,j]' # weights for model answer -x
            addx!(l.u, l.x, j, l.w1, cz)  # l.w1[cz,:] += l.u * l.x[:,j]'
            addx!(-l.u, l.x, j, l.w1, cy) # l.w1[cy,:] -= l.u * l.x[:,j]'
        end
        l.u += 1            # increment counter regardless of update
    end
end

function addx!(a::Number, x::SparseMatrixCSC, xcol::Integer, w::AbstractArray, wrow::Integer)
    @assert size(x, 1) == size(w, 2)
    a = convert(eltype(x), a)
    i1 = x.colptr[xcol]
    i2 = x.colptr[xcol+1]-1
    @inbounds @simd for i=i1:i2
        wcol = x.rowval[i]
        w[wrow,wcol] += a * x.nzval[i]
    end
    return w
end

function addx!(a::Number, x::Array, xcol::Integer, w::AbstractArray, wrow::Integer)
    @assert size(x, 1) == size(w, 2)
    a = convert(eltype(x), a)
    @inbounds @simd for i=1:size(x,1)
        w[wrow,i] += a * x[i,xcol]
    end
    return w
end

function initforw(l::Perceptron, x, predict)
    l.x = x
    if !isdefined(l,:w0)
        l.w0 = zeros(eltype(l.x), l.n, size(l.x,1))
        similar!(l,:b0,l.w0,(l.n,1); fill=0)
        similar!(l,:w1,l.w0; fill=0)
        similar!(l,:b1,l.b0; fill=0)
        l.w2 = nothing
        l.b2 = nothing
        l.u = zero(eltype(l.w0))
    end
    @assert size(l.x, 1) == size(l.w0, 2)
    if predict && (l.w2 == nothing)
        l.w2 = l.u * l.w0 - l.w1
        l.b2 = l.u * l.b0 - l.b1
        @assert maximum(abs(l.w2)) < sqrt(typemax(eltype(l.w2)))
    end
end

