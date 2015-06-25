# (c) Deniz Yuret, June 17, 2015
# This is a standalone (single layer) implementation of the averaged
# perceptron algorithm as described in:
# http://ciml.info/dl/v0_9/ciml-v0_9-ch03.pdf.

type Perceptron <: Layer; n; b; x; y; z; w0; b0; w1; b1; w2; b2; u; u2;
    Perceptron(nclass;bias=true)=new(nclass,bias)
end

function forw(l::Perceptron, x; predict=false, o...)
    initforw(l, x, predict)
    w = (predict ? l.w2 : l.w0)                # w2 averaged, w0 regular weights
    A_mul_B!(l.y, w, l.x)                      # l.y = w * l.x
    l.b && broadcast!(+, l.y, l.y, (predict ? l.b2 : l.b0)) # l.y += b
    return l.y
end

function back(l::Perceptron, z; returndx=false, o...)
    @assert size(z) == size(l.y)
    returndx && error("Perceptron does not know how to return dx")
    l.z = z   # just record the correct answers in l.z
end

function update(l::Perceptron; o...)
    y = isongpu(l.y) ? to_host(l.y) : l.y
    z = isongpu(l.z) ? to_host(l.z) : l.z
    @inbounds for j=1:size(z,2)
        (cz,cy,ymax,zmax) = (0,0,typemin(eltype(y)),typemin(eltype(z)))
        @inbounds for i=1:l.n
            z[i,j] > zmax && ((cz,zmax) = (i,z[i,j])) # find the correct answer
            y[i,j] > ymax && ((cy,ymax) = (i,y[i,j])) # find the model answer
        end
        if cz != cy                 # if model answer is not correct
            if l.b
                l.b0[cz] += one(eltype(l.b0))           # bias for correct answer up
                l.b0[cy] -= one(eltype(l.b0))           # bias for model answer down
                l.b1[cz] -= l.u
                l.b1[cy] += l.u
            end
            # The following 4 lines are eat most of the time, so define efficient function
            addx!(one(eltype(l.x)), l.x, j, l.w0, cz)    # l.w0[cz,:] += l.x[:,j]' # weights for correct answer +x
            addx!(-one(eltype(l.x)), l.x, j, l.w0, cy)   # l.w0[cy,:] -= l.x[:,j]' # weights for model answer -x
            addx!(-l.u, l.x, j, l.w1, cz)                # l.w1[cz,:] -= l.u * l.x[:,j]'
            addx!(l.u, l.x, j, l.w1, cy)                 # l.w1[cy,:] += l.u * l.x[:,j]'
        end
        l.u += 1            # increment counter regardless of update
    end
end

function addx!(a::Number, x::SparseMatrixCSC, xcol::Integer, w::AbstractMatrix, wrow::Integer)
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
    if !isdefined(l,:u)
        l.u = l.u2 = zero(eltype(l.x))
        l.y = (isongpu(x)?CudaArray:Array)(eltype(x),l.n,size(x,2))
        similar!(l,:w0,l.y,l.n,size(l.x,1); fill=0)
        similar!(l,:w1,l.w0; fill=0)
        similar!(l,:w2,l.w0; fill=0)
        if l.b
            similar!(l,:b0,l.y,l.n,1; fill=0)
            similar!(l,:b1,l.b0; fill=0)
            similar!(l,:b2,l.b0; fill=0)
        end
    end
    similar!(l,:y,l.w0,l.n,size(x,2))
    @assert size(l.x, 1) == size(l.w0, 2)
    @assert size(l.y, 1) == size(l.w0, 1)
    if predict && (l.u != l.u2)
        copy!(l.w2,l.w1); axpy!(length(l.w0), l.u, l.w0, 1, l.w2, 1) # l.w2 = l.u * l.w0 + l.w1
        l.b && (copy!(l.b2,l.b1); axpy!(length(l.b0), l.u, l.b0, 1, l.b2, 1)) # l.b2 = l.u * l.b0 + l.b1
        l.u2 = l.u
        # @assert maximum(abs(l.w2)) < sqrt(typemax(eltype(l.w2))) # does not work on gpu
    end
end

