# (c) Deniz Yuret, June 17, 2015
# This is a standalone (single layer) implementation of the averaged
# perceptron algorithm as described in:
# http://ciml.info/dl/v0_9/ciml-v0_9-ch03.pdf.

type Perceptron <: Layer; n; x; y; z; w0; b0; w1; b1; w2; b2; u; Perceptron(nclass)=new(nclass); end

function copy(l::Perceptron)
    @assert isa(l.w0, KUnet.atype()) "Perceptron cannot change array type yet"
    c = Perceptron(l.n)
    for n in names(c); c.(n) = copy(l.(n)); end
    return c
end

function forw(l::Perceptron, x; predict=false, o...)
    initforw(l, x, predict)
    l.y = (predict ? 
           l.w2' * l.x .+ l.b2 : # use averaged weights for prediction
           l.w0' * l.x .+ l.b0)  # use regular weights for training
end

function back(l::Perceptron, z; returndx=false, o...)
    returndx && error("Perceptron does not know how to return dx")
    l.z = z   # just record the correct answers in l.z
end

function update(l::Perceptron; o...)
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
            l.b1[cz] += l.u
            l.b1[cy] -= l.u
            # The following 4 lines are eat most of the time
            # Keeping w same direction as x helps a bit in spite of the transpose in forw
            # Using for loops does not help (maybe iterating over nz for sparse might)
            l.w0[:,cz] += l.x[:,j] # weights for correct answer +x
            l.w0[:,cy] -= l.x[:,j] # weights for model answer -x
            l.w1[:,cz] += l.u * l.x[:,j]
            l.w1[:,cy] -= l.u * l.x[:,j]
        end
        l.u += 1            # increment counter regardless of update
    end
end

function initforw(l::Perceptron, x, predict)
    l.x = x
    if !isdefined(l,:w0)
        # similar!(l,:w0,l.x,(size(l.x,1),l.n); fill=0)
        # TODO: This is to try the effect of full arrays, it does not do gpu yet:
        l.w0 = zeros(eltype(l.x), size(l.x,1), l.n)
        similar!(l,:b0,l.w0,(l.n,1); fill=0)
        similar!(l,:w1,l.w0; fill=0)
        similar!(l,:b1,l.b0; fill=0)
        l.w2 = nothing
        l.b2 = nothing
        l.u = 0
    end
    @assert size(l.x, 1) == size(l.w0, 1)
    if predict && (l.w2 == nothing)
        l.w2 = l.u * l.w0 - l.w1
        l.b2 = l.u * l.b0 - l.b1
        @assert maximum(abs(l.w2)) < sqrt(typemax(eltype(l.w2)))
    end
end

