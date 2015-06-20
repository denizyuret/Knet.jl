# Perceptron loss function. A multiclass perceptron can be constructed
# using an Mmul layer followed by PercLoss.

type PercLoss <: LossLayer; y; PercLoss()=new(); end
# copy(l::PercLoss;o...)=PercLoss()

# Going forward Mmul computes y=w*x and PercLoss simply records the
# output y.  size(w)=(nc,nd) where nc is the number of classes and nd
# is the number of x dimensions (i.e. features).  size(x)=(nd,nx)
# where nd is the number of features and nx is the batch size.  This
# gives us size(y)=(nc,nx) where the highest entry in each column of y
# indicates the predicted class.

forw(l::PercLoss, y; o...)=(l.y=y)

# Going back we get a z matrix with size(z)=(nc,nx) where the correct
# answer is marked with the maximum entry in each column.
# For a given column with input x, if cz is the correct answer and cy
# is the predicted answer, the multiclass perceptron update rule is:
#
# w[cz,:] += x;  w[cy,:] -= x
# 
# Note that there is no update if cz==cy.
# 
# The mmul updates are:
# dw = dy*x'
# dx = w'*dy
# 
# So the perceptron update will be performed by mmul if we pass a dy
# matrix back where in each column we have all zeros if the predicted
# answer is correct, otherwise the correct answer is marked with -1
# and the predicted answer is marked with a +1.  Think of this as the
# gradient of the loss, i.e. going in this direction will increase the
# loss.  We will overwrite the z matrix if dy not specified.

function back(l::PercLoss, z, dy=z; returndx=true, o...)
#    @assert issimilar(l.y,z)
#    @assert issimilar(l.y,dy)
    @assert size(l.y)==size(z)==size(dy)
    returndx || return
    (nc,nx) = size2(l.y)
    for j=1:nx
        (cz,cy,ymax,zmax) = (0,0,typemin(eltype(l.y)),typemin(eltype(z)))
        i1=(j-1)*nc+1; i2=j*nc
        for i=i1:i2
            l.y[i] > ymax && ((cy,ymax) = (i,l.y[i]))
            z[i] > zmax && ((cz,zmax) = (i,z[i]))
            dy[i] = zero(eltype(dy))
        end
        (cz != cy) && (dy[cz] = -one(eltype(dy)); dy[cy] = one(eltype(dy)))
    end
    return dy
end

# This update can be seen as the gradient of a perceptron loss
# function Sum(-y[I]+y[J]) where I are the indices for the correct
# answers, and J are the indices for predicted answers.

function loss(l::PercLoss, z, y=l.y)
    @assert issimilar(z,y)
    (nc,nx) = size2(y)
    cost = zero(Float64)
    for j=1:nx
        (cz,cy,ymax,zmax) = (0,0,typemin(eltype(y)),typemin(eltype(y)))
        i1=(j-1)*nc+1; i2=j*nc
        for i=i1:i2
            y[i] > ymax && ((cy,ymax) = (i,y[i]))
            z[i] > zmax && ((cz,zmax) = (i,z[i]))
        end
        (cz != cy) && (cost += y[cy]; cost -= y[cz])
    end
    return cost/nx
end

if GPU

loss(l::PercLoss, z::CudaArray)=loss(l, to_host(z), to_host(l.y))

function back(l::PercLoss, z::CudaArray{Float32}, dy=z; returndx=true, o...)
    @assert issimilar(z, l.y)
    returndx || return
    (nd,nx) = size2(z)
    ccall((:percloss32,libkunet),Void,(Cint,Cint,Ptr{Float32},Ptr{Float32},Ptr{Float32}),nd,nx,l.y,z,dy)
    return z;
end

function back(l::PercLoss, z::CudaArray{Float64}, dy=z; returndx=true, o...)
    @assert issimilar(z, l.y)
    returndx || return
    (nd,nx) = size2(z)
    ccall((:percloss64,libkunet),Void,(Cint,Cint,Ptr{Float64},Ptr{Float64},Ptr{Float64}),nd,nx,l.y,z,dy)
    return z;
end
end # if GPU
