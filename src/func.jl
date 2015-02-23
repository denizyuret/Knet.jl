export relu, drop, softmaxloss

# ACTIVATION FUNCTION INTERFACE: We use the same name for the
# activation function (relu, sigm, etc.) and its derivative.  That way
# a layer can have a single function field and the user cannot pair a
# function with the wrong derivative.  Let x be the layer input,
# y1=wx+b the linear output, and y2=f(y1) the final output.  The
# activation function takes y1 and overwrites it with y2.  The
# derivative takes dy2, the gradient of the final output, and
# overwrites it with dy1, the gradient of the linear output.  The
# derivative needs to know y2, so it takes an extra argument and can
# be distinguished from the activation function by its method
# signature.  These functions modify their arguments in place and do
# not return anything.  TODO: Implement sigmoid and maxout

relu(l,y)=for i=1:length(y) y[i]<zero(y[i])&&(y[i]=zero(y[i])) end
relu(l,y,dy)=for i=1:length(y) y[i]==zero(y[i])&&(dy[i]=zero(dy[i])) end


# PREPROCESSING FUNCTION INTERFACE: A preprocessing function
# (e.g. dropout) modifies the input x before applying the layer.
# Again, we use the same name for the function and its derivative and
# the helpers.

function drop(l, x)
    if l.dropout > 0
        resize(l, :xdrop, x)
        rand!(l.xdrop)
        drop(x, l.xdrop, l.dropout, 1/(1-l.dropout))
    end
end

function drop(l, x, dx)
    if l.dropout > 0
        drop(dx, l.xdrop, l.dropout, 1/(1-l.dropout))
    end
end

drop(x, xdrop, dropout, scale)=for i=1:length(x) x[i] = (xdrop[i] < dropout ? zero(x[i]) : scale * x[i]) end


# LOSS INTERFACE: A loss function takes y, the network output, and dy,
# the desired output.  These should have the same dimensionality, so
# use 1-of-k encoding for classification outputs.  It overwrites dy
# with the gradient of the output y with respect to the loss function.

function softmaxloss(y, dy)
    yrows,ycols = size(y)
    loss = zero(eltype(y))
    prob = similar(y, yrows)
    for j=1:ycols
        ymax = y[1,j]
        for i=2:yrows y[i,j] > ymax && (ymax = y[i,j]) end
        psum = zero(ymax)
        for i=1:yrows
            yij = y[i,j] - ymax
            prob[i] = exp(yij)
            psum += prob[i]
            dy[i,j] == 1 && (loss += yij)
        end
        loss -= log(psum)
        for i=1:yrows
            prob[i] /= psum
            dy[i,j] = (prob[i] - dy[i,j]) / ycols
        end
    end
    return loss
end

if gpu
    drop(x::CudaArray, xdrop::CudaArray, dropout, scale)=ccall((:drop,libkunet),Void,(Cint,Cmat,Cmat,Cfloat,Cfloat),length(x),x,xdrop,dropout,scale)
    relu(l,y::CudaArray)=ccall((:reluforw,libkunet),Void,(Cint,Cmat),length(y),y)
    relu(l,y::CudaArray,dy::CudaArray)=ccall((:reluback,libkunet),Void,(Cint,Cmat,Cmat),length(dy),y,dy)
    # TODO: This doesn't return the loss, just writes the gradient:
    softmaxloss(y::CudaArray,dy::CudaArray)=ccall((:softback,libkunet),Cfloat,(Cint,Cint,Cmat,Cmat),size(dy,1),size(dy,2),y,dy)
end
