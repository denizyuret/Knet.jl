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

# relu implements the rectified linear function

relu(l::Layer,y)=for i=1:length(y); (y[i]<zero(y[i]))&&(y[i]=zero(y[i])) end
relu(l::Layer,y,dy)=for i=1:length(y); (y[i]==zero(y[i]))&&(dy[i]=zero(dy[i])) end

# logp treats the linear output as unnormalized log probabilities and
# adds an offset to each column to make them into normalized log
# probabilities:

function logp(l::Layer,y)
    yrows,ycols = size(y)
    for j=1:ycols
        ymax = typemin(eltype(y))
        for i=1:yrows; y[i,j] > ymax && (ymax = y[i,j]); end
        z = zero(eltype(y))
        for i=1:yrows; z += exp((y[i,j] -= ymax)); end
        logz = log(z)
        for i=1:yrows; y[i,j] -= logz; end
    end
end

# Going back logp does not do anything because the constant added does
# not change the derivatives.
logp(l::Layer,y,dy)=nothing

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
        for i=2:yrows 
	    y[i,j] > ymax && (ymax = y[i,j]) 
	end
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
    return -loss
end

function logploss(y, dy)
    # Similar to softmaxloss, except we assume y is normalized logp
    yrows,ycols = size(y)
    loss = zero(eltype(y))
    for j=1:ycols
        for i=1:yrows
            dy[i,j] == 1 && (loss += y[i,j])
            dy[i,j] = (exp(y[i,j]) - dy[i,j]) / ycols
        end
    end
    return loss
end

if usegpu
    drop(x::CudaArray, xdrop::CudaArray, dropout, scale)=ccall((:drop,libkunet),Void,(Cint,Cmat,Cmat,Cfloat,Cfloat),length(x),x,xdrop,dropout,scale)
    relu(l::Layer,y::CudaArray)=ccall((:reluforw,libkunet),Void,(Cint,Cmat),length(y),y)
    relu(l::Layer,y::CudaArray,dy::CudaArray)=ccall((:reluback,libkunet),Void,(Cint,Cmat,Cmat),length(dy),y,dy)
    logp(l::Layer,y::CudaArray)=ccall((:logpforw,libkunet),Void,(Cint,Cint,Cmat),size(y,1),size(y,2),y)
    # TODO: This doesn't return the loss, just writes the gradient:
    softmaxloss(y::CudaArray,dy::CudaArray)=ccall((:softback,libkunet),Cfloat,(Cint,Cint,Cmat,Cmat),size(dy,1),size(dy,2),y,dy)
    logploss(y::CudaArray,dy::CudaArray)=ccall((:logploss,libkunet),Cfloat,(Cint,Cint,Cmat,Cmat),size(dy,1),size(dy,2),y,dy)
end
