# LOSS INTERFACE: A loss function takes y, the network output, and dy,
# the desired output.  These should have the same dimensionality, so
# use 1-of-k encoding for classification outputs.  It overwrites dy
# with the gradient of the output y with respect to the loss function.

# TODO: generalize to N-D

function softmaxloss(y, dy)
    # softmaxloss assumes that y consists of unnormalized logp.
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
    # Similar to softmaxloss, except we assume y is normalized logp.
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
    # TODO: This doesn't return the loss, just writes the gradient:
    softmaxloss(y::CudaArray,dy::CudaArray)=ccall((:softback,libkunet),Cfloat,(Cint,Cint,Cmat,Cmat),size(dy,1),size(dy,2),y,dy)
    logploss(y::CudaArray,dy::CudaArray)=ccall((:logploss,libkunet),Cfloat,(Cint,Cint,Cmat,Cmat),size(dy,1),size(dy,2),y,dy)
end
