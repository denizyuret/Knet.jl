using CUDArt

reluforw(y::CudaArray)=ccall((:reluforw,libkunet),Void,(Cint,Cmat),length(y),y)
reluback(y::CudaArray,dy::CudaArray)=ccall((:reluback,libkunet),Void,(Cint,Cmat,Cmat),length(y),y,dy)
softback(y::CudaArray,dy::CudaArray)=ccall((:softback,libkunet),Void,(Cint,Cint,Cmat,Cmat),size(y,1),size(y,2),y,dy)

softforw(y)=y
noop(l,x)=x
dropforw(l,x)=error("dropforw not implemented yet")
dropback(l,dx)=error("dropback not implemented yet")

function reluforw(y)
    for i=1:length(y)
        if (y[i] < 0)
            y[i] = 0
        end
    end
end

function reluback(y, dy)
    for i=1:length(dy)
        if (y[i] <= 0)
            dy[i] = 0
        end
    end
end

function softback(y, dy)
    # we do softmax here instead of in forw
    # overwriting y from unnormalized log probabilities to normalized probabilities
    # NumericExtensions.softmax!(y,y,1) allocates unnecessary memory
    # dy is a 0-1 matrix of correct answers
    # will overwrite it with the gradient
    # TODO: is this a good interface?
    # TODO: other types of final layers, losses?

    for j=1:size(y,2)
        ymax = y[1,j]
        for i=2:size(y,1)
            if (y[i,j] > ymax)
                ymax = y[i,j]
            end
        end
        ysum = zero(ymax)
        for i=1:size(y,1)
            y[i,j] = exp(y[i,j] - ymax)
            ysum += y[i,j]
        end
        for i=1:size(y,1)
            y[i,j] /= ysum
            dy[i,j] = (y[i,j] - dy[i,j]) / size(y,2)
        end
    end
end


