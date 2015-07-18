type Soft <: Layer; y; Soft()=new(); end

# Soft treats its input y as unnormalized log probabilities and
# applies the softmax function to overwrite it with normalized
# probabilities.  The normalization is across the last dimension:
# i.e. sum(y[:,...,:,i])==1 at the output. 
# (CUDNN_SOFTMAX_MODE_INSTANCE)


forw(l::Soft,y::KUdense; o...)=(softforw(y.arr;o...); l.y=y)
back(l::Soft,dy::KUdense; o...)=(softback(l.y.arr,dy.arr;o...); dy)

function softforw(y::Array; o...)
    (st,nx) = size2(y)
    for j=1:nx
        i1=(j-1)*st+1
        i2=j*st
        ymax = typemin(eltype(y))
        ysum = zero(Float64)
        for i=i1:i2; y[i] > ymax && (ymax = y[i]); end
        for i=i1:i2; ysum += (y[i]=exp(y[i] - ymax)); end
        for i=i1:i2; y[i] /= ysum; end
    end
    return y
end

function softback(y::Array,dy::Array; returndx=true, o...)
    @assert issimilar(dy,y)
    returndx || return
    (st,nx) = size2(dy)
    for j=1:nx
        i1=(j-1)*st+1
        i2=j*st
        sumydy = zero(Float64)
        for i=i1:i2; sumydy += y[i] * dy[i]; end
        for i=i1:i2; dy[i] = y[i] * (dy[i] - sumydy); end
    end
    return dy
end


if GPU
# TODO: what happened to the buggy 0.5 factor?
softforw(y::CudaArray; o...)=(cudnnSoftmaxForward(y); y)
softback(y::CudaArray,dy::CudaArray; returndx=true, o...)=(@assert issimilar(dy,y); returndx && cudnnSoftmaxBackward(y, dy); dy)
end # if GPU
