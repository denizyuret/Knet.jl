type Logp <: Layer; end

# logp treats the linear output as unnormalized log probabilities and
# adds an offset to each column to make them into normalized log
# probabilities.  The normalization is across the last dimension:
# i.e. sum(exp(y[:,...:,i]))==1 at the output.

function forw(l::Logp,y; o...)
    nd = ndims(y)
    sz = size(y, nd)
    st = stride(y, nd)
    for j=1:sz
        i1=(j-1)*st+1
        i2=j*st
        ymax = typemin(eltype(y))
        for i=i1:i2; y[i] > ymax && (ymax = y[i]); end
        z = zero(eltype(y))
        for i=i1:i2; z += exp(y[i] -= ymax); end
        logz = log(z)
        for i=i1:i2; y[i] -= logz; end
    end
    return y
end

if GPU
function forw(l::Logp,y::CudaArray; o...)
    y2 = size(y, ndims(y))
    y1 = div(length(y), y2)
    ccall((:logpforw,libkunet),Void,(Cint,Cint,Cmat),y1, y2, y)
    return y
end
end # if GPU

# Going back logp does not do anything because the constant added does
# not change the derivatives.  There are no parameters to update.  So
# we will leave these as default.
# back(l::Logp,dy)=dy
# update(l::Logp)=nothing
