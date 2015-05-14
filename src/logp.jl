type Logp <: Layer; end

# logp treats the linear output as unnormalized log probabilities and
# adds an offset to each column to make them into normalized log
# probabilities:

function forw(l::Logp,y; o...)
    y = initforw(l, y)
    yrows,ycols = size(y)
    for j=1:ycols
        ymax = typemin(eltype(y))
        for i=1:yrows; y[i,j] > ymax && (ymax = y[i,j]); end
        z = zero(eltype(y))
        for i=1:yrows; z += exp((y[i,j] -= ymax)); end
        logz = log(z)
        for i=1:yrows; y[i,j] -= logz; end
    end
    return y
end

function initforw(l::Logp, y)
    (ndims(y) == 2 ? y :
     ndims(y) == 1 ? reshape(y, length(y), 1) :
     reshape(y, int(length(y)/size(y, ndims(y))), size(y, ndims(y))))
end

if GPU
function forw(l::Logp,y::CudaArray; o...)
    y = initforw(l, y)
    ccall((:logpforw,libkunet),Void,(Cint,Cint,Cmat),size(y,1),size(y,2),y)
    return y
end
end # if GPU

# Going back logp does not do anything because the constant added does
# not change the derivatives.  There are no parameters to update.  So
# we will leave these as default.
# back(l::Logp,dy)=dy
# update(l::Logp)=nothing
