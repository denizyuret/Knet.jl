type Logp <: Layer; end

# logp treats the linear output as unnormalized log probabilities and
# adds an offset to each column to make them into normalized log
# probabilities.  The normalization is across the last dimension:
# i.e. sum(exp(y[:,...:,i]))==1 at the output.

function forw(l::Logp, y; o...)
    (nd,nx) = size2(y)
    for j=1:nx
        i1=(j-1)*nd+1
        i2=j*nd
        ymax = typemin(eltype(y))
        for i=i1:i2; y[i] > ymax && (ymax = y[i]); end
        z = zero(Float64)
        for i=i1:i2; z += exp(y[i] -= ymax); end
        logz = log(z)
        for i=i1:i2; y[i] -= logz; end
    end
    return y
end

# Going back logp does not do anything because the constant added does
# not change the derivatives.  There are no parameters to update.
back(l::Logp, dy; o...)=dy

if GPU
forw(l::Logp,y::CudaArray{Float32}; o...)=((nd,nx) = size2(y);ccall((:slogpforw,libkunet),Void,(Cint,Cint,Ptr{Float32}),nd,nx,y); y)
forw(l::Logp,y::CudaArray{Float64}; o...)=((nd,nx) = size2(y);ccall((:dlogpforw,libkunet),Void,(Cint,Cint,Ptr{Float64}),nd,nx,y); y)
end # if GPU

