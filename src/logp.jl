type Logp <: Layer; y; Logp()=new(); end

# logp treats the linear output as unnormalized log probabilities and
# adds an offset to each column to make them into normalized log
# probabilities.  The normalization is across the last dimension:
# i.e. sum(exp(y[:,...:,i]))==1 at the output.

forw(l::Logp, y::KUdense; o...)=(logpforw(y.arr); l.y=y)

# If the output is normalized logp connected to LogpLoss we should get
# dy=dx since y=x+const.
back(l::Logp, dy; o...)=dy

function logpforw(y::Array)
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
end

if GPU
logpforw(y::CudaArray{Float32}; o...)=((nd,nx) = size2(y);ccall((:logpforw32,libkunet),Void,(Cint,Cint,Ptr{Float32}),nd,nx,y))
logpforw(y::CudaArray{Float64}; o...)=((nd,nx) = size2(y);ccall((:logpforw64,libkunet),Void,(Cint,Cint,Ptr{Float64}),nd,nx,y))
end # if GPU


# We don't need back!
# For general loss functions:
# z = Σj exp(xj)
# yi = xi - logz
# dJ/dxi = Σj dJ/dyj dyj/dxi
# dyj/dxi = [i==j] - (1/z)exp(xi) = [i==j] - exp(yi)
# dJ/dxi = dJ/dyi - exp(yi) Σj dJ/dyj

# function back(l::Logp, dy; returndx=true, o...)
#     @assert issimilar(dy, l.y)
#     returndx || return
#     (nd,nx) = size2(dy)
#     for j=1:nx
#         i1=(j-1)*nd+1
#         i2=j*nd
#         dysum = zero(Float64)
#         for i=i1:i2; dysum += dy[i]; end
#         for i=i1:i2; dy[i] = dy[i] - exp(l.y[i])*dysum; end
#     end
#     return dy
# end

