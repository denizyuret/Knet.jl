type Mul <: Op; alpha; beta; Mul(;alpha=1,beta=1,o...)=new(alpha,beta); end

ninputs(::Mul)=2
canoverwrite(::Mul)=false
back_reads_x(::Mul)=true
back_reads_y(::Mul)=false

function forw(l::Mul, x1, x2, y; o...)
    x2 == nothing || size(x2) == size(y) || throw(DimensionMismatch("x2 and y should have the same size in mul"))
    x1 == nothing || size(x1) == size(y) || (x1 = reshape_to_match(x1,y))
    if x1==nothing # nothing represents zero
        l.alpha == 1 || error("mul does not support nothing^p for p!=1")
        nothing
    elseif x2==nothing
        l.beta  == 1 || error("mul does not support nothing^p for p!=1")
        nothing
    else
        bmul!(l.alpha,x1,l.beta,x2,y)
    end
end


# z = mul(x,y; alpha,beta) = x^alpha * y^beta
# dJ/dx = dJ/dz dz/dx = dJ/dz * y^beta * x^(alpha-1) * alpha
# dJ/dy = dJ/dz dz/dy = dJ/dz * x^alpha * y^(beta-1) * beta

# When broadcasting:
# zi = xj^alpha * yi^beta   where multiple i map to the same j
# dJ/dyi = dJ/dzi dzi/dyi = dJ/dzi * xj^alpha * yi(beta-1) * beta   same as before, bmul! is sufficient.
# dJ/dxj = sum_i dJ/dzi dzi/dxj = sum_i dJ/dzi * yi^beta * xj^(alpha-1) * alpha
# For dJ/dzi * yi^beta mul2 is sufficient
# For sum_i we need row sums, baddback?
# For * xj^(alpha-1) * alpha we need regular mul2 and scaling

function back(l::Mul, dy, dx1, dx2; x=nothing, o...)
    (x1,x2) = (length(x)==2 ? x : error("back(mul) needs inputs x1 and x2"))
    dx2 == nothing || size(dx2) == size(dy) || throw(DimensionMismatch("x2 and y should have the same size in mul"))
    x2  == nothing || size(x2)  == size(dy) || throw(DimensionMismatch("x2 and y should have the same size in mul"))
    dx1 == nothing || size(dx1) == size(dy) || (dx1 = reshape_to_match(dx1,dy))
    x1  == nothing || size(x1)  == size(dy) || (x1  = reshape_to_match(x1,dy))
    x1  != nothing || l.alpha == 1 || error("mul does not support nothing^p for p!=1")
    x2  != nothing || l.beta  == 1 || error("mul does not support nothing^p for p!=1")

    if dx2 != nothing  # dJ/dy = dJ/dz dz/dy = dJ/dz * x^alpha * y^(beta-1) * beta
        if x1 == nothing      # representing zero
            fillsync!(dx2, 0)
        else
            bmul!(l.alpha,x1,1,dy,dx2)
            l.beta == 1 || scale!(l.beta, bmul!(l.beta-1,x2,1,dx2,dx2))
        end
    end

    if dx1 != nothing  # dJ/dxj = sum_i dJ/dzi dzi/dxj = sum_i dJ/dzi * yi^beta * xj^(alpha-1) * alpha
        if x2 == nothing
            fillsync!(dx1, 0)
        elseif size(dx1) == size(dy)
            bmul!(l.beta,x2,1,dy,dx1)
            l.alpha == 1 || scale!(l.alpha, bmul!(1,dx1,l.alpha-1,x1,dx1))
        else
            tmp = similar(dy)
            bmul!(l.beta,x2,1,dy,tmp)
            baddback!(tmp,dx1)
            free(tmp)
            l.alpha == 1 || scale!(l.alpha, bmul!(l.alpha-1,x1,1,dx1,dx1))
        end
    end
    gpusync()
end

### bmul! broadcasting multiplication: c=a^alpha * b^beta

@gpu function bmul!{T}(alpha::Number,a::CudaArray{T},beta::Number,b::CudaArray{T},c::CudaArray{T})
    size(b) == size(c) || throw(DimensionMismatch("b and c should have the same size in bmul!"))
    if size(a) == size(b)
        T <: Float32 ? ccall((:mul32,libknet),Void,(Cint,Cfloat,Ptr{Cfloat},Cfloat,Ptr{Cfloat},Ptr{Cfloat}),length(a),T(alpha),a,T(beta),b,c) :
        T <: Float64 ? ccall((:mul64,libknet),Void,(Cint,Cdouble,Ptr{Cdouble},Cdouble,Ptr{Cdouble},Ptr{Cdouble}),length(a),T(alpha),a,T(beta),b,c) :
        error("$T not supported")
    else
        ndims(b) <= 8 || error("mul kernel supports dimensions up to 8")
        a = reshape_to_match(a,b)
        T <: Float32 ? ccall((:bmul32,libknet),Void,(Cint,Cfloat,Ptr{Cint},Ptr{Cfloat},Cfloat,Ptr{Cint},Ptr{Cfloat},Ptr{Cfloat}),ndims(a),T(alpha),cudadims(a),a,T(beta),cudadims(b),b,c) :
        T <: Float64 ? ccall((:bmul64,libknet),Void,(Cint,Cdouble,Ptr{Cint},Ptr{Cdouble},Cdouble,Ptr{Cint},Ptr{Cdouble},Ptr{Cdouble}),ndims(a),T(alpha),cudadims(a),a,T(beta),cudadims(b),b,c) :
        error("$T not supported")
    end
    gpusync()
    return c
end

function cudadims(a)
    n = ndims(a)
    s = zeros(Cint,n)
    for i=1:n; s[i]=size(a,i); end
    CudaArray(s)
end


# infersize for mul is exactly the same as add except for 1-D x1 with CUDNN_ADD_SAME_C trick:
function infersize(a::Mul, x1, x2, y)
    if x1==x2==y==nothing
        nothing
    elseif x2==y==nothing
        (x1, nothing, nothing)
    elseif x2==nothing
        infersize(a, x1, y, y)
    elseif y==nothing
        infersize(a, x1, x2, x2)
    else
        ydims = map(x2, y) do xdim, ydim
            xdim == ydim ? xdim :
            xdim == 0 ? ydim :
            ydim == 0 ? xdim :
            throw(DimensionMismatch())
        end
        x1==nothing && (return (ydims, ydims, ydims))
        length(x1) <= length(ydims) || throw(DimensionMismatch())
        xdims = [x1...]; ydims = [ydims...]
        for i=1:length(xdims)
            xdims[i] == ydims[i] ? continue :
            xdims[i] == 1 ? continue :
            xdims[i] == 0 ? continue :
            ydims[i] == 0 ? (ydims[i] = xdims[i]) :
            throw(DimensionMismatch())
        end
        xdims = tuple(xdims...); ydims = tuple(ydims...)
        return (xdims, ydims, ydims)
    end
end

# function infersize(m::Mul, x1, x2, y)
#     if x1==x2==y==nothing
#         nothing
#     elseif x1==nothing || x2==nothing || y==nothing
#         n = length(x1!=nothing ? x1 : x2!=nothing ? x2 : y!=nothing ? y : error())
#         x1 == nothing && (x1 = ntuple(i->0,n))
#         x2 == nothing && (x2 = ntuple(i->0,n))
#         y == nothing && (y = ntuple(i->0,n))
#         infersize(m,x1,x2,y)
#     else
#         length(x1)==length(x2)==length(y) || throw(DimensionMismatch())
#         dims = map(x1,x2,y) do a,b,c
#             n = 0
#             a==0 || a==n ? nothing : n==0 ? n=a : throw(DimensionMismatch())
#             b==0 || b==n ? nothing : n==0 ? n=b : throw(DimensionMismatch())
#             c==0 || c==n ? nothing : n==0 ? n=c : throw(DimensionMismatch())
#             n
#         end
#         (dims, dims, dims)
#     end
# end

# ### mul2 element-wise multiplication:

# mul2!(alpha::Number,a::Array,beta::Number,b::Array,c::Array)=(for i=1:length(c); c[i] = a[i]^alpha*b[i]^beta; end; c)
# mul2!(alpha::Number,a::CudaArray{Float32},beta::Number,b::CudaArray{Float32},c::CudaArray{Float32})=(ccall((:mul2_32,libknet),Void,(Cint,Cfloat,Ptr{Cfloat},Cfloat,Ptr{Cfloat},Ptr{Cfloat}),length(a),Cfloat(alpha),a,Cfloat(beta),b,c); gpusync(); c)
# mul2!(alpha::Number,a::CudaArray{Float64},beta::Number,b::CudaArray{Float64},c::CudaArray{Float64})=(ccall((:mul2_64,libknet),Void,(Cint,Cdouble,Ptr{Cdouble},Cdouble,Ptr{Cdouble},Ptr{Cdouble}),length(a),Cdouble(alpha),a,Cdouble(beta),b,c); gpusync(); c)

# mul(x1,x2,y; alpha=1, beta=1)=(Mul(alpha,beta),x1,x2,y)
